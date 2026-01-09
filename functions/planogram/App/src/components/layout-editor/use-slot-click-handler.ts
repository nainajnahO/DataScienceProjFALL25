import { useState } from "react"
import type { MergedSlot, EditorProduct } from "./LayoutEditor.tsx"

interface UseSlotClickHandlerProps {
  mergedSlots: MergedSlot[]
  localSlots: Record<number, EditorProduct>
  isMergedSlot: (index: number) => MergedSlot | undefined
  areAdjacent: (slot1: number, slot2: number) => boolean
  isAdjacentToMergedGroup: (slotIndex: number, mergedGroup: MergedSlot) => boolean
  findExpansionNeighbor: (slot1: number, slot2: number) => number | null
  setShowDisabledMessage: (show: boolean) => void
  setConfirmationAction: (action: any) => void
  setShowConfirmation: (show: boolean) => void
}

export function useSlotClickHandler({
  mergedSlots,
  localSlots,
  isMergedSlot,
  areAdjacent,
  isAdjacentToMergedGroup,
  findExpansionNeighbor,
  setShowDisabledMessage,
  setConfirmationAction,
  setShowConfirmation,
}: UseSlotClickHandlerProps) {
  const [firstSelectedSlot, setFirstSelectedSlot] = useState<number | null>(null)

  const handleSlotClick = (index: number) => {
    const exactStartMerge = mergedSlots.find(m => m.slots[0] === index)
    const containingMerge = isMergedSlot(index)
    const clickedMerged = exactStartMerge || containingMerge

    if (clickedMerged && clickedMerged.slots[0] === index) {
      if (clickedMerged.product) {
        setShowDisabledMessage(true)
        return
      }

      const splitType = clickedMerged.slots.length === 3 ? "split-triple" : (clickedMerged.width === 1.5 ? "split-pair" : "split")
      setConfirmationAction({
        type: splitType,
        data: { index: index, merged: clickedMerged },
      })
      setShowConfirmation(true)
    } else if (firstSelectedSlot === null) {
      setFirstSelectedSlot(index)
    } else if (firstSelectedSlot === index) {
      setFirstSelectedSlot(null)
    } else {
      const firstMerged = mergedSlots.find(m => m.slots[0] === firstSelectedSlot) || isMergedSlot(firstSelectedSlot)
      const targetMerged = exactStartMerge || containingMerge
      let isAdjacent = false

      if (!firstMerged && !targetMerged) {
        isAdjacent = areAdjacent(firstSelectedSlot, index)
      } else if (firstMerged && !targetMerged) {
        isAdjacent = isAdjacentToMergedGroup(index, firstMerged)
      } else if (!firstMerged && targetMerged) {
        isAdjacent = isAdjacentToMergedGroup(firstSelectedSlot, targetMerged)
      }

      if (isAdjacent) {
        const firstHasProduct = firstMerged ? !!firstMerged.product : !!localSlots[firstSelectedSlot]
        const targetHasProduct = targetMerged ? !!targetMerged.product : !!localSlots[index]

        if (firstHasProduct || targetHasProduct) {
          setShowDisabledMessage(true)
          setFirstSelectedSlot(null)
        } else {
          const canMerge =
            (!firstMerged && !targetMerged) ||
            (firstMerged && firstMerged.slots.length < 3 && !targetMerged) ||
            (targetMerged && targetMerged.slots.length < 3 && !firstMerged)

          if (canMerge) {
            const resultingSlotsCount = (firstMerged?.slots.length || 1) + (targetMerged?.slots.length || 1)

            if (resultingSlotsCount === 3) {
              setConfirmationAction({
                type: "create-triple-or-pair",
                data: { draggedSlot: firstSelectedSlot, targetIndex: index },
              })
            } else if (resultingSlotsCount === 2) {
              const extension = findExpansionNeighbor(firstSelectedSlot, index)
              setConfirmationAction({
                type: "merge-2-opt-3",
                data: { draggedSlot: firstSelectedSlot, targetIndex: index, extension },
              })
            }
            setShowConfirmation(true)
            setFirstSelectedSlot(null)
          } else {
            setFirstSelectedSlot(index)
          }
        }
      } else {
        setFirstSelectedSlot(index)
      }
    }
  }

  return {
    firstSelectedSlot,
    handleSlotClick,
  }
}

