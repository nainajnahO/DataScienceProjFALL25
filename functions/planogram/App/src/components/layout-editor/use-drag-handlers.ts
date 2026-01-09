import { useState } from "react"
import type { MergedSlot, EditorProduct } from "./LayoutEditor.tsx"

interface UseDragHandlersProps {
  localSlots: Record<number, EditorProduct>
  mergedSlots: MergedSlot[]
  isMergedSlot: (index: number) => MergedSlot | undefined
  isAdjacentToMergedGroup: (slotIndex: number, mergedGroup: MergedSlot) => boolean
  areAdjacent: (slot1: number, slot2: number) => boolean
  findExpansionNeighbor: (slot1: number, slot2: number) => number | null
  setShowDisabledMessage: (show: boolean) => void
  setConfirmationAction: (action: any) => void
  setShowConfirmation: (show: boolean) => void
}

export function useDragHandlers({
  localSlots,
  mergedSlots: _mergedSlots,
  isMergedSlot,
  isAdjacentToMergedGroup,
  areAdjacent,
  findExpansionNeighbor,
  setShowDisabledMessage,
  setConfirmationAction,
  setShowConfirmation,
}: UseDragHandlersProps) {
  const [draggedSlot, setDraggedSlot] = useState<number | null>(null)
  const [dragOverSlot, setDragOverSlot] = useState<number | null>(null)

  const handleDragStart = (e: React.DragEvent, index: number) => {
    setDraggedSlot(index)
    e.dataTransfer.effectAllowed = "move"
  }

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault()
    if (draggedSlot !== null && draggedSlot !== index) {
      setDragOverSlot(index)
    }
  }

  const handleDragLeave = () => {
    setDragOverSlot(null)
  }

  const handleDrop = (e: React.DragEvent, targetIndex: number) => {
    e.preventDefault()
    setDragOverSlot(null)

    if (draggedSlot === null || draggedSlot === targetIndex) {
      setDraggedSlot(null)
      return
    }

    const draggedMerged = isMergedSlot(draggedSlot)
    const targetMerged = isMergedSlot(targetIndex)

    let isAdjacent = false

    if (!draggedMerged && !targetMerged) {
      isAdjacent = areAdjacent(draggedSlot, targetIndex)
    } else if (draggedMerged && !targetMerged) {
      isAdjacent = isAdjacentToMergedGroup(targetIndex, draggedMerged)
    } else if (!draggedMerged && targetMerged) {
      isAdjacent = isAdjacentToMergedGroup(draggedSlot, targetMerged)
    }

    if (!isAdjacent) {
      setDraggedSlot(null)
      return
    }

    const draggedHasProduct = draggedMerged ? !!draggedMerged.product : !!localSlots[draggedSlot]
    const targetHasProduct = targetMerged ? !!targetMerged.product : !!localSlots[targetIndex]

    if (draggedHasProduct || targetHasProduct) {
      setDraggedSlot(null)
      setShowDisabledMessage(true)
      return
    }

    const canMerge =
      (!draggedMerged && !targetMerged) ||
      (draggedMerged && draggedMerged.slots.length < 3 && !targetMerged) ||
      (targetMerged && targetMerged.slots.length < 3 && !draggedMerged)

    if (!canMerge) {
      setDraggedSlot(null)
      return
    }

    const resultingSlotsCount = (draggedMerged?.slots.length || 1) + (targetMerged?.slots.length || 1)

    if (resultingSlotsCount === 3) {
      setConfirmationAction({
        type: "create-triple-or-pair",
        data: { draggedSlot, targetIndex },
      })
      setShowConfirmation(true)
    } else if (resultingSlotsCount === 2) {
      const extension = findExpansionNeighbor(draggedSlot, targetIndex)
      setConfirmationAction({
        type: "merge-2-opt-3",
        data: { draggedSlot, targetIndex, extension },
      })
      setShowConfirmation(true)
    }

    setDraggedSlot(null)
  }

  return {
    draggedSlot,
    dragOverSlot,
    handleDragStart,
    handleDragOver,
    handleDragLeave,
    handleDrop,
  }
}

