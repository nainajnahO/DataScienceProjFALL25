import type { MergedSlot, EditorProduct } from "./LayoutEditor.tsx"

interface UseMergeLogicProps {
  mergedSlots: MergedSlot[]
  localSlots: Record<number, EditorProduct>
  setMergedSlots: React.Dispatch<React.SetStateAction<MergedSlot[]>>
  setLocalSlots: React.Dispatch<React.SetStateAction<Record<number, EditorProduct>>>
  isMergedSlot: (index: number) => MergedSlot | undefined
  ROWS: number
  COLS: number
}

export function useMergeLogic({
  mergedSlots,
  localSlots,
  setMergedSlots,
  setLocalSlots,
  isMergedSlot,
  ROWS: _ROWS,
  COLS: _COLS,
}: UseMergeLogicProps) {
  const createMerge = (draggedSlot: number, targetIndex: number, width?: number) => {
    const draggedMerged = isMergedSlot(draggedSlot)
    const targetMerged = isMergedSlot(targetIndex)
    let newSlots: number[]
    const mergesToRemove: MergedSlot[] = []

    if (draggedMerged && !targetMerged) {
      newSlots = [...draggedMerged.slots, targetIndex].sort((a, b) => a - b)
      mergesToRemove.push(draggedMerged)
    } else if (targetMerged && !draggedMerged) {
      newSlots = [...targetMerged.slots, draggedSlot].sort((a, b) => a - b)
      mergesToRemove.push(targetMerged)
    } else {
      newSlots = [draggedSlot, targetIndex].sort((a, b) => a - b)
    }

    newSlots.forEach((slotIndex) => {
      const existingMerge = mergedSlots.find((m) => m.slots.includes(slotIndex) && !mergesToRemove.includes(m))
      if (existingMerge) mergesToRemove.push(existingMerge)
    })

    const newMerged: MergedSlot = {
      slots: newSlots,
      product: undefined,
      width: width
    }

    setMergedSlots((prev) => {
      const filtered = prev.filter((m) => !mergesToRemove.includes(m))
      return [...filtered, newMerged]
    })

    const updatedSlots = { ...localSlots }
    newSlots.forEach((slot) => delete updatedSlots[slot])
    setLocalSlots(updatedSlots)
  }

  const applyPair1_5 = (allSlots: number[]) => {
    if (allSlots.length !== 3) return

    const mergesToRemove: MergedSlot[] = []
    allSlots.forEach((slotIndex) => {
      const existingMerge = mergedSlots.find((m) => m.slots.includes(slotIndex) && !mergesToRemove.includes(m))
      if (existingMerge) mergesToRemove.push(existingMerge)
    })

    const [s1, s2, s3] = allSlots
    const pair1: MergedSlot = { slots: [s1, s2], width: 1.5 }
    const pair2: MergedSlot = { slots: [s2, s3], width: 1.5 }

    setMergedSlots((prev) => {
      const filtered = prev.filter((m) => !mergesToRemove.includes(m))
      return [...filtered, pair1, pair2]
    })

    const updatedSlots = { ...localSlots }
    allSlots.forEach((slot) => delete updatedSlots[slot])
    setLocalSlots(updatedSlots)
  }

  const createPair1_5 = (draggedSlot: number, targetIndex: number) => {
    const draggedMerged = isMergedSlot(draggedSlot)
    const targetMerged = isMergedSlot(targetIndex)
    let allSlots: number[] = []

    if (draggedMerged && !targetMerged) {
      allSlots = [...draggedMerged.slots, targetIndex].sort((a, b) => a - b)
    } else if (targetMerged && !draggedMerged) {
      allSlots = [...targetMerged.slots, draggedSlot].sort((a, b) => a - b)
    } else {
      allSlots = [draggedSlot, targetIndex].sort((a, b) => a - b)
    }

    applyPair1_5(allSlots)
  }

  const createPair1_5_From3 = (slot1: number, slot2: number, slot3: number) => {
    const allSlots = [slot1, slot2, slot3].sort((a, b) => a - b)
    applyPair1_5(allSlots)
  }

  const splitMergedSlot = (index: number) => {
    setMergedSlots((prev) => prev.filter((m) => !m.slots.includes(index)))
  }

  return {
    createMerge,
    createPair1_5,
    createPair1_5_From3,
    splitMergedSlot,
  }
}

