import type { MergedSlot } from "./LayoutEditor.tsx"

export function useSlotAdjacency(COLS: number) {
  const areAdjacent = (slot1: number, slot2: number) => {
    const row1 = Math.floor(slot1 / COLS)
    const row2 = Math.floor(slot2 / COLS)
    const col1 = slot1 % COLS
    const col2 = slot2 % COLS
    return row1 === row2 && Math.abs(col1 - col2) === 1
  }

  const isAdjacentToMergedGroup = (slotIndex: number, mergedGroup: MergedSlot) => {
    return mergedGroup.slots.some((mergedSlotIndex) => areAdjacent(slotIndex, mergedSlotIndex))
  }

  const isExpansionTargetValid = (index: number, ROWS: number, localSlots: Record<number, any>, isMergedSlot: (index: number) => MergedSlot | undefined) => {
    if (index < 0 || index >= ROWS * COLS) return false
    return !localSlots[index] && !isMergedSlot(index)
  }

  const findExpansionNeighbor = (slot1: number, slot2: number, ROWS: number, localSlots: Record<number, any>, isMergedSlot: (index: number) => MergedSlot | undefined) => {
    const min = Math.min(slot1, slot2)
    const max = Math.max(slot1, slot2)

    if (areAdjacent(max, max + 1) && isExpansionTargetValid(max + 1, ROWS, localSlots, isMergedSlot)) {
      return max + 1
    }

    if (areAdjacent(min, min - 1) && isExpansionTargetValid(min - 1, ROWS, localSlots, isMergedSlot)) {
      return min - 1
    }
    return null
  }

  return {
    areAdjacent,
    isAdjacentToMergedGroup,
    findExpansionNeighbor,
  }
}

