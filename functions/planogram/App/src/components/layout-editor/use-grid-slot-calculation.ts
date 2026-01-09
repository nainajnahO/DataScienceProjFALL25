import type { MergedSlot } from "./LayoutEditor.tsx"

interface UseGridSlotCalculationProps {
  ROWS: number
  COLS: number
  mergedSlots: MergedSlot[]
  isMergedSlot: (index: number) => MergedSlot | undefined
  isFirstOfAnyMergedGroup: (index: number) => boolean
}

export function useGridSlotCalculation({
  ROWS: _ROWS,
  COLS,
  mergedSlots: _mergedSlots,
  isMergedSlot,
  isFirstOfAnyMergedGroup,
}: UseGridSlotCalculationProps) {
  const getPositionLabel = (index: number) => {
    const row = Math.floor(index / COLS)
    const col = index % COLS
    return `${String.fromCharCode(65 + row)}${col}`
  }

  const getGridColumnSpan = (merged: MergedSlot | undefined) => {
    if (!merged) return 2
    if (merged.width === 1.5) return 3
    if (merged.slots.length === 2) return 4
    if (merged.slots.length === 3) return 6
    return 2
  }

  const calculateRowSlots = (rowIndex: number) => {
    const rowStartIndex = rowIndex * COLS
    const rowEndIndex = rowStartIndex + COLS

    const coveredByOnePointFive = new Set<number>()
    for (let i = rowStartIndex; i < rowEndIndex; i++) {
      const merged = isMergedSlot(i)
      if (merged?.width === 1.5 && isFirstOfAnyMergedGroup(i)) {
        const nextIndex = i + 1
        const nextMerged = isMergedSlot(nextIndex)
        if (!nextMerged) {
          coveredByOnePointFive.add(nextIndex)
        }
      }
    }

    const slotsToRender: Array<{ index: number; gridColumnStart: number; span: number; is1_5Width: boolean }> = []
    let currentGridColumn = 2

    for (let i = rowStartIndex; i < rowEndIndex; i++) {
      const merged = isMergedSlot(i)

      if (merged && !isFirstOfAnyMergedGroup(i)) {
        continue
      }

      if (coveredByOnePointFive.has(i) && !merged) {
        continue
      }

      const is1_5Width = merged?.width === 1.5
      const span = is1_5Width ? 3 : getGridColumnSpan(merged)

      slotsToRender.push({
        index: i,
        gridColumnStart: currentGridColumn,
        span: span,
        is1_5Width: is1_5Width,
      })

      currentGridColumn += span
    }

    return slotsToRender
  }

  return {
    getPositionLabel,
    calculateRowSlots,
  }
}

