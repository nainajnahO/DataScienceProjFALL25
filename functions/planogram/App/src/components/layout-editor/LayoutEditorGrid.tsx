import type React from "react"
import { useEffect, useState } from "react"
import LayoutSlotCell from "./layout-slot-cell.tsx"
import { useGridSlotCalculation } from "./use-grid-slot-calculation.ts"

interface Product {
  id: string
  name: string
  price: number
  category: string
  image?: string
}

interface MergedSlot {
  slots: number[]
  product?: Product
  width?: number
}

interface LayoutEditorGridProps {
  rows: number
  cols: number
  localSlots: Record<number, Product>
  mergedSlots: MergedSlot[]
  firstSelectedSlot: number | null
  draggedSlot: number | null
  dragOverSlot: number | null
  onSlotClick: (index: number) => void
  onDragStart: (e: React.DragEvent, index: number) => void
  onDragOver: (e: React.DragEvent, index: number) => void
  onDragLeave: () => void
  onDrop: (e: React.DragEvent, index: number) => void
  isMergedSlot: (index: number) => MergedSlot | undefined
  isFirstOfAnyMergedGroup: (index: number) => boolean
  areAdjacent: (slot1: number, slot2: number) => boolean
}

export default function LayoutEditorGrid({
  rows: ROWS,
  cols: COLS,
  localSlots,
  mergedSlots,
  firstSelectedSlot,
  draggedSlot,
  dragOverSlot,
  onSlotClick,
  onDragStart,
  onDragOver,
  onDragLeave,
  onDrop,
  isMergedSlot,
  isFirstOfAnyMergedGroup,
  areAdjacent,
}: LayoutEditorGridProps) {
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined') return
    const checkMobile = () => setIsMobile(window.innerWidth < 768)
    checkMobile()
    window.addEventListener("resize", checkMobile)
    return () => window.removeEventListener("resize", checkMobile)
  }, [])

  const { getPositionLabel, calculateRowSlots } = useGridSlotCalculation({
    ROWS,
    COLS,
    mergedSlots,
    isMergedSlot,
    isFirstOfAnyMergedGroup,
  })

  return (
    <div className="bg-card rounded-lg shadow-sm border border-border p-4 md:p-6 mb-4">
      <div className="overflow-x-clip">
        <div className="inline-block min-w-full">
          <div
            className="grid gap-2"
            style={{
              gridTemplateColumns: isMobile ? `0 repeat(20, 1fr)` : `auto repeat(20, 1fr)`,
              gridTemplateRows: isMobile ? `0 repeat(${ROWS}, minmax(80px, 1fr))` : `auto repeat(${ROWS}, minmax(80px, 1fr))`,
            }}
          >
            <div className="hidden md:block w-8"></div>
            {Array.from({ length: COLS }).map((_, colIndex) => (
                <div
                  key={`header-${colIndex}`}
                  className="hidden md:flex items-center justify-center text-center text-sm font-medium text-muted-foreground pb-2"
                  style={{ gridColumn: "span 2" }}
                >
                  {colIndex}
                </div>
            ))}

            {Array.from({ length: ROWS }).map((_, rowIndex) => {
              const slotsToRender = calculateRowSlots(rowIndex)

              const rowSlots = slotsToRender.map(({ index, gridColumnStart, span, is1_5Width }) => {
                const merged = isMergedSlot(index)
                const product = merged ? merged.product : localSlots[index]

                let position: string
                if (merged?.width === 1.5) {
                  const currentLabel = getPositionLabel(index)
                  const nextLabel = getPositionLabel(index + 1)
                  position = `${currentLabel}-${nextLabel}`
                } else if (merged) {
                  position = merged.slots.map((s) => getPositionLabel(s)).join("-")
                } else {
                  position = getPositionLabel(index)
                }

                const isSelected = firstSelectedSlot === index
                const isDragging = draggedSlot === index
                const isValidDropTarget = dragOverSlot === index && draggedSlot !== null && areAdjacent(draggedSlot, index)

                return (
                  <LayoutSlotCell
                    key={index}
                    index={index}
                    merged={merged}
                    product={product}
                    position={position}
                    gridColumnStart={gridColumnStart}
                    span={span}
                    is1_5Width={is1_5Width}
                    isSelected={isSelected}
                    isDragging={isDragging}
                    isValidDropTarget={isValidDropTarget}
                    onSlotClick={onSlotClick}
                    onDragStart={onDragStart}
                    onDragOver={onDragOver}
                    onDragLeave={onDragLeave}
                    onDrop={onDrop}
                  />
                )
              })

              return (
                <>
                  <div
                    key={`row-label-${rowIndex}`}
                    className="hidden md:flex items-center justify-center text-sm font-medium text-muted-foreground pr-2"
                    style={{ gridRow: rowIndex + 2, gridColumn: 1 }}
                  >
                    {String.fromCharCode(65 + rowIndex)}
                  </div>
                  {rowSlots}
                </>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
