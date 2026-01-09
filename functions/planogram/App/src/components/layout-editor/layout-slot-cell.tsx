import type React from "react"

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

interface LayoutSlotCellProps {
  index: number
  merged: MergedSlot | undefined
  product: Product | undefined
  position: string
  gridColumnStart: number
  span: number
  is1_5Width: boolean
  isSelected: boolean
  isDragging: boolean
  isValidDropTarget: boolean
  onSlotClick: (index: number) => void
  onDragStart: (e: React.DragEvent, index: number) => void
  onDragOver: (e: React.DragEvent, index: number) => void
  onDragLeave: () => void
  onDrop: (e: React.DragEvent, index: number) => void
}

export default function LayoutSlotCell({
  index,
  merged,
  product,
  position,
  gridColumnStart,
  span,
  is1_5Width,
  isSelected,
  isDragging,
  isValidDropTarget,
  onSlotClick,
  onDragStart,
  onDragOver,
  onDragLeave,
  onDrop,
}: LayoutSlotCellProps) {
  const hasProduct = !!product
  const isTriple = merged && merged.slots.length === 3

  const colorClass = is1_5Width
    ? "bg-orange-100 border-orange-300 text-orange-900 dark:bg-orange-900/30 dark:text-orange-100 dark:border-orange-800 hover:bg-orange-200 dark:hover:bg-orange-900/50"
    : isTriple
      ? "bg-fuchsia-100 border-fuchsia-300 text-fuchsia-900 dark:bg-fuchsia-900/30 dark:text-fuchsia-100 dark:border-fuchsia-800 hover:bg-fuchsia-200 dark:hover:bg-fuchsia-900/50"
      : merged
        ? "bg-indigo-100 border-indigo-300 text-indigo-900 dark:bg-indigo-900/30 dark:text-indigo-100 dark:border-indigo-800 hover:bg-indigo-200 dark:hover:bg-indigo-900/50"
        : hasProduct
          ? "bg-emerald-100 border-emerald-300 text-emerald-900 dark:bg-emerald-900/30 dark:text-emerald-100 dark:border-emerald-800 hover:bg-emerald-200 dark:hover:bg-emerald-900/50"
          : "bg-background border-border text-muted-foreground hover:bg-accent/50 hover:text-foreground"

  return (
    <button
      key={index}
      draggable={true}
      onDragStart={(e) => onDragStart(e, index)}
      onDragOver={(e) => onDragOver(e, index)}
      onDragLeave={onDragLeave}
      onDrop={(e) => onDrop(e, index)}
      onClick={() => onSlotClick(index)}
      style={{
        gridColumn: `${gridColumnStart} / span ${span}`,
      }}
      className={`
        h-full rounded-md border transition-all duration-200
        flex flex-col items-center justify-center text-xs font-medium
        hover:shadow-md cursor-grab
        ${isDragging
          ? "opacity-50 scale-95"
          : isValidDropTarget
            ? "ring-2 ring-blue-500 bg-blue-50 dark:bg-blue-900/20 border-blue-500"
            : isSelected
              ? "ring-2 ring-primary border-primary shadow-lg z-10"
              : colorClass
        }
      `}
      title={
        merged
          ? `${is1_5Width ? "1.5-width" : "Merged"} slot: ${position}${hasProduct ? ` - ${product.name}` : ""} (Click to ${merged.slots.length === 3 ? "split with options" : "unmerge"})`
          : hasProduct
            ? product.name
            : is1_5Width
              ? "1.5-width slot (empty, paired with previous 1.5-width slot)"
              : "Empty slot"
      }
    >
      <span className="font-bold">{position}</span>
      {hasProduct && product.image && (
        <img
          src={product.image || "/placeholder.svg"}
          alt={product.name}
          className="w-8 h-8 object-contain mt-1 pointer-events-none"
        />
      )}
    </button>
  )
}

