import React, { useState } from "react"
import type { MachineSlot } from "../../lib/types.ts"
import SpiralBackground from "../../public/Spilar_Background.png"
import SpiralForeground from "../../public/Spilar_foreground.png"
import { useImageRotation } from "./use-image-rotation.ts"

interface SlotCellProps {
  position: string
  slot: MachineSlot | undefined
  shouldRenderAsOnePointFive: boolean
  flexValue: number
  isLastInRow: boolean
  onSlotClick: (position: string) => void
  onMoveProduct?: (from: string, to: string) => void
  onDropProduct?: (product: any, to: string) => void
  onDragStart?: () => void
  onDragEnd?: () => void
}

const SlotCell: React.FC<SlotCellProps> = ({
    position,
    slot,
    shouldRenderAsOnePointFive,
    flexValue,
    isLastInRow,
    onSlotClick,
    onMoveProduct,
  onDropProduct,
    onDragStart,
    onDragEnd
}) => {
  const hasProduct = !!slot?.product_name
  const [isDragOver, setIsDragOver] = useState(false)
  const containerRef = React.useRef<HTMLDivElement>(null)
  const dragTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  const { displayImageUrl, imageError, setImageError } = useImageRotation({
    imageUrl: slot?.image_url,
    hasProduct,
    slotWidth: slot?.width || 1
  })

    React.useEffect(() => {
        if (containerRef.current) {
      containerRef.current.style.opacity = "1"
        }
  }, [slot])

  const renderSpirals = (isForeground: boolean) => {
    const slotWidth = slot?.width || 1
    const spiralCount = (slotWidth === 1 || slotWidth === 1.5 || shouldRenderAsOnePointFive) ? 1 : 2
    const spiralSize = (slotWidth === 1.5 || shouldRenderAsOnePointFive) ? 'w-[65%]' : slotWidth === 1 ? 'w-[90%]' : slotWidth === 3 ? 'w-[35%]' : 'w-[50%]'
    const SpiralImage = isForeground ? SpiralForeground : SpiralBackground
    const zIndex = isForeground ? 15 : 5

    return (
      <>
        {Array.from({ length: spiralCount }).map((_, idx) => {
          const isLeft = idx === 0
          const positionClass = spiralCount === 1
            ? "left-1/2 -translate-x-1/2"
            : isLeft ? "left-0" : "right-0"

          return isForeground ? (
            <img
              key={`spiral-fg-${idx}`}
              src={SpiralImage}
              alt=""
              className={`absolute ${spiralSize} h-auto object-contain ${positionClass} md:block hidden pointer-events-none`}
              style={{ zIndex, bottom: '-10%' }}
            />
          ) : (
            <React.Fragment key={`spiral-${idx}`}>
              <img
                src={SpiralImage}
                alt=""
                className={`absolute ${spiralSize} h-auto object-contain ${positionClass} md:block hidden pointer-events-none`}
                style={{ zIndex, bottom: '-10%' }}
              />
            </React.Fragment>
          )
        })}
      </>
    )
  }

    return (
        <div
            key={`slot-${position}`}
            onClick={() => onSlotClick(position)}
      className={`group cursor-pointer relative flex md:items-end items-center justify-center md:min-h-[50px] min-h-[90px] ${!isLastInRow ? 'md:border-r border-r-2 md:border-gray-700 border-gray-800' : ''} ${isDragOver ? 'ring-2 ring-blue-500 ring-inset bg-blue-500/20' : ''}`}
            style={{
                gridColumn: `span ${shouldRenderAsOnePointFive ? 3 : (slot?.width ? (slot.width === 2 ? 4 : slot.width === 3 ? 6 : 2) : 2)}`,
                minWidth: 0,
                flex: `${flexValue} ${flexValue} 0%`,
                flexBasis: 0,
            }}
            title={hasProduct ? slot.product_name : `Click to add product (${position})`}
      onDragEnter={(e) => {
        e.preventDefault()
        e.stopPropagation()
        // Check if dragging a product from sidebar
        if (e.dataTransfer.types.includes("application/json")) {
          setIsDragOver(true)
          e.dataTransfer.dropEffect = "copy"
        } else if (e.dataTransfer.types.includes("text/plain")) {
          setIsDragOver(true)
          e.dataTransfer.dropEffect = "move"
        }
      }}
      onDragOver={(e) => {
        e.preventDefault()
        e.stopPropagation()
        // Check if dragging a product from sidebar
        if (e.dataTransfer.types.includes("application/json")) {
          e.dataTransfer.dropEffect = "copy"
        } else {
          e.dataTransfer.dropEffect = "move"
        }
      }}
      onDragLeave={(e) => {
        e.preventDefault()
        e.stopPropagation()
        // Only clear if we're leaving the slot entirely (not just moving to a child)
        const rect = e.currentTarget.getBoundingClientRect()
        const x = e.clientX
        const y = e.clientY
        if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
          setIsDragOver(false)
        }
      }}
      onDrop={(e) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragOver(false)
        // Check if it's a product drop from sidebar
        const productData = e.dataTransfer.getData("application/json")
        if (productData) {
          try {
            const data = JSON.parse(productData)
            if (data.type === "product" && data.product && onDropProduct) {
              onDropProduct(data.product, position)
              return
            }
          } catch (err) {
            // Not valid JSON, continue to position check
          }
        }
        // Check if it's a position move from another slot
        const fromPosition = e.dataTransfer.getData("text/plain")
        if (fromPosition && fromPosition !== position && onMoveProduct) {
          onMoveProduct(fromPosition, position)
        }
      }}
    >
      {renderSpirals(false)}

            <div
                ref={containerRef}
                className={`relative w-full h-full flex md:items-end items-center justify-center ${hasProduct ? 'cursor-grab active:cursor-grabbing' : ''}`}
                style={{ zIndex: 10 }}
                draggable={hasProduct}
                onDragStart={(e) => {
                    if (hasProduct) {
            e.dataTransfer.setData("text/plain", position)
            e.dataTransfer.effectAllowed = "move"

                        dragTimeoutRef.current = setTimeout(() => {
                            if (containerRef.current) {
                containerRef.current.style.opacity = "0"
                            }
              if (onDragStart) onDragStart()
            }, 0)
                    }
                }}
                onDragEnd={() => {
                    if (dragTimeoutRef.current) {
            clearTimeout(dragTimeoutRef.current)
            dragTimeoutRef.current = null
                    }
                    if (containerRef.current) {
            containerRef.current.style.opacity = "1"
                    }
          if (onDragEnd) onDragEnd()
                }}
            >
                {hasProduct && displayImageUrl && !imageError ? (
                    <>
                        <div
                            className="absolute md:bottom-1 bottom-1 w-[70%] h-3 rounded-full opacity-30 blur-md pointer-events-none"
                            style={{
                                background: "radial-gradient(ellipse, rgba(0,0,0,0.8), transparent 70%)"
                            }}
                        />
                        <img
                            src={displayImageUrl}
                            alt={slot.product_name}
                            onError={() => setImageError(true)}
                            className="w-[85%] max-h-[85%] object-contain relative z-10 md:mb-1 mb-1 group-hover:scale-110 group-hover:brightness-125 transition-[transform,filter] duration-300 pointer-events-none"
                            style={{
                                filter: "drop-shadow(0 4px 8px rgba(0,0,0,0.6))"
                            }}
                        />
                        {slot.stock_current === 0 ? (
                            <div
                                className="hidden md:block absolute top-1 left-1 px-1.5 py-0.5 rounded text-xs font-bold text-white z-20 pointer-events-none"
                                style={{
                                    background: "linear-gradient(135deg, #991b1b, #7f1d1d)",
                                    boxShadow: "0 2px 4px rgba(0,0,0,0.3)"
                                }}
                            >
                                OUT
                            </div>
                        ) : slot.stock_current <= 2 && slot.stock_current > 0 && (
                            <div
                                className="hidden md:block absolute top-1 left-1 px-1.5 py-0.5 rounded text-xs font-bold text-white z-20 pointer-events-none"
                                style={{
                                    background: "linear-gradient(135deg, #ef4444, #dc2626)",
                                    boxShadow: "0 2px 4px rgba(0,0,0,0.3)"
                                }}
                            >
                                LOW
                            </div>
                        )}
                    </>
                ) : hasProduct ? (
                    <div
                        className="w-[70%] h-[70%] rounded-lg border-2 border-gray-600 flex items-center justify-center transition-all duration-200 group-hover:border-gray-500 pointer-events-none bg-gray-800"
                        style={{
                            background: "linear-gradient(to bottom right, #1f2937, #111827)",
                            boxShadow: "0 4px 8px rgba(0,0,0,0.5), inset 0 2px 4px rgba(255,255,255,0.05)"
                        }}
                    >
                        <span className="font-bold text-gray-500 text-2xl">
                            {slot.product_name.substring(0, 1)}
                        </span>
                    </div>
                ) : (
                    <div
                        className="md:w-[60%] w-[70%] md:aspect-square md:rounded-full md:border-0 border-2 border-dashed border-gray-600 flex items-center justify-center transition-all duration-200 md:mb-0 mb-0 md:h-auto h-[60%] rounded-lg pointer-events-none"
                    >
                        <span className="md:text-gray-700 text-gray-600 text-xs md:text-base">+</span>
                    </div>
                )}
            </div>

      {renderSpirals(true)}
        </div>
  )
}

export default SlotCell
