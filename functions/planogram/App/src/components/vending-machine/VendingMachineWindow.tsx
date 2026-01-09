import type { MachineSlot } from "../../lib/types.ts"
import VendingMachineColumnHeaders from "./VendingMachineColumnHeaders.tsx";
import VendingMachineRowLabels from "./VendingMachineRowLabels.tsx";
import VendingMachineGrid from "./VendingMachineGrid.tsx";

// VENDING MACHINE WINDOW COMPONENT
interface VendingMachineWindowProps {
  rows: number
  cols: number
  slots: MachineSlot[]
  onSlotClick: (position: string) => void
  onMoveProduct?: (from: string, to: string) => void
  onDropProduct?: (product: any, to: string) => void
  onProductDragStart?: () => void
  onProductDragEnd?: () => void
}

export default function VendingMachineWindow({
  rows,
  cols,
  slots,
  onSlotClick,
  onMoveProduct,
  onDropProduct,
  onProductDragStart,
  onProductDragEnd
}: VendingMachineWindowProps) {
  const slotMap = new Map(slots.map((slot) => [slot.position, slot]))

  const getCoveredPositions = (): Set<string> => {
    const covered = new Set<string>()

    slots.forEach((slot) => {
      if (slot.width > 1) {
        const row = slot.position.charCodeAt(0) - 65
        const col = Number.parseInt(slot.position.slice(1))

        if (slot.width === 1.5) {
          // A 1.5-width slot covers its position and the next position
          // e.g., slot at A6 covers positions A6 and A7 (visually "A6-A7")
          // Mark the next position as covered ONLY if there's no slot defined there
          const nextCol = col + 1
          if (nextCol < cols) {
            const nextPos = String.fromCharCode(65 + row) + nextCol
            // Only mark as covered if no slot exists at that position
            if (!slotMap.has(nextPos)) {
              covered.add(nextPos)
            }
          }
        } else {
          // For widths 2, 3, etc., mark all covered positions
          for (let i = 1; i < slot.width; i++) {
            const coveredCol = col + i
            if (coveredCol < cols) {
              const coveredPos = String.fromCharCode(65 + row) + coveredCol
              covered.add(coveredPos)
            }
          }
        }
      }
    })

    return covered
  }

  const coveredPositions = getCoveredPositions()

  return (
    <div className="flex-shrink-0 w-full lg:flex-1 flex items-center justify-center p-0 md:p-4 transition-all duration-300">
      {/* Vending Machine Outer Casing */}
      <div className={`relative w-full max-w-full md:max-w-4xl transition-all duration-300`}>
        {/* Machine Body with 3D effect */}
        <div
          className="relative rounded-3xl overflow-hidden"
          style={{
            background: "linear-gradient(145deg, #1a1a1a 0%, #2d2d2d 50%, #1a1a1a 100%)",
            boxShadow: `
              0 50px 100px -20px rgba(0, 0, 0, 0.8),
              inset 0 1px 2px rgba(255, 255, 255, 0.1),
              inset 0 -10px 30px rgba(0, 0, 0, 0.5)
            `
          }}
        >
          {/* Main Display Window with Glass Effect */}
          <div className="relative p-3 flex">
            {/* Row Labels on the frame */}
            <div className="hidden md:block mr-2">
              <VendingMachineRowLabels rows={rows} />
            </div>

            <div className="flex-1 flex flex-col min-h-[680px] max-h-[80vh]">
              {/* Column Headers on the frame */}
              <div className="mb-2">
                <VendingMachineColumnHeaders cols={cols} />
              </div>

              <div
                className="relative rounded-xl overflow-hidden flex-1"
                style={{
                  background: "linear-gradient(135deg, rgba(0,0,0,0.9) 0%, rgba(20,20,20,0.95) 100%)",
                  boxShadow: `
                    inset 0 0 60px rgba(0, 0, 0, 0.8),
                    inset 0 4px 8px rgba(0, 0, 0, 0.6),
                    0 4px 20px rgba(0, 0, 0, 0.5)
                  `,
                  border: "3px solid #333"
                }}
              >
                {/* Glass reflection overlay */}
                <div
                  className="absolute inset-0 pointer-events-none z-10"
                  style={{
                    background: `
                      linear-gradient(
                        135deg,
                        rgba(255,255,255,0.1) 0%,
                        rgba(255,255,255,0.02) 30%,
                        rgba(255,255,255,0) 50%,
                        rgba(255,255,255,0.01) 70%,
                        rgba(255,255,255,0.05) 100%
                      )
                    `,
                    mixBlendMode: "overlay"
                  }}
                />

                {/* Inner lighting */}
                <div
                  className="absolute top-0 left-0 right-0 h-8 pointer-events-none"
                  style={{
                    background: "linear-gradient(to bottom, rgba(255,255,255,0.05), transparent)",
                  }}
                />

                {/* Product Display Area */}
                <div className="relative h-full flex flex-col overflow-hidden p-2">
                  <div className="flex-1 flex min-h-0 items-stretch overflow-hidden">
                    <div className="flex-1 overflow-auto min-h-0">
                      <div className="h-full">
                        <VendingMachineGrid
                          rows={rows}
                          cols={cols}
                          slots={slots}
                          slotMap={slotMap}
                          coveredPositions={coveredPositions}
                          onSlotClick={onSlotClick}
                          onMoveProduct={onMoveProduct}
                          onDropProduct={onDropProduct}
                          onProductDragStart={onProductDragStart}
                          onProductDragEnd={onProductDragEnd}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Side shadows for depth */}
          <div
            className="absolute inset-y-0 left-0 w-4 pointer-events-none"
            style={{
              background: "linear-gradient(to right, rgba(0,0,0,0.5), transparent)"
            }}
          />
          <div
            className="absolute inset-y-0 right-0 w-4 pointer-events-none"
            style={{
              background: "linear-gradient(to left, rgba(0,0,0,0.5), transparent)"
            }}
          />
        </div>
      </div>
    </div>
  )
}