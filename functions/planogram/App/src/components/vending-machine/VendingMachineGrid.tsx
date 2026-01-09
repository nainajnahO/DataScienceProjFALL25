import React from "react";
import { indexToPosition, getGridColumnSpan } from "./vending-machine-utils.ts";
import type { MachineSlot } from "../../lib/types.ts";
import SlotCell from "./SlotCell.tsx";
import ShelfRow from "./ShelfRow.tsx";

interface VendingMachineGridProps {
    rows: number;
    cols: number;
    slots: MachineSlot[];
    slotMap: Map<string, MachineSlot>;
    coveredPositions: Set<string>;
    onSlotClick: (position: string) => void;
    onMoveProduct?: (from: string, to: string) => void;
    onDropProduct?: (product: any, to: string) => void;
    onProductDragStart?: () => void;
    onProductDragEnd?: () => void;
}

const VendingMachineGrid: React.FC<VendingMachineGridProps> = ({
    rows,
    cols,
    slotMap,
    coveredPositions,
    onSlotClick,
    onMoveProduct,
    onDropProduct,
    onProductDragStart,
    onProductDragEnd
}) => {
    return (
        <div className="w-full h-full flex flex-col md:bg-transparent bg-[#0a0a0f]">
            {Array.from({ length: rows }).map((_, rowIdx) => {
                // Row Slots (grid)
                const rowStart = rowIdx * cols;

                // Helper function to calculate flex value for a slot position
                const calculateSlotFlexValue = (colIdx: number): number => {
                    const index = rowStart + colIdx;
                    const position = indexToPosition(index, cols);
                    if (coveredPositions.has(position)) return 0;

                    const slot = slotMap.get(position);
                    const row = position.charCodeAt(0) - 65;
                    const col = Number.parseInt(position.slice(1));

                    // Check if paired with col-2 (we are second in pair)
                    const prevPairPos = col >= 2 ? String.fromCharCode(65 + row) + (col - 2) : null;
                    const prevPairSlot = prevPairPos ? slotMap.get(prevPairPos) : null;

                    // Check if paired with col+2 (we are first in pair)
                    const nextPairPos = String.fromCharCode(65 + row) + (col + 2);
                    const nextPairSlot = slotMap.get(nextPairPos);

                    // A slot should be 1.5-width if:
                    // 1. It has width 1.5 explicitly set, AND
                    // 2. It's actually paired with another 1.5-width at col±2
                    const isFirstInPair = slot?.width === 1.5 && nextPairSlot && nextPairSlot.width === 1.5;
                    const isSecondInPair = slot?.width === 1.5 && prevPairSlot && prevPairSlot.width === 1.5;
                    const shouldRenderAsOnePointFive = Boolean(isFirstInPair || isSecondInPair);

                    return shouldRenderAsOnePointFive ? 3 : (slot?.width ? getGridColumnSpan(slot.width) : 2);
                };

                // Store flex values for shelf alignment
                const slotFlexValues: number[] = [];

                const productCells = Array.from({ length: cols }).map((_, colIdx) => {
                    const index = rowStart + colIdx;
                    const position = indexToPosition(index, cols);
                    if (coveredPositions.has(position)) return null;
                    const slot = slotMap.get(position);

                    // Use shared calculation function
                    const flexValue = calculateSlotFlexValue(colIdx);
                    slotFlexValues.push(flexValue); // Store for shelf

                    // Determine if this slot should render as 1.5-width
                    const row = position.charCodeAt(0) - 65;
                    const col = Number.parseInt(position.slice(1));

                    // Check if paired with col-2 (we are second in pair)
                    const prevPairPos = col >= 2 ? String.fromCharCode(65 + row) + (col - 2) : null;
                    const prevPairSlot = prevPairPos ? slotMap.get(prevPairPos) : null;

                    // Check if paired with col+2 (we are first in pair)
                    const nextPairPos = String.fromCharCode(65 + row) + (col + 2);
                    const nextPairSlot = slotMap.get(nextPairPos);

                    // A slot should be 1.5-width if:
                    // 1. It has width 1.5 explicitly set, AND
                    // 2. It's actually paired with another 1.5-width at col±2
                    const isFirstInPair = slot?.width === 1.5 && nextPairSlot && nextPairSlot.width === 1.5;
                    const isSecondInPair = slot?.width === 1.5 && prevPairSlot && prevPairSlot.width === 1.5;
                    const shouldRenderAsOnePointFive = Boolean(isFirstInPair || isSecondInPair);

                    // Check if this is the last non-covered position
                    const isLastInRow = colIdx === cols - 1 || Array.from({ length: cols - colIdx - 1 }).every((_, i) => {
                        const futureIndex = rowStart + colIdx + i + 1;
                        const futurePosition = indexToPosition(futureIndex, cols);
                        return coveredPositions.has(futurePosition);
                    });

                    return (
                        <SlotCell
                            key={`slot-${position}`}
                            position={position}
                            slot={slot}
                            shouldRenderAsOnePointFive={shouldRenderAsOnePointFive}
                            flexValue={flexValue}
                            isLastInRow={isLastInRow}
                            onSlotClick={onSlotClick}
                            onMoveProduct={onMoveProduct}
                            onDropProduct={onDropProduct}
                            onDragStart={onProductDragStart}
                            onDragEnd={onProductDragEnd}
                        />
                    );
                });

                // Filter out null values (covered positions) to avoid layout issues
                const visibleCells = productCells.filter((cell) => cell !== null);

                return (
                    <div key={`rowSet-${rowIdx}`} className="flex flex-col w-full flex-1 min-h-0">
                        <div className="flex flex-row items-stretch w-full flex-1 min-h-0"
                            style={{ columnGap: "clamp(0.1rem, 0.8vw, 0.5rem)" }}>
                            <div className="flex flex-row flex-1 min-h-0">
                                {visibleCells}
                            </div>
                        </div>
                        {/* Shelf at the bottom of the row */}
                        <ShelfRow
                            rowIdx={rowIdx}
                            cols={cols}
                            slotMap={slotMap}
                            coveredPositions={coveredPositions}
                            slotFlexValues={slotFlexValues}
                            visibleCellsCount={visibleCells.length}
                        />
                    </div>
                );
            })}
        </div>
    );
};

export default VendingMachineGrid;
