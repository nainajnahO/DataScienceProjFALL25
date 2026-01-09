import React from "react";
import {indexToPosition} from "./vending-machine-utils.ts";
import type {MachineSlot} from "../../lib/types.ts";

interface ShelfRowProps {
    rowIdx: number;
    cols: number;
    slotMap: Map<string, MachineSlot>;
    coveredPositions: Set<string>;
    slotFlexValues: number[];
    visibleCellsCount: number;
}

const ShelfRow: React.FC<ShelfRowProps> = ({
    rowIdx,
    cols,
    slotMap,
    coveredPositions,
    slotFlexValues,
    visibleCellsCount
}) => {
    const rowStart = rowIdx * cols;

    return (
        <div
            className="w-full relative md:h-[12px] h-[18px]"
            style={{
                marginTop: "2px",
            }}
        >
            {/* Shelf main bar */}
            <div
                className="absolute inset-0"
                style={{
                    background: "linear-gradient(to bottom, #999 0%, #666 30%, #444 60%, #555 100%)",
                    borderRadius: "2px",
                    boxShadow: "0 6px 14px rgba(0,0,0,0.9), inset 0 2px 3px rgba(255,255,255,0.3), inset 0 -3px 6px rgba(0,0,0,0.6)",
                }}
            />
            {/* Shelf front edge highlight */}
            <div
                className="absolute bottom-0 left-0 right-0 md:h-1 h-[2px]"
                style={{
                    background: "linear-gradient(to right, transparent, rgba(255,255,255,0.25) 50%, transparent)",
                }}
            />
            
            {/* Vertical separators and stock indicators matching slot widths */}
            <div className="absolute inset-0 flex flex-row items-stretch w-full" style={{ columnGap: "clamp(0.1rem, 0.8vw, 0.5rem)" }}>
                <div className="flex flex-row flex-1">
                    {Array.from({length: visibleCellsCount}).map((_, cellIdx) => {
                        const flexValue = slotFlexValues[cellIdx];
                        const isLastCell = cellIdx === visibleCellsCount - 1;
                        
                        // Find the actual position for this visible cell
                        let visibleCount = 0;
                        let cellPosition = '';
                        for (let i = 0; i < cols; i++) {
                            const idx = rowStart + i;
                            const pos = indexToPosition(idx, cols);
                            if (!coveredPositions.has(pos)) {
                                if (visibleCount === cellIdx) {
                                    cellPosition = pos;
                                    break;
                                }
                                visibleCount++;
                            }
                        }
                        
                        const cellSlot = slotMap.get(cellPosition);
                        const hasProduct = cellSlot?.product_name;
                        const stockPercent = cellSlot?.stock_max ? (cellSlot.stock_current / cellSlot.stock_max) : 0;
                        
                        return (
                            <div
                                key={cellIdx}
                                className="relative h-full"
                                style={{
                                    flex: `${flexValue} ${flexValue} 0%`,
                                    flexBasis: 0,
                                }}
                            >
                                {/* Stock indicator bar at the top of shelf */}
                                {hasProduct && cellSlot?.stock_max && (
                                    <div className="absolute top-0 left-0 right-0 md:h-1 h-[4px] bg-black/80">
                                        <div
                                            className="h-full transition-all duration-300"
                                            style={{
                                                width: cellSlot.stock_current === 0 ? '100%' : `${Math.round(stockPercent * 100)}%`,
                                                background: cellSlot.stock_current <= 2 
                                                    ? 'rgb(239, 68, 68)' 
                                                    : `hsl(${Math.round(stockPercent * 120)}, 90%, 50%)`,
                                                boxShadow: cellSlot.stock_current <= 2 
                                                    ? '0 0 8px rgb(239, 68, 68)' 
                                                    : `0 0 8px hsl(${Math.round(stockPercent * 120)}, 90%, 50%)`,
                                            }}
                                            title={`Stock: ${cellSlot.stock_current}/${cellSlot.stock_max}`}
                                        />
                                    </div>
                                )}
                                
                                {/* Vertical separator line on the right - positioned to align with product cell borders */}
                                {!isLastCell && (
                                    <div
                                        className="absolute top-0 bottom-0 border-r border-gray-700 z-10"
                                        style={{
                                            right: 0,
                                            borderImage: "linear-gradient(to bottom, transparent, #333 20%, #222 50%, #333 80%, transparent) 1",
                                        }}
                                    />
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};

export default ShelfRow;

