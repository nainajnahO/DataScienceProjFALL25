import React from "react";

interface VendingMachineColumnHeadersProps {
  cols: number;
}

const VendingMachineColumnHeaders: React.FC<VendingMachineColumnHeadersProps> = ({ cols }) => (
  <div className="hidden md:flex">
    {/* Column numbers */}
    <div
      className="flex-1 flex flex-row h-6"
      style={{
        background: "linear-gradient(145deg, #2a2a2a, #1a1a1a)",
        borderRadius: "6px",
        boxShadow: "0 2px 4px rgba(0,0,0,0.5), inset 0 1px 1px rgba(255,255,255,0.05)",
        border: "1px solid #333",
        columnGap: "clamp(0.25rem, 0.8vw, 0.5rem)",
        paddingLeft: "2px",
        paddingRight: "2px"
      }}
    >
      {Array.from({ length: cols }).map((_, i) => (
        <React.Fragment key={i}>
          <div 
            className="text-gray-400 text-xs font-bold flex items-center justify-center min-w-0"
            style={{ 
              flex: "2 2 0%"
            }}
          >
            {i}
          </div>
          {i < cols - 1 && (
            <div 
              className="w-px h-full"
              style={{
                background: "linear-gradient(to bottom, transparent, #555 20%, #555 80%, transparent)"
              }}
            />
          )}
        </React.Fragment>
      ))}
    </div>
  </div>
)

export default VendingMachineColumnHeaders;
