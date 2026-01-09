import React from "react";

interface VendingMachineRowLabelsProps {
  rows: number;
}

const VendingMachineRowLabels: React.FC<VendingMachineRowLabelsProps> = ({ rows }) => (
  <div
    className="flex flex-col w-6 h-full"
    style={{
      background: "linear-gradient(145deg, #2a2a2a, #1a1a1a)",
      borderRadius: "6px",
      boxShadow: "0 2px 4px rgba(0,0,0,0.5), inset 0 1px 1px rgba(255,255,255,0.05)",
      border: "1px solid #333",
      paddingTop: "2px",
      paddingBottom: "2px"
    }}
  >
    {Array.from({ length: rows }).map((_, i) => (
      <React.Fragment key={i}>
        <div 
          className="flex items-center justify-center text-gray-400 text-sm font-bold flex-1"
        >
          {String.fromCharCode(65 + i)}
        </div>
        {i < rows - 1 && (
          <div 
            className="w-full h-px"
            style={{
              background: "linear-gradient(to right, transparent, #555 20%, #555 80%, transparent)"
            }}
          />
        )}
      </React.Fragment>
    ))}
  </div>
);

export default VendingMachineRowLabels;
