import MachineHealthinessBadge from "./machine-healthiness-badge.tsx"
import type { ReactNode } from "react"

interface VendingMachineControlsProps {

  onDownload: () => void
  healthinessGrade: string | null
  editLayoutAction?: ReactNode
}

export default function VendingMachineControls({

  onDownload,
  healthinessGrade,
  editLayoutAction,
}: VendingMachineControlsProps) {


  return (
    <div className="mt-6 flex justify-between items-center w-full max-w-full md:max-w-4xl">
      <MachineHealthinessBadge grade={healthinessGrade} />

      <div className="flex items-center gap-4">


        {editLayoutAction}

        <button
          onClick={onDownload}
          className="group relative bg-blue-600 hover:bg-blue-700 text-white px-6 py-3.5 rounded-xl shadow-lg hover:shadow-xl font-semibold transition-all duration-300 flex items-center gap-2 transform hover:scale-105 hover:-translate-y-0.5"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 group-hover:animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          <span>Download PNG</span>
        </button>
      </div>
    </div>
  )
}

