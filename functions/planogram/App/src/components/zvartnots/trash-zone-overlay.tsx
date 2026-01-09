import { useRef, useState } from "react"

interface TrashZoneOverlayProps {
  isDraggingProduct: boolean
  onDrop: (position: string) => void
}

export default function TrashZoneOverlay({ isDraggingProduct, onDrop }: TrashZoneOverlayProps) {
  const trashZoneRef = useRef<HTMLDivElement>(null)
  const [isTrashHovered, setIsTrashHovered] = useState(false)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = "move"
    setIsTrashHovered(true)
  }

  const handleDragLeave = () => {
    setIsTrashHovered(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const position = e.dataTransfer.getData("text/plain")
    if (position) {
      onDrop(position)
    }
  }

  if (!isDraggingProduct) return null

  return (
    <div
      ref={trashZoneRef}
      className="fixed inset-0 z-40 flex flex-col lg:flex-row bg-black/1 backdrop-blur-[0.1px] transition-all duration-[1500ms] ease-in-out"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="flex-1 hidden lg:block" />
      <div className="flex-1 flex items-center justify-center pt-[30vh] lg:pt-0">
        <div
          className="flex flex-col items-center justify-center"
          style={{ pointerEvents: 'none' }}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className={`h-16 w-16 mb-2 text-gray-400 transition-transform duration-300 ${isTrashHovered ? 'scale-110 -rotate-12 text-red-500 animate-bounce' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
          </svg>
          <span className={`text-sm font-medium uppercase tracking-wider transition-colors duration-300 ${isTrashHovered ? 'text-red-500' : 'text-gray-400'}`}>Remove</span>
        </div>
      </div>
    </div>
  )
}

