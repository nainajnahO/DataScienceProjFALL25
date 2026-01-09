import { useNavigate } from "react-router-dom"
import type { AppMachine, MachineSlot } from "../../lib/types"
import { newMachineStore } from "@/lib/newMachineStore"

interface EditLayoutButtonProps {
  machine: AppMachine
  slots: MachineSlot[]
}

export default function EditLayoutButton({ machine, slots }: EditLayoutButtonProps) {
  const navigate = useNavigate()

  const handleClick = () => {
    const updatedMachine = { ...machine, slots }
    if (machine.id === "new") {
      newMachineStore.setMachine(updatedMachine)
    }
    navigate(`/machines/${machine.id}/layout`, { state: { machine: updatedMachine } })
  }

  return (
    <button
      onClick={handleClick}
      className="group relative bg-white hover:bg-gray-50 text-gray-900 px-6 py-3.5 rounded-xl shadow-lg hover:shadow-xl font-semibold transition-all duration-300 flex items-center gap-3 transform hover:scale-105 hover:-translate-y-0.5 border border-gray-200"
    >
      <div className="absolute inset-0 bg-gradient-to-r from-gray-100 to-gray-50 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 group-hover:rotate-12 transition-transform duration-300 text-gray-700" viewBox="0 0 20 20" fill="currentColor">
        <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
      </svg>
      <span className="relative z-10">Edit Layout</span>
      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 group-hover:translate-x-1 transition-transform duration-300 text-gray-700" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
      </svg>
    </button>
  )
}

