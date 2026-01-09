import { useRef, useEffect } from "react"
import { getGradeColor } from "./healthiness-utils"

interface ProductFilterDropdownProps {
  isOpen: boolean
  onClose: () => void
  selectedHealthinessScores: string[]
  pendingHealthinessScores: string[]
  onToggleHealthinessScore: (grade: string) => void
  onApplyFilters: () => void
  onUndoFilters: () => void
}

export default function ProductFilterDropdown({
  isOpen,
  onClose,
  selectedHealthinessScores,
  pendingHealthinessScores,
  onToggleHealthinessScore,
  onApplyFilters,
  onUndoFilters,
}: ProductFilterDropdownProps) {
  const filterDropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (filterDropdownRef.current && !filterDropdownRef.current.contains(event.target as Node)) {
        onClose()
      }
    }

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside)
      return () => document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [isOpen, onClose])

  if (!isOpen) return null

  const healthinessGrades = ['A', 'B', 'C', 'D', 'E']

  return (
    <div 
      ref={filterDropdownRef}
      className="mt-2 bg-card border border-border rounded-lg shadow-lg p-3 md:p-4"
    >
      <div className="mb-3">
        <p className="text-xs text-muted-foreground">
          {selectedHealthinessScores.length === 0 
            ? 'No filters active' 
            : `Active filters: ${selectedHealthinessScores.sort().join(', ')}`}
        </p>
      </div>

      <div className="mb-4">
        <label className="text-sm font-medium text-card-foreground mb-3 block">Filter by Healthiness Score</label>
        <div className="flex gap-2 flex-wrap">
          {healthinessGrades.map((grade) => {
            const isSelected = pendingHealthinessScores.includes(grade)
            return (
              <button
                key={grade}
                onClick={() => onToggleHealthinessScore(grade)}
                className={`
                  p-2 rounded-md border-2 transition-all duration-200 text-center w-fit
                  ${isSelected 
                    ? 'bg-blue-500/20 border-blue-500 text-blue-700 dark:text-blue-300 shadow-md' 
                    : 'bg-background/50 border-border hover:border-blue-300 dark:hover:border-blue-600 text-card-foreground hover:shadow-sm'
                  }
                `}
              >
                <span className={`px-2 py-1 text-xs md:text-sm font-bold rounded ${getGradeColor(grade)}`}>
                  {grade}
                </span>
              </button>
            )
          })}
        </div>
      </div>

      <div className="flex gap-2 justify-end">
        <button
          onClick={onUndoFilters}
          className="px-3 py-1.5 text-sm rounded-lg border border-input bg-background text-card-foreground hover:bg-accent transition-colors"
        >
          Undo
        </button>
        <button
          onClick={onApplyFilters}
          className="px-3 py-1.5 text-sm rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors"
        >
          Apply
        </button>
      </div>
    </div>
  )
}

