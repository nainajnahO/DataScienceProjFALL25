import { useRef } from "react"
import ProductFilterDropdown from "./product-filter-dropdown.tsx"

interface ProductSearchSectionProps {
  searchQuery: string
  onSearchChange: (query: string) => void
  showFilterDropdown: boolean
  onToggleFilter: () => void
  selectedHealthinessScores: string[]
  pendingHealthinessScores: string[]
  onToggleHealthinessScore: (grade: string) => void
  onApplyFilters: () => void
  onUndoFilters: () => void
  onCloseFilter: () => void
}

export default function ProductSearchSection({
  searchQuery,
  onSearchChange,
  showFilterDropdown,
  onToggleFilter,
  selectedHealthinessScores,
  pendingHealthinessScores,
  onToggleHealthinessScore,
  onApplyFilters,
  onUndoFilters,
  onCloseFilter,
}: ProductSearchSectionProps) {
  const searchInputRef = useRef<HTMLInputElement>(null)

  return (
    <div className="mb-2 md:mb-4">
      <div className="flex gap-2">
        <input
          ref={searchInputRef}
          type="text"
          placeholder="Search products..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="flex-1 px-3 py-1.5 md:px-4 md:py-2 bg-background border border-input rounded-lg focus:outline-none focus:ring-2 focus:ring-ring text-sm md:text-base text-foreground"
        />
        <button
          onClick={onToggleFilter}
          className={`px-3 py-1.5 md:px-4 md:py-2 bg-background border border-input rounded-lg hover:bg-accent transition-colors flex items-center justify-center ${showFilterDropdown ? 'bg-accent' : ''}`}
          aria-label="Filter products"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-foreground" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="3" y1="6" x2="21" y2="6" />
            <circle cx="18" cy="6" r="2" fill="none" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <circle cx="12" cy="12" r="2" fill="none" />
            <line x1="3" y1="18" x2="21" y2="18" />
            <circle cx="6" cy="18" r="2" fill="none" />
          </svg>
        </button>
      </div>
      
      <ProductFilterDropdown
        isOpen={showFilterDropdown}
        onClose={onCloseFilter}
        selectedHealthinessScores={selectedHealthinessScores}
        pendingHealthinessScores={pendingHealthinessScores}
        onToggleHealthinessScore={onToggleHealthinessScore}
        onApplyFilters={onApplyFilters}
        onUndoFilters={onUndoFilters}
      />
    </div>
  )
}

