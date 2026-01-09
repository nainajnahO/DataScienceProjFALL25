import { useState, useRef, useEffect } from "react"
import type { Product } from "../../lib/types"
import ProductListItem from "./product-list-item.tsx"
import ProductSearchSection from "./product-search-section.tsx"
import { useProductFiltering } from "./use-product-filtering.ts"

interface ProductSidebarProps {
  products: Product[]
  onProductDragStart: () => void
  onProductDragEnd: () => void
}

interface DraggableProductItemProps {
  product: Product
  onProductDragStart: () => void
  onProductDragEnd: () => void
}

function DraggableProductItem({ product, onProductDragStart, onProductDragEnd }: DraggableProductItemProps) {
  const elementRef = useRef<HTMLDivElement>(null)
  const dragTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const handleDragStart = (e: React.DragEvent) => {
    e.dataTransfer.setData("application/json", JSON.stringify({ type: "product", product }))
    e.dataTransfer.effectAllowed = "copy"

    // Use setTimeout to let browser capture drag image before hiding element
    dragTimeoutRef.current = setTimeout(() => {
      if (elementRef.current) {
        elementRef.current.style.opacity = "0"
      }
      onProductDragStart()
    }, 0)
  }

  const handleDragEnd = () => {
    if (dragTimeoutRef.current) {
      clearTimeout(dragTimeoutRef.current)
      dragTimeoutRef.current = null
    }
    if (elementRef.current) {
      elementRef.current.style.opacity = "1"
    }
    onProductDragEnd()
  }

  return (
    <div
      ref={elementRef}
      draggable={true}
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      className="cursor-grab active:cursor-grabbing"
    >
      <ProductListItem
        product={product}
        onClick={() => {}}
        disabled={false}
        showWidth={true}
      />
    </div>
  )
}

export default function ProductSidebar({ products, onProductDragStart, onProductDragEnd }: ProductSidebarProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedHealthinessScores, setSelectedHealthinessScores] = useState<string[]>([])
  const [pendingHealthinessScores, setPendingHealthinessScores] = useState<string[]>([])
  const [showFilterDropdown, setShowFilterDropdown] = useState(false)

  const filteredProducts = useProductFiltering(products || [], searchQuery, selectedHealthinessScores)

  useEffect(() => {
    if (showFilterDropdown) {
      setPendingHealthinessScores([...selectedHealthinessScores])
    }
  }, [showFilterDropdown, selectedHealthinessScores])

  const handleToggleHealthinessScore = (grade: string) => {
    setPendingHealthinessScores(prev => 
      prev.includes(grade) 
        ? prev.filter(g => g !== grade)
        : [...prev, grade]
    )
  }

  const handleApplyFilters = () => {
    setSelectedHealthinessScores([...pendingHealthinessScores])
    setShowFilterDropdown(false)
  }

  const handleUndoFilters = () => {
    setSelectedHealthinessScores([])
    setPendingHealthinessScores([])
    setShowFilterDropdown(false)
  }

  return (
    <div className="w-full h-full flex flex-col bg-card border border-border rounded-lg shadow-sm overflow-hidden">
      <div className="p-3 border-b border-border flex-shrink-0">
        <ProductSearchSection
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          showFilterDropdown={showFilterDropdown}
          onToggleFilter={() => setShowFilterDropdown(!showFilterDropdown)}
          selectedHealthinessScores={selectedHealthinessScores}
          pendingHealthinessScores={pendingHealthinessScores}
          onToggleHealthinessScore={handleToggleHealthinessScore}
          onApplyFilters={handleApplyFilters}
          onUndoFilters={handleUndoFilters}
          onCloseFilter={() => setShowFilterDropdown(false)}
        />
      </div>
      <div className="flex-1 flex flex-col min-h-0">
        <div className="overflow-y-auto flex-1 p-3">
          <div className="grid grid-cols-2 gap-2">
            {filteredProducts.map((product) => (
              <DraggableProductItem
                key={product.id}
                product={product}
                onProductDragStart={onProductDragStart}
                onProductDragEnd={onProductDragEnd}
              />
            ))}
            {filteredProducts.length === 0 && (
              <p className="text-center text-muted-foreground py-8 col-span-2">No products found</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

