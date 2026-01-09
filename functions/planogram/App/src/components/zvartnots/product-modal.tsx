import { useState, useEffect } from "react"
import type { MachineSlot, Product } from "../../lib/types"
import { useProducts } from "@/hooks/useProducts"
import RecommendedProducts from "./recommended-products.tsx"
import ProductListItem from "./product-list-item.tsx"
import ProductConfirmationDialog from "./product-confirmation-dialog.tsx"
import ProductManualForm from "./product-manual-form.tsx"
import ProductCurrentDisplay from "./product-current-display.tsx"
import ProductSearchSection from "./product-search-section.tsx"
import { useProductFiltering } from "./use-product-filtering.ts"

export interface ProductModalProps {
  isOpen: boolean
  onClose: () => void
  onSelectProduct: (product: Product) => void
  onUpdateStock?: (position: string, newStock: number) => void
  onRemoveProduct?: (position: string) => void
  currentSlot?: MachineSlot | undefined
  selectedPosition?: string | null
}

export default function ProductModal({ isOpen, onClose, onSelectProduct, onUpdateStock: _onUpdateStock, onRemoveProduct, currentSlot, selectedPosition }: ProductModalProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const [showConfirmation, setShowConfirmation] = useState(false)
  const [pendingProduct, setPendingProduct] = useState<Product | null>(null)
  const [showFilterDropdown, setShowFilterDropdown] = useState(false)
  const [selectedHealthinessScores, setSelectedHealthinessScores] = useState<string[]>([])
  const [pendingHealthinessScores, setPendingHealthinessScores] = useState<string[]>([])
  const [isManualMode, setIsManualMode] = useState(false)

  const { products, loading, error } = useProducts()

  useEffect(() => {
    if (isOpen && !isManualMode) {
      const timer = setTimeout(() => {
        const input = document.querySelector('input[placeholder="Search products..."]') as HTMLInputElement
        input?.focus()
      }, 50)
      return () => clearTimeout(timer)
    }
  }, [isOpen, isManualMode])

  useEffect(() => {
    if (!isOpen) {
      setShowConfirmation(false)
      setPendingProduct(null)
      setSearchQuery("")
      setIsManualMode(false)
      setShowFilterDropdown(false)
      setSelectedHealthinessScores([])
      setPendingHealthinessScores([])
    }
  }, [isOpen])

  useEffect(() => {
    if (showFilterDropdown) {
      setPendingHealthinessScores([...selectedHealthinessScores])
    }
  }, [showFilterDropdown])

  // All hooks must be called before any early returns
  const filteredProducts = useProductFiltering(products || [], searchQuery, selectedHealthinessScores)

  if (!isOpen) return null

  const hasProduct = currentSlot?.product_name && currentSlot?.product_name !== ""
  const slotWidth = currentSlot?.width || 1
  const currentProduct = (products || []).find(p => p.name === currentSlot?.product_name || p.product_name === currentSlot?.product_name)

  const handleRemove = () => {
    if (onRemoveProduct && selectedPosition) {
      if (confirm('Are you sure you want to remove this product?')) {
        onRemoveProduct(selectedPosition)
        onClose()
      }
    }
  }

  const handleProductClick = (product: Product) => {
    const productWidth = product.width || 1
    if (productWidth > slotWidth) {
      return
    }

    if (hasProduct) {
      setPendingProduct(product)
      setShowConfirmation(true)
    } else {
      onSelectProduct(product)
      onClose()
    }
  }

  const handleConfirmChange = () => {
    if (pendingProduct) {
      onSelectProduct(pendingProduct)
      setShowConfirmation(false)
      setPendingProduct(null)
      onClose()
    }
  }

  const handleCancelChange = () => {
    setShowConfirmation(false)
    setPendingProduct(null)
  }

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

  const targetRecommendations = ["Cloetta Egg", "Thai Cup Noodles Chicken", "Risifrutti Hallon"]
  const safeProducts = products || []
  const recommendedProducts = safeProducts.length > 0
    ? (() => {
        const found = targetRecommendations
          .map(name => safeProducts.find(p => p.name.toLowerCase() === name.toLowerCase()))
          .filter((p): p is Product => !!p)

        if (found.length < 3) {
          const existingIds = new Set(found.map(p => p.id))
          const fillers = safeProducts.filter(p => !existingIds.has(p.id)).slice(0, 3 - found.length)
          return [...found, ...fillers]
        }
        return found
      })()
    : []

  return (
    <div className="fixed inset-0 bg-black/50 dark:bg-black/70 flex items-center justify-center z-50 p-2 md:p-4" onClick={onClose}>
      <div
        className="bg-white dark:bg-gray-800 bg-popover dark:bg-popover border border-border rounded-lg p-3 md:p-6 max-w-2xl w-full max-h-[calc(100vh-2rem)] md:max-h-[calc(100vh-4rem)] min-h-[200px] flex flex-col shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-2 md:mb-4">
          <div className="flex items-center gap-2">
            {isManualMode && (
              <button
                onClick={() => setIsManualMode(false)}
                className="text-sm text-blue-500 hover:text-blue-600 font-medium"
              >
                ← Back
              </button>
            )}
            <h2 className="text-lg md:text-2xl font-bold text-popover-foreground">
              {isManualMode ? "Create Product" : "Select Product"}
            </h2>
          </div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground text-2xl md:text-3xl font-bold leading-none">
            ×
          </button>
        </div>

        {error && (
          <div className="mb-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <p className="text-sm text-red-600 dark:text-red-400 mb-2">Failed to load products</p>
            <p className="text-xs text-red-500 dark:text-red-300">{error}</p>
          </div>
        )}

        {loading && (
          <div className="flex-1 flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
              <p className="text-lg text-popover-foreground">Loading products...</p>
            </div>
          </div>
        )}

        {hasProduct && !isManualMode && (
          <ProductCurrentDisplay
            currentSlot={currentSlot}
            currentProduct={currentProduct}
            onRemove={handleRemove}
          />
        )}

        {!loading && !error && (
          <>
            {!hasProduct && !isManualMode && (
              <RecommendedProducts
                products={recommendedProducts}
                onProductClick={handleProductClick}
              />
            )}

            <div className="flex-1 flex flex-col min-h-0">
              {!isManualMode ? (
                <>
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

                  <div className="overflow-y-auto flex-1">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 md:gap-3">
                      {filteredProducts.map((product) => (
                        <ProductListItem
                          key={product.id}
                          product={product}
                          onClick={handleProductClick}
                          disabled={(product.width || 1) > slotWidth}
                          slotWidth={slotWidth}
                        />
                      ))}
                    </div>
                    {filteredProducts.length === 0 && <p className="text-center text-muted-foreground py-8">No products found</p>}
                  </div>
                  <div className="flex justify-end pt-2 md:pt-4 border-t border-border">
                    <button
                      onClick={() => setIsManualMode(true)}
                      className="px-3 py-1.5 md:px-4 md:py-2 bg-blue-600 text-white rounded-lg whitespace-nowrap text-sm font-medium hover:bg-blue-700 transition-colors"
                    >
                      Create New
                    </button>
                  </div>
                </>
              ) : (
                <ProductManualForm
                  onProductCreate={(product) => {
                    handleProductClick(product)
                  }}
                  onCancel={() => setIsManualMode(false)}
                />
              )}
            </div>
          </>
        )}
      </div>

      <ProductConfirmationDialog
        isOpen={showConfirmation}
        currentSlot={currentSlot}
        pendingProduct={pendingProduct}
        onConfirm={handleConfirmChange}
        onCancel={handleCancelChange}
      />
    </div>
  )
}
