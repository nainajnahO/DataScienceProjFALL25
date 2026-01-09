import type { MachineSlot, Product } from "../../lib/types"
import { getHealthinessGrade, getGradeColor } from "./healthiness-utils"

interface ProductCurrentDisplayProps {
  currentSlot: MachineSlot | undefined
  currentProduct: Product | undefined
  onRemove: () => void
}

export default function ProductCurrentDisplay({ currentSlot, currentProduct, onRemove }: ProductCurrentDisplayProps) {
  if (!currentSlot?.product_name || currentSlot.product_name === "") return null

  const currentProductHealthinessGrade = currentProduct ? getHealthinessGrade(currentProduct) : null

  return (
    <div className="mb-2 md:mb-4 p-2 md:p-4 bg-card border border-border rounded-lg">
      <div className="flex flex-col md:flex-row md:items-start gap-2 md:gap-4">
        <div className="flex items-center gap-2 md:gap-3 flex-1">
          {currentSlot.image_url && (
            <img
              src={currentSlot.image_url}
              alt={currentSlot.product_name}
              className="w-10 h-10 md:w-20 md:h-20 object-contain"
            />
          )}
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="text-sm md:text-lg font-bold text-card-foreground">{currentSlot.product_name}</h3>
              {currentProductHealthinessGrade && (
                <span className={`px-2 py-0.5 text-xs md:text-sm font-bold rounded ${getGradeColor(currentProductHealthinessGrade)}`}>
                  {currentProductHealthinessGrade}
                </span>
              )}
            </div>
            <p className="text-xs md:text-sm text-muted-foreground">{currentSlot.category}</p>
            <p className="text-xs md:text-base font-bold text-green-600 dark:text-green-400 mt-0.5 md:mt-1">{currentSlot.price} kr</p>

            <button
              onClick={onRemove}
              className="mt-2 px-3 py-1 bg-red-100 hover:bg-red-200 text-red-700 dark:bg-red-900/30 dark:hover:bg-red-900/50 dark:text-red-400 rounded-md text-xs font-medium transition-colors flex items-center gap-1"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              Remove
            </button>
          </div>
        </div>

        <div className="border-t md:border-t-0 md:border-l border-border pt-2 md:pt-0 md:pl-4 flex flex-col md:flex-col gap-3 md:gap-4">
          <div>
            <label className="text-xs md:text-sm font-medium text-muted-foreground mb-1.5 block">Spiral Width</label>
            <div className="text-lg md:text-2xl font-bold text-card-foreground">
              {currentSlot.width || 1}
            </div>
          </div>
          <div>
            <label className="text-xs md:text-sm font-medium text-muted-foreground mb-1.5 block">Product Width</label>
            <div className="text-lg md:text-2xl font-bold text-card-foreground">
              {currentProduct?.width || currentSlot.width || 1}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

