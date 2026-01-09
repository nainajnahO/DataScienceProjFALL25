import { useState } from "react"
import type { Product } from "../../lib/types"
import { getHealthinessGrade, getGradeColor } from "./healthiness-utils"

interface ProductListItemProps {
  product: Product
  onClick: (product: Product) => void
  disabled?: boolean
  slotWidth?: number
  showWidth?: boolean
}

export default function ProductListItem({ product, onClick, disabled, slotWidth, showWidth }: ProductListItemProps) {
  const [imageError, setImageError] = useState(false)
  const healthinessGrade = getHealthinessGrade(product)
  const hasImage = product.image && !imageError

  return (
    <div
      onClick={() => !disabled && onClick(product)}
      className={`p-2 md:p-4 bg-card border rounded-lg transition-all flex flex-col gap-2 ${disabled
        ? "opacity-50 cursor-not-allowed border-border"
        : "hover:bg-accent hover:border-accent-foreground cursor-pointer border-border"
        }`}
    >
      <div className="flex gap-2 md:gap-3">
        {hasImage ? (
          <img
            src={product.image}
            alt={product.name}
            className="w-12 h-12 md:w-16 md:h-16 object-contain rounded flex-shrink-0"
            onError={() => setImageError(true)}
          />
        ) : (
          <div className="w-12 h-12 md:w-16 md:h-16 flex items-center justify-center bg-muted rounded flex-shrink-0 border border-border">
            <span className="text-xs md:text-sm font-bold text-muted-foreground">
              {product.name.substring(0, 2).toUpperCase()}
            </span>
          </div>
        )}
        <div className="flex-1 min-w-0">
          <div className="flex justify-between items-start gap-2">
            <h3 className="font-semibold text-sm md:text-base text-card-foreground truncate pr-2 flex-1">{product.name}</h3>
            <div className="flex items-center gap-1.5 shrink-0">
              {healthinessGrade && (
                <span className={`px-2 py-0.5 text-xs md:text-sm font-bold rounded ${getGradeColor(healthinessGrade)}`}>
                  {healthinessGrade}
                </span>
              )}
              {disabled && (
                <span className="shrink-0 px-2 py-0.5 bg-yellow-100 text-yellow-800 text-xs uppercase font-bold rounded">
                  Too Wide ({product.width || 1} &gt; {slotWidth})
                </span>
              )}
            </div>
          </div>
          <p className="text-xs md:text-sm text-muted-foreground">{product.category}</p>
          <p className="text-sm md:text-lg font-bold text-green-600 dark:text-green-400 mt-0.5 md:mt-1">{product.price} kr</p>
        </div>
      </div>
      {showWidth && (
        <div className="border-t border-border pt-2 mt-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-muted-foreground">Product Width</span>
            <span className="text-sm font-bold text-card-foreground">{product.width || 1}</span>
          </div>
        </div>
      )}
    </div>
  )
}

