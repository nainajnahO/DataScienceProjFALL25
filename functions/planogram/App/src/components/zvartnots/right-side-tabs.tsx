import { useState } from "react"
import StatsSection from "./stats-section.tsx"
import ProductSidebar from "./product-sidebar.tsx"

interface RightSideTabsProps {
  selectedLocation: string | null
  onLocationChange: (location: string | null) => void
  products: any[]
  placedProducts: any[]
  onProductDragStart: () => void
  onProductDragEnd: () => void
  address: string
  onAddressChange: (address: string) => void
  onAutofill: (weights: any) => void
  onOptimize: (weights: any) => void
}

export default function RightSideTabs({
  selectedLocation,
  onLocationChange,
  products,
  placedProducts,
  onProductDragStart,
  onProductDragEnd,
  address,
  onAddressChange,
  onAutofill,
  onOptimize
}: RightSideTabsProps) {
  const [activeTab, setActiveTab] = useState<"information" | "products">("information")

  return (
    <div className="flex flex-col h-full w-full overflow-hidden">
      {/* Tab Navigation */}
      <div className="flex border-b border-border flex-shrink-0">
        <button
          onClick={() => setActiveTab("information")}
          className={`px-4 py-2 font-medium text-sm transition-colors ${activeTab === "information"
            ? "text-primary border-b-2 border-primary"
            : "text-muted-foreground hover:text-foreground"
            }`}
        >
          Information
        </button>
        <button
          onClick={() => setActiveTab("products")}
          className={`px-4 py-2 font-medium text-sm transition-colors ${activeTab === "products"
            ? "text-primary border-b-2 border-primary"
            : "text-muted-foreground hover:text-foreground"
            }`}
        >
          Products
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
        {activeTab === "information" && (
          <div className="flex-1 overflow-y-auto min-h-0">
            <StatsSection
              selectedLocation={selectedLocation}
              onLocationChange={onLocationChange}
              products={products}
              placedProducts={placedProducts}
              address={address}
              onAddressChange={onAddressChange}
              onAutofill={onAutofill}
              onOptimize={onOptimize}
            />
          </div>
        )}
        {activeTab === "products" && (
          <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
            <ProductSidebar
              products={products || []}
              onProductDragStart={onProductDragStart}
              onProductDragEnd={onProductDragEnd}
            />
          </div>
        )}
      </div>
    </div>
  )
}

