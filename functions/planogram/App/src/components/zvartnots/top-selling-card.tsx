interface TopSellingCardProps {
  topProducts: { name: string; sales: number }[]
  worstProducts: { name: string; sales: number }[]
}

export default function TopSellingCard({ topProducts, worstProducts }: TopSellingCardProps) {
  return (
    <div className="w-full">
      <div className="grid grid-cols-2 gap-6">
        <div>
          <h3 className="text-muted-foreground text-sm font-medium mb-4">Estimated Top Selling</h3>
          <div className="space-y-3">
            {topProducts.map((product, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-lg font-bold text-card-foreground">#{index + 1}</span>
                  <span className="text-sm text-card-foreground">{product.name}</span>
                </div>
                <span className="text-xs font-semibold text-green-600 dark:text-green-400">{product.sales}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-muted-foreground text-sm font-medium mb-4">Estimated Worst Selling</h3>
          <div className="space-y-3">
            {worstProducts.map((product, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-lg font-bold text-card-foreground">#{index + 1}</span>
                  <span className="text-sm text-card-foreground">{product.name}</span>
                </div>
                <span className="text-xs font-semibold text-red-600 dark:text-red-400">{product.sales}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

