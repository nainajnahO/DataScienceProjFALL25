import type { Product } from "../../lib/types";

interface RecommendedProductsProps {
  products: Product[];
  onProductClick: (product: Product) => void;
}

export default function RecommendedProducts({ products, onProductClick }: RecommendedProductsProps) {
  return (
    <div className="mb-4">
      <h3 className="text-sm font-semibold text-popover-foreground mb-3 flex items-center gap-2">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-500" viewBox="0 0 20 20" fill="currentColor">
          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
        </svg>
        Recommended for You
      </h3>
      <div className="grid grid-cols-3 gap-2">
        {products.map((product) => (
          <div
            key={product.id}
            onClick={() => onProductClick(product)}
            className="p-3 bg-card border-2 border-yellow-500/50 rounded-lg hover:bg-accent hover:border-yellow-500 cursor-pointer transition-all flex flex-col items-center gap-2"
          >
            {product.image && (
              <img
                src={product.image || "/placeholder.svg"}
                alt={product.name}
                className="w-12 h-12 object-contain rounded"
              />
            )}
            <div className="text-center">
              <h4 className="font-semibold text-xs text-card-foreground">{product.name}</h4>
              <p className="text-xs font-bold text-green-600 dark:text-green-400">{product.price} kr</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

