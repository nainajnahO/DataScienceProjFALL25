import { useState, useEffect } from "react"
import { productStore } from "@/lib/productStore"
import type { Product } from "@/lib/types"

/**
 * React hook for accessing the global product store
 * Automatically subscribes to store changes and initializes on first use
 *
 * @returns {object} Products data and utilities
 * @returns {Product[]} products - Array of all products
 * @returns {boolean} loading - Loading state
 * @returns {string | null} error - Error message if fetch failed
 * @returns {() => Promise<void>} refresh - Function to manually refresh products
 *
 * @example
 * function ProductList() {
 *   const { products, loading, error, refresh } = useProducts()
 *
 *   if (loading) return <div>Loading...</div>
 *   if (error) return <div>Error: {error}</div>
 *
 *   return (
 *     <div>
 *       {products.map(p => <div key={p.id}>{p.name}</div>)}
 *       <button onClick={refresh}>Refresh</button>
 *     </div>
 *   )
 * }
 */
export function useProducts() {
  const [products, setProducts] = useState<Product[]>(productStore.getProducts())
  const [loading, setLoading] = useState<boolean>(productStore.isLoading())
  const [error, setError] = useState<string | null>(productStore.getError())

  useEffect(() => {
    // Initialize store if not already done
    if (!productStore.isInitialized() && !productStore.isLoading()) {
      productStore.initialize().catch(console.error)
    }

    // Subscribe to store changes
    const unsubscribe = productStore.subscribe(() => {
      setProducts(productStore.getProducts())
      setLoading(productStore.isLoading())
      setError(productStore.getError())
    })

    // Cleanup subscription on unmount
    return unsubscribe
  }, [])

  return {
    products,
    loading,
    error,
    refresh: () => productStore.refresh(),
  }
}

/**
 * Hook for searching products with debouncing
 * Filters products based on search query
 *
 * @param query - Search query string
 * @returns Filtered products and loading/error states
 *
 * @example
 * function SearchableProductList() {
 *   const [query, setQuery] = useState("")
 *   const { products, loading, error } = useProductSearch(query)
 *
 *   return (
 *     <>
 *       <input value={query} onChange={e => setQuery(e.target.value)} />
 *       {products.map(p => <div key={p.id}>{p.name}</div>)}
 *     </>
 *   )
 * }
 */
export function useProductSearch(query: string) {
  const { products: allProducts, loading, error } = useProducts()
  const [filteredProducts, setFilteredProducts] = useState<Product[]>([])

  useEffect(() => {
    if (query.trim()) {
      setFilteredProducts(productStore.searchProducts(query))
    } else {
      setFilteredProducts(allProducts)
    }
  }, [query, allProducts])

  return {
    products: filteredProducts,
    loading,
    error,
  }
}

/**
 * Hook for getting products by category
 *
 * @param category - Product category to filter by
 * @returns Products in the specified category
 *
 * @example
 * function BeveragesSection() {
 *   const { products, loading } = useProductsByCategory("Beverages")
 *   return <div>{products.map(p => <div>{p.name}</div>)}</div>
 * }
 */
export function useProductsByCategory(category: string) {
  const { products: allProducts, loading, error } = useProducts()
  const [categoryProducts, setCategoryProducts] = useState<Product[]>([])

  useEffect(() => {
    setCategoryProducts(productStore.getProductsByCategory(category))
  }, [category, allProducts])

  return {
    products: categoryProducts,
    loading,
    error,
  }
}
