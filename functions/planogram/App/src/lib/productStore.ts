import type { Product } from "./types"
import { getAllProducts } from "./services/products"

type Listener = () => void

/**
 * Global store for product catalog
 * Fetches products once from Firestore and caches globally
 * Implements observer pattern for reactive updates
 * Similar pattern to newMachineStore.ts
 */
class ProductStore {
  private products: Product[] = []
  private loading: boolean = false
  private error: string | null = null
  private initialized: boolean = false
  private listeners: Set<Listener> = new Set()

  /**
   * Initialize the store by fetching products from Firestore
   * Only fetches once - subsequent calls return cached data
   * Call this early in app lifecycle (e.g., App.tsx useEffect)
   */
  async initialize(): Promise<void> {
    // If already initialized or currently loading, skip
    if (this.initialized || this.loading) {
      console.log("📦 Product store already initialized or loading")
      return
    }

    console.log("🔄 Initializing product store...")
    this.loading = true
    this.error = null
    this.notifyListeners()

    try {
      this.products = await getAllProducts()
      this.initialized = true
      this.loading = false
      console.log(`✅ Product store initialized with ${this.products.length} products`)
      this.notifyListeners()
    } catch (err) {
      this.error = err instanceof Error ? err.message : "Failed to load products"
      this.loading = false
      this.initialized = false
      console.error("❌ Product store initialization failed:", this.error)
      this.notifyListeners()
      throw err
    }
  }

  /**
   * Get all cached products
   * Returns empty array if not initialized
   */
  getProducts(): Product[] {
    return this.products
  }

  /**
   * Get single product by Firestore document ID
   */
  getProductById(id: string): Product | undefined {
    return this.products.find(p => p.id === id)
  }

  /**
   * Get single product by product name
   */
  getProductByName(name: string): Product | undefined {
    return this.products.find(p => p.product_name === name || p.name === name)
  }

  /**
   * Get products by category
   */
  getProductsByCategory(category: string): Product[] {
    return this.products.filter(p => p.category === category)
  }

  /**
   * Search products by name or category
   * Case-insensitive search
   */
  searchProducts(query: string): Product[] {
    if (!query || query.trim() === "") {
      return this.products
    }

    const lowerQuery = query.toLowerCase()
    return this.products.filter(p =>
      p.name.toLowerCase().includes(lowerQuery) ||
      p.product_name.toLowerCase().includes(lowerQuery) ||
      p.category.toLowerCase().includes(lowerQuery)
    )
  }

  /**
   * Get loading state
   */
  isLoading(): boolean {
    return this.loading
  }

  /**
   * Get error state
   */
  getError(): string | null {
    return this.error
  }

  /**
   * Check if store has been initialized
   */
  isInitialized(): boolean {
    return this.initialized
  }

  /**
   * Force refresh products from Firestore
   * Useful for manual refresh or after product updates
   */
  async refresh(): Promise<void> {
    console.log("🔄 Refreshing product store...")
    this.initialized = false
    await this.initialize()
  }

  /**
   * Clear the product store
   * Resets to initial state
   */
  clear(): void {
    this.products = []
    this.loading = false
    this.error = null
    this.initialized = false
    this.notifyListeners()
  }

  /**
   * Subscribe to store changes
   * Returns unsubscribe function
   *
   * @example
   * const unsubscribe = productStore.subscribe(() => {
   *   console.log("Products updated:", productStore.getProducts())
   * })
   * // Later: unsubscribe()
   */
  subscribe(listener: Listener): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  /**
   * Notify all listeners of state changes
   * Called internally when products/loading/error state changes
   */
  private notifyListeners(): void {
    this.listeners.forEach(listener => listener())
  }

  /**
   * Get current store state for debugging
   */
  getState() {
    return {
      productCount: this.products.length,
      loading: this.loading,
      error: this.error,
      initialized: this.initialized,
      listenerCount: this.listeners.size
    }
  }
}

// Export singleton instance
export const productStore = new ProductStore()
