import type { Product } from "./types"
import { getAllHealthinessGrades } from "./services/healthinessGrades"

type Listener = () => void

/**
 * Global store for healthiness grade mappings
 * Fetches grades once from Firestore and caches globally
 * Implements observer pattern for reactive updates
 * Similar pattern to productStore.ts
 */
class HealthinessGradeStore {
  private gradeMap: Map<string, string> = new Map()
  private loading: boolean = false
  private error: string | null = null
  private initialized: boolean = false
  private listeners: Set<Listener> = new Set()

  /**
   * Initialize the store by fetching healthiness grades from Firestore
   * Only fetches once - subsequent calls return cached data
   * Call this early in app lifecycle (e.g., App.tsx useEffect)
   */
  async initialize(): Promise<void> {
    // If already initialized or currently loading, skip
    if (this.initialized || this.loading) {
      return
    }
    this.loading = true
    this.error = null
    this.notifyListeners()

    try {
      this.gradeMap = await getAllHealthinessGrades()
      this.initialized = true
      this.loading = false
      this.notifyListeners()
    } catch (err) {
      this.error = err instanceof Error ? err.message : "Failed to load healthiness grades"
      this.loading = false
      this.initialized = false
      console.error("❌ Healthiness grade store initialization failed:", this.error)
      this.notifyListeners()
      throw err
    }
  }

  /**
   * Normalize EAN/barcode value for lookup (convert to string, trim whitespace)
   */
  private normalizeEan(ean: string | number | undefined | null): string | null {
    if (ean == null) return null
    const normalized = String(ean).trim()
    return normalized || null
  }

  /**
   * Get healthiness grade for a product by EAN
   * Returns the grade if found, otherwise null
   */
  getGradeByEan(ean: string | number | undefined | null): string | null {
    const normalized = this.normalizeEan(ean)
    if (!normalized) return null
    return this.gradeMap.get(normalized) || null
  }

  /**
   * Get healthiness grade for a product
   * Looks up by product.barcode, product.ean, or any ean/barcode field
   * Falls back to null if not found or if product doesn't have barcode/ean
   */
  getGradeForProduct(product: Product): string | null {
    // Try multiple possible field names for EAN/barcode
    // Products might have it as barcode, ean, or stored in nested fields
    const productAny = product as any
    const ean = productAny.barcode || productAny.ean || product.barcode || (product as any).ean
    
    if (!ean) {
      return null
    }
    
    const normalized = this.normalizeEan(ean)
    if (!normalized) {
      return null
    }
    
    const grade = this.gradeMap.get(normalized) || null
    return grade
  }

  /**
   * Get all cached grades as a map
   */
  getAllGrades(): Map<string, string> {
    return new Map(this.gradeMap)
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
   * Force refresh healthiness grades from Firestore
   * Useful for manual refresh or after grade updates
   */
  async refresh(): Promise<void> {
    this.initialized = false
    await this.initialize()
  }

  /**
   * Clear the healthiness grade store
   * Resets to initial state
   */
  clear(): void {
    this.gradeMap.clear()
    this.loading = false
    this.error = null
    this.initialized = false
    this.notifyListeners()
  }

  /**
   * Subscribe to store changes
   * Returns unsubscribe function
   */
  subscribe(listener: Listener): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  /**
   * Notify all listeners of state changes
   * Called internally when grades/loading/error state changes
   */
  private notifyListeners(): void {
    this.listeners.forEach(listener => listener())
  }

  /**
   * Get current store state for debugging
   */
  getState() {
    return {
      gradeCount: this.gradeMap.size,
      loading: this.loading,
      error: this.error,
      initialized: this.initialized,
      listenerCount: this.listeners.size
    }
  }
}

// Export singleton instance
export const healthinessGradeStore = new HealthinessGradeStore()
