// Type definitions for the vending machine application

export interface MachineSlot {
  position: string // e.g., "A0", "B3"
  product_name: string
  image_url: string
  price: number
  stock_current: number
  stock_max: number
  category: string
  width: number // Number of columns this product spans (mapped from w_spec)
  w_spec?: number // Optional: Width specification from Firestore (1, 1.5, 2, 3)
  is_discount: boolean
  discount: number
}

export interface MachineLocation {
  address: string
  city?: string
  country?: string
}

export interface AppMachine {
  id: string
  machine_name: string
  machine_model: string
  machine_key: string
  machine_sub_group: string
  last_sale: string
  n_sales: number
  refillers: string[]
  slots: MachineSlot[]
  location?: MachineLocation
}

export interface Product {
  id: string // Firestore document ID
  product_name: string // Primary name from Firestore
  name: string // Alias for product_name (backward compatibility)
  width: number // Width from Firestore (1, 1.5, 2, 3)
  price: number
  category: string
  image?: string // Generated Firebase Storage URL

  // Extensible for additional Firestore fields
  description?: string
  barcode?: string
  supplier?: string
  weight?: number
  nutritional_info?: Record<string, any>
  [key: string]: any // Allow any additional Firestore fields
}

// Firestore product document structure
export interface FirestoreProduct {
  product_name: string
  width: number
  price?: number
  category?: string
  description?: string
  barcode?: string
  supplier?: string
  weight?: number
  nutritional_info?: Record<string, any>
  [key: string]: any // Map all Firestore fields
}
