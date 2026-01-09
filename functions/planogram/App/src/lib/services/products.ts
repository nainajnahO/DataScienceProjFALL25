import { db } from "../firebase"
import { collection, getDocs, getDoc, doc } from "firebase/firestore"
import type { Product, FirestoreProduct } from "../types"
import { generateProductImageUrl } from "../utils/storage"

/**
 * Firestore collection name for products
 */
const COLLECTION_NAME = "products"

/**
 * Maps Firestore product document to Product interface
 * Automatically generates Firebase Storage URL from product_name
 *
 * @param docId - Firestore document ID
 * @param data - Firestore document data
 * @returns Product object with generated image URL
 */
function mapFirestoreProduct(docId: string, data: FirestoreProduct): Product {
  const productName = data.product_name || ""

  return {
    id: docId,
    product_name: productName,
    name: productName, // Alias for backward compatibility
    width: data.width || 1,
    price: data.price || (() => {
      const w = data.width || 1
      if (w === 1) return 15
      if (w === 1.5) return 20
      if (w === 2) return 30
      if (w === 3) return 70
      return 15 // Default fallback
    })(),
    category: data.category || "Uncategorized",
    image: generateProductImageUrl(productName),

    // Map all other Firestore fields
    description: data.description,
    barcode: data.barcode,
    supplier: data.supplier,
    weight: data.weight,
    nutritional_info: data.nutritional_info,

    // Spread any additional fields not explicitly defined
    ...Object.entries(data).reduce((acc, [key, value]) => {
      const knownFields = [
        'product_name', 'width', 'price', 'category',
        'description', 'barcode', 'supplier', 'weight', 'nutritional_info'
      ]
      if (!knownFields.includes(key)) {
        acc[key] = value
      }
      return acc
    }, {} as Record<string, any>)
  }
}

/**
 * Fetches all products from Firestore "products" collection
 * Maps each document to Product interface with generated image URLs
 *
 * @returns Promise<Product[]> - Array of products with Firebase Storage URLs
 * @throws Error if Firestore query fails
 *
 * @example
 * const products = await getAllProducts()
 * // products[0].image: "https://firebasestorage.googleapis.com/v0/b/.../o/product_images%2FCoca-Cola.png?alt=media"
 */
export async function getAllProducts(): Promise<Product[]> {
  try {
    const colRef = collection(db, COLLECTION_NAME)
    const snapshot = await getDocs(colRef)

    if (snapshot.empty) {
      console.warn(`No products found in Firestore collection "${COLLECTION_NAME}"`)
      return []
    }

    const products = snapshot.docs.map((doc) => {
      const data = doc.data() as FirestoreProduct
      return mapFirestoreProduct(doc.id, data)
    })

    console.log(`✅ Loaded ${products.length} products from Firestore`)
    return products
  } catch (error) {
    console.error("❌ Error fetching products from Firestore:", error)
    throw new Error(`Failed to fetch products: ${error}`)
  }
}

/**
 * Fetches a single product by Firestore document ID
 *
 * @param productId - Firestore document ID
 * @returns Promise<Product | null> - Product or null if not found
 */
export async function getProductById(productId: string): Promise<Product | null> {
  try {
    const docRef = doc(db, COLLECTION_NAME, productId)
    const snapshot = await getDoc(docRef)

    if (snapshot.exists()) {
      const data = snapshot.data() as FirestoreProduct
      return mapFirestoreProduct(snapshot.id, data)
    }

    console.warn(`Product with ID "${productId}" not found`)
    return null
  } catch (error) {
    console.error(`Error fetching product ${productId}:`, error)
    return null
  }
}
