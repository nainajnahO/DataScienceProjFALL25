/**
 * Firebase Storage URL generation utilities
 * Handles mapping product names to Firebase Storage image URLs
 */

const STORAGE_BUCKET = "uno-y-b48fb.appspot.com"
const PRODUCT_IMAGES_PATH = "product_images"
const PLACEHOLDER_IMAGE = "/placeholder.svg"

/**
 * Generates Firebase Storage URL from product name
 * Properly encodes special characters (spaces, ä, ö, !, etc.)
 *
 * @param productName - Product name from Firestore (e.g., "ICA Soppa Jordärtskocka")
 * @returns Firebase Storage URL with alt=media parameter
 *
 * @example
 * generateProductImageUrl("Coca-Cola 33cl")
 * // Returns: "https://firebasestorage.googleapis.com/v0/b/uno-y-b48fb.appspot.com/o/product_images%2FCoca-Cola%2033cl.png?alt=media"
 */
export function generateProductImageUrl(productName: string): string {
  if (!productName || productName.trim() === "") {
    return PLACEHOLDER_IMAGE
  }

  // URL encode the product name to handle special characters
  const encodedName = encodeURIComponent(productName)

  return `https://firebasestorage.googleapis.com/v0/b/${STORAGE_BUCKET}/o/${PRODUCT_IMAGES_PATH}%2F${encodedName}.png?alt=media`
}

/**
 * Returns the placeholder image path
 * Used as fallback when product image is missing or fails to load
 */
export function getPlaceholderImage(): string {
  return PLACEHOLDER_IMAGE
}

/**
 * Validates if an image URL is accessible
 * Useful for error handling and debugging
 *
 * @param url - Image URL to validate
 * @returns Promise that resolves to true if image is accessible
 */
export async function validateImageUrl(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: 'HEAD' })
    return response.ok
  } catch {
    return false
  }
}
