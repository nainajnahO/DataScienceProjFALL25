import { useState } from "react"
import { getPlaceholderImage } from "@/lib/utils/storage"

interface ProductImageProps {
  src?: string
  alt: string
  className?: string
  onError?: () => void
}

/**
 * Product image component with automatic fallback to placeholder
 * Handles missing or broken image URLs gracefully
 *
 * @example
 * <ProductImage
 *   src={product.image}
 *   alt={product.name}
 *   className="w-12 h-12 object-contain"
 * />
 */
export default function ProductImage({ src, alt, className, onError }: ProductImageProps) {
  const [imageSrc, setImageSrc] = useState(src || getPlaceholderImage())
  const [hasError, setHasError] = useState(false)

  const handleError = () => {
    if (!hasError) {
      setHasError(true)
      setImageSrc(getPlaceholderImage())
      onError?.()
      console.warn(`Failed to load image for "${alt}": ${src}`)
    }
  }

  return (
    <img
      src={imageSrc}
      alt={alt}
      className={className}
      onError={handleError}
      loading="lazy" // Lazy load images for better performance
    />
  )
}
