import { useState, useEffect } from "react"

interface UseImageRotationProps {
  imageUrl: string | undefined
  hasProduct: boolean
  slotWidth: number
}

export function useImageRotation({ imageUrl, hasProduct, slotWidth }: UseImageRotationProps) {
  const [displayImageUrl, setDisplayImageUrl] = useState<string | undefined>(undefined)
  const [imageError, setImageError] = useState(false)

  useEffect(() => {
    setImageError(false)
    setDisplayImageUrl(imageUrl)
  }, [imageUrl])

  useEffect(() => {
    if (!imageUrl || !hasProduct) return

    const isSingleSlot = slotWidth <= 1.2
    if (!isSingleSlot) {
      setDisplayImageUrl(imageUrl)
      return
    }

    let isMounted = true
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.src = imageUrl

    img.onload = () => {
      if (!isMounted) return

      const isImageWide = img.naturalWidth > img.naturalHeight

      if (isImageWide) {
        const canvas = document.createElement('canvas')
        canvas.width = img.naturalHeight
        canvas.height = img.naturalWidth

        const ctx = canvas.getContext('2d')
        if (ctx) {
          ctx.translate(canvas.width / 2, canvas.height / 2)
          ctx.rotate(90 * Math.PI / 180)
          ctx.drawImage(img, -img.naturalWidth / 2, -img.naturalHeight / 2)

          try {
            const rotatedUrl = canvas.toDataURL()
            setDisplayImageUrl(rotatedUrl)
          } catch (e) {
            console.warn("Could not rotate image due to CORS:", e)
            setDisplayImageUrl(imageUrl)
          }
        }
      } else {
        setDisplayImageUrl(imageUrl)
      }
    }

    img.onerror = () => {
      if (isMounted) {
        setImageError(true)
        setDisplayImageUrl(undefined)
      }
    }

    return () => {
      isMounted = false
    }
  }, [imageUrl, hasProduct, slotWidth])

  return { displayImageUrl, imageError, setImageError }
}

