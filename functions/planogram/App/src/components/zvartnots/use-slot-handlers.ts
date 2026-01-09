import { useState } from "react"
import type { MachineSlot, Product } from "../../lib/types"

interface UseSlotHandlersProps {
  slots: MachineSlot[]
  setSlots: React.Dispatch<React.SetStateAction<MachineSlot[]>>
  products: Product[]
  onError?: (message: string) => void
}

export function useSlotHandlers({ slots: _slots, setSlots, products, onError }: UseSlotHandlersProps) {
  const [selectedPosition, setSelectedPosition] = useState<string | null>(null)

  const handleSelectProduct = (product: Product) => {
    if (selectedPosition !== null) {
      setSlots((prev) => {
        const existingSlotIndex = prev.findIndex((slot) => slot.position === selectedPosition)

        if (existingSlotIndex >= 0) {
          return prev.map((slot) =>
            slot.position === selectedPosition
              ? {
                ...slot,
                product_name: product.name,
                image_url: product.image || "",
                price: product.price,
                category: product.category,
                stock_current: slot.product_name !== product.name ? 10 : slot.stock_current,
                stock_max: 10,
              }
              : slot,
          )
        } else {
          const newSlot: MachineSlot = {
            position: selectedPosition,
            product_name: product.name,
            image_url: product.image || "",
            price: product.price,
            category: product.category,
            stock_current: 10,
            stock_max: 10,
            width: 1,
            is_discount: false,
            discount: 0,
          }
          return [...prev, newSlot]
        }
      })
    }
  }

  const handleMoveProduct = (fromPosition: string, toPosition: string) => {
    setSlots((prev) => {
      if (fromPosition === toPosition) return prev

      const fromIndex = prev.findIndex((s) => s.position === fromPosition)
      const toIndex = prev.findIndex((s) => s.position === toPosition)

      if (fromIndex === -1) return prev

      const oldSlots = [...prev]
      const fromSlot = oldSlots[fromIndex]
      const getProductFields = (slot: MachineSlot) => ({
        product_name: slot.product_name,
        image_url: slot.image_url,
        price: slot.price,
        category: slot.category,
        stock_current: slot.stock_current,
        stock_max: slot.stock_max,
      })

      const emptyProductFields = {
        product_name: "",
        image_url: "",
        price: 0,
        category: "",
        stock_current: 0,
        stock_max: 10,
      }

      const sourceFields = getProductFields(fromSlot)
      const sourceProductDef = products.find(p => p.name === sourceFields.product_name)
      const sourceWidth = sourceProductDef?.width || 1

      if (toIndex >= 0) {
        const toSlot = oldSlots[toIndex]
        const targetFields = getProductFields(toSlot)
        const targetProductDef = products.find(p => p.name === targetFields.product_name)
        const targetWidth = targetProductDef?.width || 1

        const targetSlotWidth = toSlot.width || 1
        if (sourceWidth > targetSlotWidth) {
          alert(`Cannot move "${sourceFields.product_name}" (Width ${sourceWidth}) into slot ${toSlot.position} (Width ${targetSlotWidth})`)
          return prev
        }

        if (targetFields.product_name) {
          const sourceSlotWidth = fromSlot.width || 1
          if (targetWidth > sourceSlotWidth) {
            alert(`Cannot move "${targetFields.product_name}" (Width ${targetWidth}) into slot ${fromSlot.position} (Width ${sourceSlotWidth})`)
            return prev
          }
        }

        oldSlots[toIndex] = { ...toSlot, ...sourceFields }
        oldSlots[fromIndex] = { ...fromSlot, ...targetFields }
      } else {
        if (sourceWidth > 1) {
          alert(`Cannot move wide product to undefined slot.`)
          return prev
        }

        oldSlots.push({
          position: toPosition,
          width: 1,
          is_discount: false,
          discount: 0,
          ...sourceFields,
        })

        oldSlots[fromIndex] = { ...fromSlot, ...emptyProductFields }
      }

      return oldSlots
    })
  }

  const handleDropProduct = (product: Product, toPosition: string) => {
    setSlots((prev) => {
      const toIndex = prev.findIndex((s) => s.position === toPosition)
      const productWidth = product.width || 1

      if (toIndex >= 0) {
        const toSlot = prev[toIndex]
        const targetSlotWidth = toSlot.width || 1

        if (productWidth > targetSlotWidth) {
          if (onError) {
            onError(`Cannot place "${product.name}" (Width ${productWidth}) into slot ${toSlot.position} (Width ${targetSlotWidth})`)
          }
          return prev
        }

        const oldSlots = [...prev]
        const existingProduct = oldSlots[toIndex].product_name

        oldSlots[toIndex] = {
          ...toSlot,
          product_name: product.name,
          image_url: product.image || "",
          price: product.price,
          category: product.category,
          stock_current: existingProduct !== product.name ? 10 : toSlot.stock_current,
          stock_max: 10,
        }

        return oldSlots
      } else {
        if (productWidth > 1) {
          if (onError) {
            onError(`Cannot place wide product to undefined slot.`)
          }
          return prev
        }

        const newSlot: MachineSlot = {
          position: toPosition,
          product_name: product.name,
          image_url: product.image || "",
          price: product.price,
          category: product.category,
          stock_current: 10,
          stock_max: 10,
          width: 1,
          is_discount: false,
          discount: 0,
        }

        return [...prev, newSlot]
      }
    })
  }

  const handleUpdateStock = (position: string, newStock: number) => {
    setSlots((prev) =>
      prev.map((slot) =>
        slot.position === position
          ? { ...slot, stock_current: newStock }
          : slot
      )
    )
  }

  const handleRemoveProduct = (position: string) => {
    setSlots((prev) =>
      prev.map((slot) =>
        slot.position === position
          ? {
            ...slot,
            product_name: "",
            image_url: "",
            price: 0,
            category: "",
            stock_current: 0,
            stock_max: 10,
          }
          : slot
      )
    )
  }

  return {
    selectedPosition,
    setSelectedPosition,
    handleSelectProduct,
    handleMoveProduct,
    handleDropProduct,
    handleUpdateStock,
    handleRemoveProduct,
  }
}

