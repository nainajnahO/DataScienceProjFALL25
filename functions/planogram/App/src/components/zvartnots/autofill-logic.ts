import type { MachineSlot, Product } from "../../lib/types"

export function calculateGridDimensions(slots: MachineSlot[]): { rows: number; cols: number } {
  let maxRow = 0
  let maxCol = 0

  if (slots && slots.length > 0) {
    slots.forEach((slot) => {
      const row = slot.position.charCodeAt(0) - 65
      const col = Number.parseInt(slot.position.slice(1))
      const width = slot.width || 1
      const occupied = Math.ceil(width)
      const endCol = col + occupied - 1

      if (row > maxRow) maxRow = row
      if (endCol > maxCol) maxCol = endCol
    })
  }

  return {
    rows: Math.max(maxRow + 1, 5),
    cols: Math.max(maxCol + 1, 8),
  }
}

export function getCoveredPositions(slots: MachineSlot[], cols: number, slotMap: Map<string, MachineSlot>): Set<string> {
  const covered = new Set<string>()

  slots.forEach((slot) => {
    if (slot.width > 1) {
      const row = slot.position.charCodeAt(0) - 65
      const col = Number.parseInt(slot.position.slice(1))

      if (slot.width === 1.5) {
        const nextCol = col + 1
        if (nextCol < cols) {
          const nextPos = String.fromCharCode(65 + row) + nextCol
          if (!slotMap.has(nextPos)) {
            covered.add(nextPos)
          }
        }
      } else {
        for (let i = 1; i < slot.width; i++) {
          const coveredCol = col + i
          if (coveredCol < cols) {
            const coveredPos = String.fromCharCode(65 + row) + coveredCol
            covered.add(coveredPos)
          }
        }
      }
    }
  })

  return covered
}

export function generateAllPositions(rows: number, cols: number, coveredPositions: Set<string>): string[] {
  const allPositions: string[] = []
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const position = String.fromCharCode(65 + row) + col
      if (!coveredPositions.has(position)) {
        allPositions.push(position)
      }
    }
  }
  return allPositions
}

export function autofillSlots(
  slots: MachineSlot[],
  products: Product[],
  rows: number,
  cols: number
): MachineSlot[] {
  const slotMap = new Map<string, MachineSlot>()
  slots.forEach(slot => {
    slotMap.set(slot.position, slot)
  })

  const coveredPositions = getCoveredPositions(slots, cols, slotMap)
  const allPositions = generateAllPositions(rows, cols, coveredPositions)

  const sortedProducts = [...products]
    .sort((a, b) => 
      (a.name || a.product_name || '').localeCompare(b.name || b.product_name || '')
    )
    .slice(0, 50)

  const updatedSlots: MachineSlot[] = []
  const usedProductIndices = new Set<number>()

  allPositions.forEach(position => {
    const existingSlot = slotMap.get(position)
    
    if (existingSlot && existingSlot.product_name) {
      updatedSlots.push(existingSlot)
      return
    }

    const slotWidth = existingSlot?.width || 1

    let productAssigned = false
    for (let i = 0; i < sortedProducts.length; i++) {
      if (usedProductIndices.has(i)) continue

      const product = sortedProducts[i]
      const productWidth = product.width || 1

      if (productWidth <= slotWidth) {
        const newSlot: MachineSlot = {
          position,
          product_name: product.name || product.product_name,
          image_url: product.image || '',
          price: product.price || 0,
          category: product.category || '',
          stock_current: 10,
          stock_max: 10,
          width: slotWidth,
          is_discount: false,
          discount: 0,
        }
        updatedSlots.push(newSlot)
        usedProductIndices.add(i)
        productAssigned = true
        break
      }
    }

    if (!productAssigned && existingSlot) {
      updatedSlots.push(existingSlot)
    }
  })

  return updatedSlots
}

