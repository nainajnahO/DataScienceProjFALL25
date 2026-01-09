import { useNavigate } from "react-router-dom"
import type { AppMachine } from "@/lib/types.ts"
import { useProducts } from "@/hooks/useProducts"
import LayoutEditor, { type EditorProduct, type MergedSlot } from "@/components/layout-editor/LayoutEditor.tsx"
import { newMachineStore } from "@/lib/newMachineStore.ts"

// Remove local merged slot definition as we imported it
// interface MergedSlot {
//   slots: number[]
//   product?: Product
//   width?: number
// }

interface LayoutEditorClientProps {
  machine: AppMachine
}

function calculateGridDimensions(machine: AppMachine): { rows: number; cols: number } {
  let maxRow = 0
  let maxCol = 0

  if (machine.slots && machine.slots.length > 0) {
    machine.slots.forEach((slot) => {
      if (!slot.position) return // Safety check

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
    rows: Math.max(maxRow + 1, 6),
    cols: Math.max(maxCol + 1, 8),
  }
}

export default function LayoutEditorClient({ machine }: LayoutEditorClientProps) {
  const navigate = useNavigate()
  const { products: _products, loading } = useProducts()

  const { rows, cols } = calculateGridDimensions(machine)

  // Convert machine slots to the format expected by LayoutEditor
  const initialSlots: Record<number, EditorProduct> = {}
  const initialMergedSlots: MergedSlot[] = []

  machine.slots.forEach((slot) => {
    if (!slot.position) return // Safety check

    const row = slot.position.charCodeAt(0) - 65
    const col = Number.parseInt(slot.position.slice(1))
    const index = row * cols + col

    if (slot.product_name) {
      initialSlots[index] = {
        id: `product-${index}`,
        name: slot.product_name,
        price: slot.price || 0,
        category: slot.category || "Unknown",
        image: slot.image_url,
      }
    }

    // Handle merged slots (width > 1 or width === 1.5)
    if (slot.width === 1.5) {
      // 1.5-width slots occupy a single grid position but render wider
      initialMergedSlots.push({
        slots: [index],
        product: initialSlots[index],
        width: 1.5,
      })
    } else if (slot.width > 1) {
      // Multi-slot items (2, 3, etc.) - these actually occupy multiple consecutive positions
      const slots: number[] = []
      for (let i = 0; i < slot.width; i++) {
        slots.push(index + i)
      }

      initialMergedSlots.push({
        slots,
        product: initialSlots[index],
      })

      // Remove products from merged slots except the first
      slots.slice(1).forEach((slotIndex) => {
        delete initialSlots[slotIndex]
      })
    }
  })

  const handleUpdateSlots = (slots: Record<number, EditorProduct>, mergedSlots: MergedSlot[]) => {
    console.log("[v0] Layout updated:", { slots, mergedSlots })

    // Convert back to MachineSlot[] format
    const updatedMachineSlots: typeof machine.slots = []

    // Track which indices are part of merged slots
    const mergedIndices = new Set<number>()

    // Helper to find existing slot data to preserve
    const getExistingSlot = (pos: string) => machine.slots.find(s => s.position === pos)

    // Process merged slots first
    mergedSlots.forEach((merged) => {
      const firstIndex = merged.slots[0]
      const row = Math.floor(firstIndex / cols)
      const col = firstIndex % cols
      const position = `${String.fromCharCode(65 + row)}${col}`
      const existing = getExistingSlot(position)

      // Determine width
      let width: number
      if (merged.width === 1.5) {
        width = 1.5
      } else if (merged.slots.length === 2) {
        width = 2
      } else if (merged.slots.length === 3) {
        width = 3
      } else {
        width = merged.slots.length
      }

      // Check if product changed to reset stock
      const hasProductChanged = existing?.product_name !== (merged.product?.name || "")
      const stockCurrent = hasProductChanged ? 10 : (existing?.stock_current ?? 0)
      const stockMax = existing?.stock_max ?? 10

      updatedMachineSlots.push({
        position,
        product_name: merged.product?.name || "",
        image_url: merged.product?.image || "",
        price: merged.product?.price || 0,
        category: merged.product?.category || "",
        stock_current: stockCurrent,
        stock_max: stockMax,
        width,
        is_discount: existing?.is_discount ?? false,
        discount: existing?.discount ?? 0,
      })

      // Mark all indices in this merge as processed
      merged.slots.forEach(idx => mergedIndices.add(idx))
    })

    // Process regular slots (not part of merges)
    Object.entries(slots).forEach(([indexStr, product]) => {
      const index = Number.parseInt(indexStr)
      if (mergedIndices.has(index)) return // Skip merged slots

      const row = Math.floor(index / cols)
      const col = index % cols
      const position = `${String.fromCharCode(65 + row)}${col}`
      const existing = getExistingSlot(position)

      // Check if product changed to reset stock
      const hasProductChanged = existing?.product_name !== product.name
      const stockCurrent = hasProductChanged ? 10 : (existing?.stock_current ?? 0)
      const stockMax = existing?.stock_max ?? 10

      updatedMachineSlots.push({
        position,
        product_name: product.name,
        image_url: product.image || "",
        price: product.price,
        category: product.category,
        stock_current: stockCurrent,
        stock_max: stockMax,
        width: 1,
        is_discount: existing?.is_discount ?? false,
        discount: existing?.discount ?? 0,
      })
    })

    // Create updated machine object
    const updatedMachine = {
      ...machine,
      slots: updatedMachineSlots,
    }

    // If this is the "new" machine, save to global store
    if (machine.id === "new") {
      newMachineStore.setMachine(updatedMachine)
    }

    // Navigate back
    if (machine.id === "new") {
      navigate("/")
    } else {
      navigate(`/machines/${machine.id}`)
    }
  }

  const handleBack = () => {
    if (machine.id === "new") {
      navigate("/")
    } else {
      navigate(`/machines/${machine.id}`)
    }
  }

  // Show loading state while products are being fetched
  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-lg">Loading products...</p>
        </div>
      </div>
    )
  }

  return (
    <LayoutEditor
      slots={initialSlots}
      mergedSlots={initialMergedSlots}
      onUpdateSlots={handleUpdateSlots}
      onBack={handleBack}
      rows={rows}
      cols={cols}
      machineName={machine.machine_name}
    />
  )
}
