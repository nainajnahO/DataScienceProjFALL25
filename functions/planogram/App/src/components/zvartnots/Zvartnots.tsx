import { useState, useEffect, useRef, useMemo } from "react"
import type { AppMachine } from "../../lib/types"
import VendingMachineWindow from "../vending-machine/VendingMachineWindow.tsx"
import ProductModal from "./product-modal.tsx"
import RightSideTabs from "./right-side-tabs.tsx"
import ErrorToast from "./error-toast.tsx"
import { useProducts } from "@/hooks/useProducts"
import { calculateGridDimensions } from "./autofill-logic.ts"
import { getHealthinessGrade, getGradeFromScore } from "./healthiness-utils"
import VendingMachineControls from "./vending-machine-controls.tsx"
import TrashZoneOverlay from "./trash-zone-overlay.tsx"
import EditLayoutButton from "./edit-layout-button.tsx"
import { useSlotHandlers } from "./use-slot-handlers.ts"

interface ZvartnotsProps {
  machine: AppMachine
}

export default function Zvartnots({ machine }: ZvartnotsProps) {
  const { products } = useProducts()
  const { rows, cols } = calculateGridDimensions(machine.slots)
  const [slots, setSlots] = useState(machine.slots)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [selectedLocation, setSelectedLocation] = useState<string | null>(null)
  const [address, setAddress] = useState("")
  const [isDraggingProduct, setIsDraggingProduct] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [showError, setShowError] = useState(false)

  useEffect(() => {
    setSlots(machine.slots)
  }, [machine.slots])

  const planogramRef = useRef<HTMLDivElement>(null)

  const showErrorToast = (message: string) => {
    setErrorMessage(message)
    setShowError(true)
  }

  const {
    selectedPosition,
    setSelectedPosition,
    handleSelectProduct,
    handleMoveProduct,
    handleDropProduct,
    handleUpdateStock,
    handleRemoveProduct,
  } = useSlotHandlers({ slots, setSlots, products, onError: showErrorToast })

  const handleDownload = async () => {
    if (planogramRef.current) {
      try {
        const rect = planogramRef.current.getBoundingClientRect()
        // @ts-ignore
        const result = await window.ipcRenderer.invoke('capture-planogram', {
          x: rect.x + window.scrollX,
          y: rect.y + window.scrollY,
          width: rect.width,
          height: rect.height
        })

        if (!result.success) {
          if (result.error !== 'Cancelled') {
            alert(`Failed to save: ${result.error}`)
          }
        }
      } catch (err: any) {
        console.error("Failed to download planogram:", err)
        alert(`Failed to trigger download: ${err?.message || err}`)
      }
    }
  }

  const handleSlotClick = (position: string) => {
    setSelectedPosition(position)
    setIsModalOpen(true)
  }

  const handleCloseModal = () => {
    setIsModalOpen(false)
    setSelectedPosition(null)
  }

  const [isProcessing, setIsProcessing] = useState(false)

  const enrichSlots = (newSlots: any[]) => {
    if (!products) return newSlots
    return newSlots.map(slot => {
      if (!slot.product_name) return slot

      // Try to find matching product in catalogue
      let match = products.find(p => String(p.ean) === String(slot.ean))
      if (!match && slot.product_name) {
        match = products.find(p => (p.name || p.product_name) === slot.product_name)
      }

      if (match) {
        return {
          ...slot,
          image_url: match.image_url || match.image || slot.image_url,
          price: match.price || slot.price // Use catalogue price (retail) if available
        }
      }
      return slot
    })
  }

  const handleAutofill = async (weights: any) => {
    if (!selectedLocation || !address || isProcessing) return
    setIsProcessing(true)
    try {
      // @ts-ignore
      const result = await window.ipcRenderer.invoke('autofill-machine', {
        slots,
        machineId: selectedLocation,
        action: 'fill',
        address,
        static_weights: weights?.static_weights,
        dynamic_weights: weights?.dynamic_weights
      })
      if (result.success && result.slots) {
        const enriched = enrichSlots(result.slots)
        setSlots(enriched)
      } else {
        showErrorToast(result.error || "Autofill failed")
      }
    } catch (err: any) {
      showErrorToast(`Autofill error: ${err.message}`)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleOptimize = async (weights: any) => {
    if (!selectedLocation || !address || isProcessing) return
    setIsProcessing(true)
    try {
      // @ts-ignore
      const result = await window.ipcRenderer.invoke('autofill-machine', {
        slots,
        machineId: selectedLocation,
        action: 'optimize',
        address,
        static_weights: weights?.static_weights,
        dynamic_weights: weights?.dynamic_weights
      })
      if (result.success && result.slots) {
        const enriched = enrichSlots(result.slots)
        setSlots(enriched)
        alert("Optimization complete! Check the updated planogram.")
      } else {
        showErrorToast(result.error || "Optimization failed")
      }
    } catch (err: any) {
      showErrorToast(`Optimization error: ${err.message}`)
    } finally {
      setIsProcessing(false)
    }
  }

  const currentSlot = selectedPosition ? slots.find(slot => slot.position === selectedPosition) : undefined

  const machineHealthinessGrade = useMemo(() => {
    const filledSlots = slots.filter(slot => slot.product_name && slot.product_name !== '')
    if (filledSlots.length === 0) return null

    const healthinessScores: number[] = []
    filledSlots.forEach(slot => {
      const product = products.find(p => p.name === slot.product_name || p.product_name === slot.product_name)
      if (product) {
        const grade = getHealthinessGrade(product)
        if (grade) {
          const score = ['A', 'B', 'C', 'D', 'E'].indexOf(grade)
          if (score !== -1) {
            healthinessScores.push(score)
          }
        }
      }
    })

    if (healthinessScores.length === 0) return null

    const average = healthinessScores.reduce((sum, score) => sum + score, 0) / healthinessScores.length
    return getGradeFromScore(average)
  }, [slots, products])

  const handleProductDragStart = () => {
    setIsDraggingProduct(true)
  }

  const handleProductDragEnd = () => {
    setIsDraggingProduct(false)
  }

  const handleDropOnTrash = (position: string) => {
    handleRemoveProduct(position)
    setIsDraggingProduct(false)
  }

  // Calculate products currently placed in the machine
  const placedProducts = useMemo(() => {
    if (!products || !slots) return []
    // Get unique product names from filled slots
    const placedNames = new Set(
      slots
        .filter(s => s.product_name && s.product_name.trim() !== "")
        .map(s => s.product_name)
    )
    // Filter full products list
    return products.filter(p => placedNames.has(p.name || p.product_name))
  }, [slots, products])

  return (
    <div className="p-4 md:p-6 min-h-screen overflow-auto bg-background text-foreground relative">
      <div className="flex flex-col lg:flex-row gap-6 lg:gap-8 lg:items-start relative">
        <div ref={planogramRef} className="flex-shrink-0 w-full lg:flex-1 transition-all duration-300 relative z-50 flex items-center justify-center flex-col">
          <div className="w-full max-w-full md:max-w-4xl">
            <VendingMachineWindow
              rows={rows}
              cols={cols}
              slots={slots}
              onSlotClick={handleSlotClick}
              onMoveProduct={handleMoveProduct}
              onDropProduct={handleDropProduct}
              onProductDragStart={handleProductDragStart}
              onProductDragEnd={handleProductDragEnd}
            />

            <VendingMachineControls
              onDownload={handleDownload}
              healthinessGrade={machineHealthinessGrade}
              editLayoutAction={<EditLayoutButton machine={machine} slots={slots} />}
            />
          </div>
        </div>

        <div className={`flex flex-col gap-6 lg:gap-8 flex-1 min-h-0 transition-opacity duration-300 ${isDraggingProduct ? 'opacity-30 blur-sm pointer-events-none' : 'opacity-100'}`}>
          <div className="min-h-[680px] max-h-[80vh] flex flex-col overflow-hidden">
            <RightSideTabs
              selectedLocation={selectedLocation}
              onLocationChange={setSelectedLocation}
              products={products || []}
              placedProducts={placedProducts}
              address={address}
              onAddressChange={setAddress}
              onProductDragStart={handleProductDragStart}
              onProductDragEnd={handleProductDragEnd}
              onAutofill={handleAutofill}
              onOptimize={handleOptimize}
            />
          </div>
          <div className="flex flex-col sm:flex-row gap-4 lg:justify-start flex-shrink-0">
          </div>
        </div>
      </div>

      <TrashZoneOverlay
        isDraggingProduct={isDraggingProduct}
        onDrop={handleDropOnTrash}
      />

      <ProductModal
        isOpen={isModalOpen}
        onClose={handleCloseModal}
        onSelectProduct={handleSelectProduct}
        onUpdateStock={handleUpdateStock}
        onRemoveProduct={handleRemoveProduct}
        currentSlot={currentSlot}
        selectedPosition={selectedPosition}
      />

      <ErrorToast
        message={errorMessage || ""}
        isVisible={showError}
        onClose={() => {
          setShowError(false)
          setErrorMessage(null)
        }}
      />
    </div>
  )
}
