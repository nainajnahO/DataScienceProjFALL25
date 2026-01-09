import type { Product } from "../../lib/types"
import type { MachineSlot } from "../../lib/types"

interface ProductConfirmationDialogProps {
  isOpen: boolean
  currentSlot: MachineSlot | undefined
  pendingProduct: Product | null
  onConfirm: () => void
  onCancel: () => void
}

export default function ProductConfirmationDialog({
  isOpen,
  currentSlot,
  pendingProduct,
  onConfirm,
  onCancel,
}: ProductConfirmationDialogProps) {
  if (!isOpen || !pendingProduct) return null

  return (
    <div className="absolute inset-0 bg-black/60 flex items-center justify-center p-4 z-10">
      <div className="bg-popover border-2 border-border rounded-lg p-6 max-w-md w-full shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-xl font-bold text-popover-foreground mb-4">Are you sure?</h3>
        <p className="text-muted-foreground mb-2">
          You are about to replace <span className="font-semibold text-foreground">{currentSlot?.product_name}</span> with <span className="font-semibold text-foreground">{pendingProduct.name}</span>.
        </p>
        <p className="text-sm text-muted-foreground mb-6">
          This will change the product in this slot.
        </p>
        <div className="flex gap-3">
          <button
            onClick={onCancel}
            className="flex-1 px-4 py-2 bg-background border border-input rounded-lg hover:bg-accent text-foreground font-semibold transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold transition-colors"
          >
            Replace
          </button>
        </div>
      </div>
    </div>
  )
}

