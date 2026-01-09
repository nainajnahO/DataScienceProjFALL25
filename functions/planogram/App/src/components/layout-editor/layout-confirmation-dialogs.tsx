interface ConfirmationAction {
  type: "merge-double" | "merge-2-opt-3" | "create-triple-or-pair" | "split" | "split-triple" | "split-pair"
  data: any
}

interface LayoutConfirmationDialogsProps {
  showConfirmation: boolean
  confirmationAction: ConfirmationAction | null
  onAction: (actionChoice?: string) => void
  onCancel: () => void
}

export default function LayoutConfirmationDialogs({
  showConfirmation,
  confirmationAction,
  onAction,
  onCancel,
}: LayoutConfirmationDialogsProps) {
  if (!showConfirmation || !confirmationAction) return null

  return (
    <div className="fixed inset-0 bg-black/60 dark:bg-black/70 flex items-center justify-center z-50">
      <div className="bg-popover rounded-lg shadow-2xl border-2 border-border p-6 w-96 mx-4">
        {confirmationAction.type === "create-triple-or-pair" ? (
          <>
            <h2 className="text-xl font-semibold text-popover-foreground mb-4">Merge 3 Slots</h2>
            <p className="text-muted-foreground mb-6">Choose layout:</p>
            <div className="flex flex-col gap-3">
              <button
                onClick={() => onAction("triple")}
                className="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-md font-medium text-lg transition-colors"
              >
                Create Triple Slot (1 Product)
              </button>
              <button
                onClick={() => onAction("pair")}
                className="px-8 py-4 bg-orange-600 hover:bg-orange-700 text-white rounded-md font-medium text-lg transition-colors"
              >
                Create 1.5 Pair (2 Products)
              </button>
              <button
                onClick={onCancel}
                className="px-8 py-4 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-md font-medium text-lg transition-colors"
              >
                Cancel
              </button>
            </div>
          </>
        ) : confirmationAction.type === "merge-2-opt-3" ? (
          <>
            <h2 className="text-xl font-semibold text-popover-foreground mb-4">Merge Options</h2>
            <p className="text-muted-foreground mb-6">Create a Double Slot (2 slots) or a 1.5 Pair (requires 3rd neighbor).</p>
            <div className="flex flex-col gap-3">
              <button onClick={() => onAction("double")} className="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-md font-medium text-lg transition-colors">Create Double Slot</button>
              <button
                onClick={() => onAction("pair-1.5")}
                disabled={confirmationAction.data.extension === null}
                className={`px-8 py-4 rounded-md font-medium text-lg transition-colors ${confirmationAction.data.extension === null
                  ? "bg-gray-400 text-gray-200 cursor-not-allowed opacity-50"
                  : "bg-orange-600 hover:bg-orange-700 text-white"
                  }`}
              >
                Create 1.5 Pair (3 Slots)
              </button>
              <button onClick={onCancel} className="px-8 py-4 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-md font-medium text-lg transition-colors">Cancel</button>
            </div>
          </>
        ) : confirmationAction.type === "split-pair" ? (
          <>
            <h2 className="text-xl font-semibold text-popover-foreground mb-4">Split 1.5 Pair?</h2>
            <p className="text-muted-foreground mb-6">This will remove both 1.5-width slots and revert to 3 single slots.</p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={onCancel}
                className="px-8 py-4 bg-secondary hover:bg-secondary/80 text-secondary-foreground rounded-md font-medium text-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => onAction()}
                className="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-md font-medium text-lg transition-colors"
              >
                Confirm
              </button>
            </div>
          </>
        ) : confirmationAction.type === "split-triple" ? (
          <>
            <h2 className="text-xl font-semibold text-popover-foreground mb-4">Split Triple Slot?</h2>
            <p className="text-muted-foreground mb-6">Choose how to split this triple-width slot:</p>
            <div className="flex flex-col gap-3">
              <button
                onClick={() => onAction("singles")}
                className="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-md font-medium text-lg transition-colors"
              >
                Split into 3 Singles
              </button>
              <button
                onClick={() => onAction("1.5s")}
                className="px-8 py-4 bg-orange-600 hover:bg-orange-700 text-white rounded-md font-medium text-lg transition-colors"
              >
                Split into Two 1.5-Width Slots
              </button>
              <button
                onClick={onCancel}
                className="px-8 py-4 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-md font-medium text-lg transition-colors"
              >
                Cancel
              </button>
            </div>
          </>
        ) : (
          <>
            <h2 className="text-xl font-semibold text-popover-foreground mb-4">
              Split Slot?
            </h2>
            <p className="text-muted-foreground mb-6">
              Are you sure you want to split this merged slot back into individual slots? Subsequent slots will be renumbered.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={onCancel}
                className="px-8 py-4 bg-secondary hover:bg-secondary/80 text-secondary-foreground rounded-md font-medium text-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => onAction()}
                className="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-md font-medium text-lg transition-colors"
              >
                Confirm
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

