interface LayoutDisabledMessageProps {
  show: boolean
  onClose: () => void
}

export default function LayoutDisabledMessage({ show, onClose }: LayoutDisabledMessageProps) {
  if (!show) return null

  return (
    <div className="fixed inset-0 bg-black/60 dark:bg-black/70 flex items-center justify-center z-50">
      <div className="bg-popover rounded-lg shadow-2xl border-2 border-border p-6 w-96 mx-4">
        <h2 className="text-xl font-semibold text-popover-foreground mb-4">Cannot Change Layout</h2>
        <p className="text-muted-foreground mb-6">Slots containing products cannot be merged or split. Please remove the products first.</p>
        <div className="flex justify-end">
          <button
            onClick={onClose}
            className="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-md font-medium text-lg transition-colors"
          >
            OK
          </button>
        </div>
      </div>
    </div>
  )
}

