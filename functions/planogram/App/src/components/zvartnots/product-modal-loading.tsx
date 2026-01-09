interface ProductModalLoadingProps {
  onClose: () => void
}

export function ProductModalLoading({ onClose: _onClose }: ProductModalLoadingProps) {
  return (
    <div className="fixed inset-0 bg-black/50 dark:bg-black/70 flex items-center justify-center z-50">
      <div className="bg-popover border border-border rounded-lg p-8">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-lg text-popover-foreground">Loading products...</p>
        </div>
      </div>
    </div>
  )
}

export function ProductModalError({ error, onClose }: { error: string; onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/50 dark:bg-black/70 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-popover border border-border rounded-lg p-8 max-w-md" onClick={(e) => e.stopPropagation()}>
        <div className="text-center">
          <p className="text-lg text-red-600 dark:text-red-400 mb-4">Failed to load products</p>
          <p className="text-sm text-muted-foreground mb-6">{error}</p>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

