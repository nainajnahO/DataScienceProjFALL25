interface LocationHelpModalProps {
  onClose: () => void
}

export default function LocationHelpModal({ onClose }: LocationHelpModalProps) {
  return (
    <>
      <div 
        className="fixed inset-0 bg-black/50 z-40"
        onClick={onClose}
      />
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        <div className="bg-card dark:bg-card border border-border rounded-lg shadow-xl max-w-lg w-full p-6 relative">
          <button
            onClick={onClose}
            className="absolute top-4 right-4 text-muted-foreground hover:text-card-foreground transition-colors"
            aria-label="Close"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
          
          <div className="flex items-start gap-3 pr-8">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-500 dark:text-blue-400 mt-0.5 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
            </svg>
            <div className="flex-1">
              <h4 className="text-lg font-semibold text-card-foreground mb-3">What is Location?</h4>
              <p className="text-sm text-muted-foreground mb-4">
                Select the type of location where your vending machine is placed. This helps the system recommend products that are popular and suitable for that specific location type.
              </p>
              <p className="text-sm text-card-foreground">
                Different locations attract different customers with different preferences. For example, products that sell well at a gym might be different from products that sell well at a school. By selecting the correct location type, you'll get better product recommendations tailored to your customers.
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

