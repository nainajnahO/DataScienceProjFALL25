interface ScoringHelpModalProps {
  onClose: () => void
}

export default function ScoringHelpModal({ onClose }: ScoringHelpModalProps) {
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
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            <div className="flex-1">
              <h4 className="text-lg font-semibold text-card-foreground mb-3">What are Scoring Weights?</h4>
              <p className="text-sm text-muted-foreground mb-4">
                These sliders help you decide which factors are most important when recommending products for your vending machine.
              </p>
              <div className="space-y-3 text-sm text-card-foreground">
                <div>
                  <strong className="text-card-foreground">Uniqueness Score:</strong> How important it is to have different types of products (variety). Higher values mean you want more product variety.
                </div>
                <div>
                  <strong className="text-card-foreground">Cousin Score:</strong> How important it is to have products that customers often buy together. Higher values mean you want products that sell well together.
                </div>
                <div>
                  <strong className="text-card-foreground">Inventory Score:</strong> How important it is that products run out at similar times, making restocking easier. Higher values mean you want products that need restocking at the same time.
                </div>
                <div>
                  <strong className="text-card-foreground">Healthiness Score:</strong> How important it is to have healthy products in your vending machine. Higher values mean you want more nutritious and healthy options, while lower values mean you're okay with more traditional snacks and drinks.
                </div>
                <div>
                  <strong className="text-card-foreground">Location Score:</strong> How important it is to match products to your selected location type. Higher values mean you want products that are popular and suitable for your specific location (like gym, school, or office), while lower values mean location doesn't matter as much.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

