interface CollapsibleCardProps {
  title: string
  isCollapsed: boolean
  onToggle: () => void
  children: React.ReactNode
  rightContent?: React.ReactNode
}

export default function CollapsibleCard({ title, isCollapsed, onToggle, children, rightContent }: CollapsibleCardProps) {
  return (
    <div className="bg-card dark:bg-card border border-border rounded-lg shadow-md overflow-hidden">
      <div 
        className="flex items-center justify-between p-4 cursor-pointer hover:bg-background/50 transition-colors"
        onClick={onToggle}
      >
        <h3 className="text-muted-foreground text-sm font-medium">{title}</h3>
        <div className="flex items-center gap-2">
          {rightContent}
          <button
            onClick={(e) => {
              e.stopPropagation()
              onToggle()
            }}
            className="text-muted-foreground hover:text-card-foreground transition-all"
            aria-label={isCollapsed ? "Expand" : "Collapse"}
          >
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              className={`h-5 w-5 transition-transform duration-200 ${isCollapsed ? '' : 'rotate-180'}`}
              viewBox="0 0 20 20" 
              fill="currentColor"
            >
              <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
      </div>
      {!isCollapsed && (
        <div className="p-6 pt-0">
          {children}
        </div>
      )}
    </div>
  )
}

