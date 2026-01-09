interface LocationSelectorProps {
  locations: string[]
  selectedLocation: string | null
  onLocationSelect: (location: string) => void
  address: string
  onAddressChange: (address: string) => void
}

export default function LocationSelector({
  locations,
  selectedLocation,
  onLocationSelect,
  address,
  onAddressChange
}: LocationSelectorProps) {
  return (
    <div className="w-full">
      <div className="flex items-center gap-2 mb-3">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z" clipRule="evenodd" />
        </svg>
        <h3 className="text-muted-foreground text-sm font-medium">Location Category</h3>
      </div>
      <div className="grid grid-cols-5 gap-2 mb-4">
        {locations.map((location) => {
          const isSelected = selectedLocation === location
          return (
            <button
              key={location}
              onClick={() => onLocationSelect(location)}
              className={`
                p-2 rounded-md border-2 transition-all duration-200 text-center
                ${isSelected
                  ? 'bg-blue-500/20 border-blue-500 text-blue-700 dark:text-blue-300 shadow-md'
                  : 'bg-background/50 border-border hover:border-blue-300 dark:hover:border-blue-600 text-card-foreground hover:shadow-sm'
                }
              `}
            >
              <div className="font-medium text-xs leading-tight">{location}</div>
            </button>
          )
        })}
      </div>

      <div className="space-y-1">
        <label className="text-xs text-muted-foreground font-medium ml-0.5">Address</label>
        <input
          type="text"
          value={address}
          onChange={(e) => onAddressChange(e.target.value)}
          placeholder="e.g. Stockholm Central"
          className="w-full px-3 py-2 bg-background border border-input rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all placeholder:text-muted-foreground/50"
        />
      </div>
    </div>
  )
}

