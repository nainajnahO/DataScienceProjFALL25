interface ScoringSliderItemProps {
  label: string
  value: number
  localValue: string
  onChange: (value: number) => void
  onLocalChange: (value: string) => void
  onBlur: () => void
  disabled?: boolean
}

export default function ScoringSliderItem({
  label,
  value,
  localValue,
  onChange,
  onLocalChange,
  onBlur,
  disabled = false,
}: ScoringSliderItemProps) {
  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const clampedValue = Math.max(0, Math.min(1, parseFloat(e.target.value)))
    onChange(clampedValue)
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onLocalChange(e.target.value)
    const numValue = parseFloat(e.target.value)
    if (!isNaN(numValue)) {
      const clampedValue = Math.max(0, Math.min(1, numValue))
      onChange(clampedValue)
    }
  }

  return (
    <div>
      <div className="flex items-center gap-3">
        <label className="text-sm font-medium text-card-foreground w-32">{label}</label>
        <input
          type="number"
          min="0"
          max="1"
          step="0.1"
          value={localValue}
          onChange={handleInputChange}
          onBlur={onBlur}
          disabled={disabled}
          className="w-16 px-2 py-1 text-sm border border-border rounded bg-background text-card-foreground text-center focus:outline-none focus:ring-2 focus:ring-blue-500 [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none [-moz-appearance:textfield] disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <div className="flex-1 relative">
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={value}
            onChange={handleSliderChange}
            disabled={disabled}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none accent-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <div className="flex justify-between mt-1 px-1">
            <span className="text-xs text-muted-foreground">0</span>
            <span className="text-xs text-muted-foreground">0.5</span>
            <span className="text-xs text-muted-foreground">1</span>
          </div>
        </div>
      </div>
    </div>
  )
}

