import { useState, useEffect } from "react"
import ScoringSliderItem from "./scoring-slider-item.tsx"

interface ScoringSlidersProps {
  uniquenessScore: number
  cousinScore: number
  inventoryScore: number
  healthinessScore: number
  locationScore: number
  confidenceScore: number
  onUniquenessScoreChange: (value: number) => void
  onCousinScoreChange: (value: number) => void
  onInventoryScoreChange: (value: number) => void
  onHealthinessScoreChange: (value: number) => void
  onLocationScoreChange: (value: number) => void
  onConfidenceScoreChange: (value: number) => void
  onAutofill: () => void
  onOptimize: () => void
  disabled?: boolean
  onQuestionMarkClick?: () => void
}

export default function ScoringSliders({
  uniquenessScore,
  cousinScore,
  inventoryScore,
  healthinessScore,
  locationScore,
  confidenceScore,
  onUniquenessScoreChange,
  onCousinScoreChange,
  onInventoryScoreChange,
  onHealthinessScoreChange,
  onLocationScoreChange,
  onConfidenceScoreChange,
  onAutofill,
  onOptimize,
  disabled = false,
  onQuestionMarkClick: _onQuestionMarkClick,
}: ScoringSlidersProps) {
  const [localUniqueness, setLocalUniqueness] = useState(uniquenessScore.toFixed(1))
  const [localCousin, setLocalCousin] = useState(cousinScore.toFixed(1))
  const [localInventory, setLocalInventory] = useState(inventoryScore.toFixed(1))
  const [localHealthiness, setLocalHealthiness] = useState(healthinessScore.toFixed(1))
  const [localLocation, setLocalLocation] = useState(locationScore.toFixed(1))
  const [localConfidence, setLocalConfidence] = useState(confidenceScore.toFixed(1))

  useEffect(() => {
    setLocalUniqueness(uniquenessScore.toFixed(1))
  }, [uniquenessScore])

  useEffect(() => {
    setLocalCousin(cousinScore.toFixed(1))
  }, [cousinScore])

  useEffect(() => {
    setLocalInventory(inventoryScore.toFixed(1))
  }, [inventoryScore])

  useEffect(() => {
    setLocalHealthiness(healthinessScore.toFixed(1))
  }, [healthinessScore])

  useEffect(() => {
    setLocalLocation(locationScore.toFixed(1))
  }, [locationScore])

  useEffect(() => {
    setLocalConfidence(confidenceScore.toFixed(1))
  }, [confidenceScore])

  const handleInputBlur = (
    currentValue: string,
    actualValue: number,
    setLocal: (value: string) => void
  ) => {
    const numValue = parseFloat(currentValue)
    if (isNaN(numValue) || numValue < 0 || numValue > 1) {
      setLocal(actualValue.toFixed(1))
    } else {
      setLocal(numValue.toFixed(1))
    }
  }

  return (
    <div className="w-full relative">
      {disabled && (
        <div className="mb-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
          <div className="flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-600 dark:text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <p className="text-sm text-yellow-800 dark:text-yellow-200">Please select a location and enter address first used to adjust scoring weights</p>
          </div>
        </div>
      )}
      <div className="space-y-4">
        <ScoringSliderItem
          label="Uniqueness Score"
          value={uniquenessScore}
          localValue={localUniqueness}
          onChange={onUniquenessScoreChange}
          onLocalChange={setLocalUniqueness}
          onBlur={() => handleInputBlur(localUniqueness, uniquenessScore, setLocalUniqueness)}
          disabled={disabled}
        />
        <ScoringSliderItem
          label="Cousin Score"
          value={cousinScore}
          localValue={localCousin}
          onChange={onCousinScoreChange}
          onLocalChange={setLocalCousin}
          onBlur={() => handleInputBlur(localCousin, cousinScore, setLocalCousin)}
          disabled={disabled}
        />
        <ScoringSliderItem
          label="Inventory Score"
          value={inventoryScore}
          localValue={localInventory}
          onChange={onInventoryScoreChange}
          onLocalChange={setLocalInventory}
          onBlur={() => handleInputBlur(localInventory, inventoryScore, setLocalInventory)}
          disabled={disabled}
        />
        <ScoringSliderItem
          label="Healthiness Score"
          value={healthinessScore}
          localValue={localHealthiness}
          onChange={onHealthinessScoreChange}
          onLocalChange={setLocalHealthiness}
          onBlur={() => handleInputBlur(localHealthiness, healthinessScore, setLocalHealthiness)}
          disabled={disabled}
        />
        <ScoringSliderItem
          label="Location Score"
          value={locationScore}
          localValue={localLocation}
          onChange={onLocationScoreChange}
          onLocalChange={setLocalLocation}
          onBlur={() => handleInputBlur(localLocation, locationScore, setLocalLocation)}
          disabled={disabled}
        />
        <ScoringSliderItem
          label="Confidence Score"
          value={confidenceScore}
          localValue={localConfidence}
          onChange={onConfidenceScoreChange}
          onLocalChange={setLocalConfidence}
          onBlur={() => handleInputBlur(localConfidence, confidenceScore, setLocalConfidence)}
          disabled={disabled}
        />

        <div className="flex gap-2 pt-4">
          <button
            onClick={onAutofill}
            disabled={disabled}
            className={`flex-1 px-4 py-3 text-sm font-semibold rounded-lg shadow-md transition-all duration-300 flex items-center justify-center gap-2 transform ${!disabled
              ? 'bg-blue-600 hover:bg-blue-700 text-white hover:scale-105 hover:-translate-y-0.5'
              : 'bg-gray-300 dark:bg-gray-700 text-gray-500 cursor-not-allowed opacity-60'
              }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Autofill
          </button>
          <button
            onClick={onOptimize}
            disabled={disabled}
            className={`flex-1 px-4 py-3 text-sm font-semibold rounded-lg shadow-md transition-all duration-300 flex items-center justify-center gap-2 transform ${!disabled
              ? 'bg-purple-600 hover:bg-purple-700 text-white hover:scale-105 hover:-translate-y-0.5'
              : 'bg-gray-300 dark:bg-gray-700 text-gray-500 cursor-not-allowed opacity-60'
              }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Optimize
          </button>
        </div>
      </div>
    </div>
  )
}

export { ScoringSlidersQuestionMark } from "./scoring-sliders-question-mark.tsx"

