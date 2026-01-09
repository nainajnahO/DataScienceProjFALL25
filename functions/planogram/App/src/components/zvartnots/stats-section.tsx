import { useState, useEffect } from "react"
import LocationSelector from "./location-selector.tsx"
import ScoringSliders, { ScoringSlidersQuestionMark } from "./scoring-sliders.tsx"
import StatCard from "./stat-card.tsx"
import TopSellingCard from "./top-selling-card.tsx"
import CollapsibleCard from "./collapsible-card.tsx"
import ScoringHelpModal from "./scoring-help-modal.tsx"
import LocationHelpModal from "./location-help-modal.tsx"
import LocationQuestionMark from "./location-question-mark.tsx"

interface StatsSectionProps {
  selectedLocation: string | null
  onLocationChange: (location: string | null) => void
  products: any[]
  placedProducts?: any[]
  address: string
  onAddressChange: (address: string) => void
  onAutofill: (weights: { static_weights: any, dynamic_weights: any }) => void
  onOptimize: (weights: { static_weights: any, dynamic_weights: any }) => void
}

export default function StatsSection({
  selectedLocation,
  onLocationChange,
  products,
  placedProducts,
  address,
  onAddressChange,
  onAutofill,
  onOptimize
}: StatsSectionProps) {
  const [uniquenessScore, setUniquenessScore] = useState(1.0)
  const [cousinScore, setCousinScore] = useState(0.0)
  const [inventoryScore, setInventoryScore] = useState(0.8)
  const [healthinessScore, setHealthinessScore] = useState(0.6)
  const [locationScore, setLocationScore] = useState(1.0)
  const [confidenceScore, setConfidenceScore] = useState(0.8)
  const [isLocationCollapsed, setIsLocationCollapsed] = useState(false)
  const [isScoringCollapsed, setIsScoringCollapsed] = useState(true)
  const [isTopSellingCollapsed, setIsTopSellingCollapsed] = useState(true)
  const [showScoringHelp, setShowScoringHelp] = useState(false)
  const [showLocationHelp, setShowLocationHelp] = useState(false)

  // Forecast State
  const [estimatedRecall, setEstimatedRecall] = useState<number | null>(null)
  // address prop used instead
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [predictionError, setPredictionError] = useState<string | null>(null)

  const locationTypes = [
    "GYM",
    "LEISURE ENTERT. VENUES",
    "MALL",
    "PETROL STATION",
    "SCHOOLS, UNIV",
    "SPORTS GROUNDS",
    "WAITING ROOM",
    "WORK",
    "Övrigt",
  ]

  const topSellingProducts = [
    { name: "Coca Cola", sales: 287 },
    { name: "Snickers", sales: 213 },
    { name: "Water", sales: 189 },
  ]

  const worstSellingProducts = [
    { name: "Pringles", sales: 23 },
    { name: "KitKat", sales: 18 },
    { name: "Twix", sales: 12 },
  ]

  // Automatic Prediction Effect
  useEffect(() => {
    if (!address || !selectedLocation) {
      setEstimatedRecall(null)
      setPredictionError(null)
      return
    }

    const timer = setTimeout(() => {
      predictRevenue()
    }, 1000) // 1 second debounce

    return () => clearTimeout(timer)
  }, [address, selectedLocation, placedProducts])

  const predictRevenue = async () => {
    if (!address || !selectedLocation) return
    const activeProducts = placedProducts && placedProducts.length > 0 ? placedProducts : products

    // If activeProducts is empty, clear prediction (unless we want to predict 0)
    // Actually, backend handles empty products returning 0 or similar.
    // User requested automatic updates as soon as product is placed.
    // If placedProducts is passed, it reflects planogram state.

    setPredictionLoading(true)
    setPredictionError(null)

    try {
      const productNames = activeProducts.map(p => p.name || p.product_name)

      // @ts-ignore
      const result = await window.ipcRenderer.invoke('predict-location', {
        products: productNames,
        address,
        category: selectedLocation
      })

      if (result.error) {
        throw new Error(result.error)
      }

      const details = result.details || []
      let totalRevenue = 0

      details.forEach((item: any) => {
        const productData = activeProducts.find(p => (p.name || p.product_name) === item.product)
        const salesCount = item.rule_based_prediction || 0
        const price = productData?.price || 0
        totalRevenue += salesCount * price
      })

      setEstimatedRecall(Math.round(totalRevenue))
    } catch (err: any) {
      console.error(err)
      setPredictionError(err.message || "Prediction failed")
    } finally {
      setPredictionLoading(false)
    }
  }


  const handleAutofillClick = () => {
    const weights = {
      static_weights: {
        healthiness: healthinessScore,
        location: locationScore,
        confidence: confidenceScore
      },
      dynamic_weights: {
        uniqueness: uniquenessScore,
        cousin: cousinScore,
        inventory: inventoryScore
      }
    }
    onAutofill(weights)
  }

  const handleOptimizeClick = () => {
    const weights = {
      static_weights: {
        healthiness: healthinessScore,
        location: locationScore,
        confidence: confidenceScore
      },
      dynamic_weights: {
        uniqueness: uniquenessScore,
        cousin: cousinScore,
        inventory: inventoryScore
      }
    }
    onOptimize(weights)
  }

  return (
    <div className="flex-1 w-full">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-1 gap-4">
        <CollapsibleCard
          title="Location"
          isCollapsed={isLocationCollapsed}
          onToggle={() => setIsLocationCollapsed(!isLocationCollapsed)}
          rightContent={
            <LocationQuestionMark onClick={() => setShowLocationHelp(!showLocationHelp)} />
          }
        >
          <LocationSelector
            locations={locationTypes}
            selectedLocation={selectedLocation}
            onLocationSelect={onLocationChange}
            address={address}
            onAddressChange={onAddressChange}
          />
        </CollapsibleCard>

        <CollapsibleCard
          title="Scoring Weights"
          isCollapsed={isScoringCollapsed}
          onToggle={() => setIsScoringCollapsed(!isScoringCollapsed)}
          rightContent={
            <ScoringSlidersQuestionMark onClick={() => setShowScoringHelp(!showScoringHelp)} />
          }
        >
          <ScoringSliders
            uniquenessScore={uniquenessScore}
            cousinScore={cousinScore}
            inventoryScore={inventoryScore}
            healthinessScore={healthinessScore}
            locationScore={locationScore}
            onUniquenessScoreChange={setUniquenessScore}
            onCousinScoreChange={setCousinScore}
            onInventoryScoreChange={setInventoryScore}
            onHealthinessScoreChange={setHealthinessScore}
            onLocationScoreChange={setLocationScore}
            confidenceScore={confidenceScore}
            onConfidenceScoreChange={setConfidenceScore}
            onAutofill={handleAutofillClick}
            onOptimize={handleOptimizeClick}
            disabled={!selectedLocation || !address}
            onQuestionMarkClick={() => setShowScoringHelp(!showScoringHelp)}
          />
        </CollapsibleCard>

        <CollapsibleCard
          title="Estimated Top Selling"
          isCollapsed={isTopSellingCollapsed}
          onToggle={() => setIsTopSellingCollapsed(!isTopSellingCollapsed)}
        >
          <TopSellingCard topProducts={topSellingProducts} worstProducts={worstSellingProducts} />
        </CollapsibleCard>

        <div className="grid grid-cols-1 gap-4">
          <StatCard
            title="Estimated income next 4 weeks"
            value={
              predictionLoading
                ? "Calculating..."
                : (predictionError ? "Error" : (estimatedRecall !== null ? `${estimatedRecall.toLocaleString()} kr` : "--- kr"))
            }
            change={
              predictionError
                ? predictionError
                : (predictionLoading ? "Updating forecast" : (estimatedRecall !== null ? "Based on location" : "Enter location to start"))
            }
            isPositive={!predictionError}
          />
        </div>
      </div>
      {showScoringHelp && (
        <ScoringHelpModal onClose={() => setShowScoringHelp(false)} />
      )}
      {showLocationHelp && (
        <LocationHelpModal onClose={() => setShowLocationHelp(false)} />
      )}
    </div>
  )
}

