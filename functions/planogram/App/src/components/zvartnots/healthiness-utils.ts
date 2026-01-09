import type { Product } from "@/lib/types"
import { healthinessGradeStore } from "@/lib/healthinessGradeStore"

/**
 * Gets the healthiness grade (A-E) for a product by looking up its EAN in the store
 * Returns null if the product doesn't have an EAN or if the grade isn't found
 */
export function getHealthinessGrade(product: Product): string | null {
  return healthinessGradeStore.getGradeForProduct(product)
}

/**
 * Converts a numeric healthiness score (0-4) to a letter grade (A-E)
 */
export function getGradeFromScore(score: number): string {
  const grades = ['A', 'B', 'C', 'D', 'E']
  const index = Math.round(score)
  return grades[Math.max(0, Math.min(4, index))]
}

/**
 * Returns the color classes for a healthiness grade badge
 */
export function getGradeColor(grade: string | null): string {
  switch (grade) {
    case 'A':
      return 'bg-green-500 text-white'
    case 'B':
      return 'bg-green-400 text-white'
    case 'C':
      return 'bg-yellow-500 text-white'
    case 'D':
      return 'bg-orange-500 text-white'
    case 'E':
      return 'bg-red-500 text-white'
    default:
      return 'bg-gray-500 text-white'
  }
}

