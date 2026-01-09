import { useMemo } from "react"
import type { Product } from "../../lib/types"
import { getHealthinessGrade } from "./healthiness-utils"

export function useProductFiltering(
  products: Product[],
  searchQuery: string,
  selectedHealthinessScores: string[]
) {
  const filteredProducts = useMemo(() => {
    return products
      .filter(
        (product) => {
          const matchesSearch = product.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            product.category.toLowerCase().includes(searchQuery.toLowerCase())
          const productGrade = getHealthinessGrade(product)
          // If filters are selected, only show products that have a grade matching the filter
          // If no filters, show all products (including those without grades)
          const matchesHealthiness = selectedHealthinessScores.length === 0 || (productGrade && selectedHealthinessScores.includes(productGrade))
          return matchesSearch && matchesHealthiness
        }
      )
      .sort((a, b) => a.name.localeCompare(b.name))
  }, [products, searchQuery, selectedHealthinessScores])

  return filteredProducts
}

