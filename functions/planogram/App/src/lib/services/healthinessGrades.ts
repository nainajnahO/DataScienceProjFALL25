import { db } from "../firebase"
import { collection, getDocs } from "firebase/firestore"

/**
 * Firestore collection name for healthiness grade mappings
 */
const COLLECTION_NAME = "planogram_letter_grade_mapping"

/**
 * Interface for healthiness grade mapping document
 */
export interface HealthinessGradeMapping {
  ean: string
  healthiness_letter: string
}

/**
 * Fetches all healthiness grade mappings from Firestore
 * Creates a map of EAN -> healthiness_letter for quick lookups
 *
 * @returns Promise<Map<string, string>> - Map of EAN to healthiness letter grade
 * @throws Error if Firestore query fails
 */
export async function getAllHealthinessGrades(): Promise<Map<string, string>> {
  try {
    const colRef = collection(db, COLLECTION_NAME)
    const snapshot = await getDocs(colRef)

    if (snapshot.empty) {
      console.warn(`No healthiness grades found in Firestore collection "${COLLECTION_NAME}"`)
      return new Map()
    }

    const gradeMap = new Map<string, string>()
    
    snapshot.docs.forEach((doc) => {
      const data = doc.data()
      
      // Normalize EAN: convert to string and trim whitespace
      // Handle both number and string EANs
      let ean = data.ean
      if (ean == null || ean === '') {
        return
      }
      
      if (typeof ean === 'number') {
        ean = String(ean)
      } else {
        ean = String(ean).trim()
      }
      
      const healthinessLetter = String(data.healthiness_letter || "").trim().toUpperCase()
      
      if (!healthinessLetter || !['A', 'B', 'C', 'D', 'E'].includes(healthinessLetter)) {
        return
      }
      
      if (ean && healthinessLetter) {
        gradeMap.set(ean, healthinessLetter)
      }
    })

    console.log(`✅ Loaded ${gradeMap.size} healthiness grades from Firestore`)
    return gradeMap
  } catch (error) {
    console.error("❌ Error fetching healthiness grades from Firestore:", error)
    throw new Error(`Failed to fetch healthiness grades: ${error}`)
  }
}
