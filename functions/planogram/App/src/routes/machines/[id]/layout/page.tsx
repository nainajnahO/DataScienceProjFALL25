import { useParams, useLocation, Navigate } from "react-router-dom"
import { useEffect, useState } from "react"
import { getMachineById } from "@/lib/services/machines.ts"
import type { AppMachine } from "@/lib/types.ts"
import LayoutEditorClient from "./layout-editor-client.tsx"
import { newMachineStore } from "@/lib/newMachineStore.ts"

export default function LayoutEditorPage() {
  const { id } = useParams<{ id: string }>()
  const location = useLocation()
  const [machine, setMachine] = useState<AppMachine | null>(location.state?.machine || null)
  const [loading, setLoading] = useState(!location.state?.machine)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // If machine data was passed via navigation state, use it immediately
    if (location.state?.machine) {
      setMachine(location.state.machine)
      setLoading(false)
      return
    }

    // Checking global store for 'new' machine as fallback
    if (id === "new") {
      const stored = newMachineStore.getMachine()
      if (stored) {
        setMachine(stored)
        setLoading(false)
        return
      }
    }

    // Otherwise, fetch from Firestore (fallback for direct URL access)
    async function fetchMachine() {
      if (!id) return

      try {
        setLoading(true)
        const data = await getMachineById(id)
        setMachine(data)
      } catch (err) {
        console.error("Error fetching machine:", err)
        setError(err instanceof Error ? err.message : "Failed to fetch machine")
      } finally {
        setLoading(false)
      }
    }

    fetchMachine()
  }, [id, location.state])

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 dark:border-gray-100"></div>
          <p className="mt-2 text-muted-foreground">Loading machine...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-red-500 dark:text-red-400">{error}</p>
        </div>
      </div>
    )
  }

  if (!machine) {
    return <Navigate to={`/machines/${id}/not-found`} replace />
  }

  return <LayoutEditorClient machine={machine} />
}
