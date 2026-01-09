import { Link, useParams, Navigate } from "react-router-dom"
import { useEffect, useState } from "react"
import { getMachineById } from "@/lib/services/machines.ts"
import type { AppMachine } from "@/lib/types.ts"
import Zvartnots from "@/components/zvartnots/Zvartnots.tsx"

export default function PlanogramPage() {
  const { id } = useParams<{ id: string }>()
  const [machine, setMachine] = useState<AppMachine | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
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
  }, [id])

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
          <Link to="/" className="text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300 mt-4 inline-block">
            ← Back to Machines
          </Link>
        </div>
      </div>
    )
  }

  if (!machine) {
    return <Navigate to={`/machines/${id}/not-found`} replace />
  }

  return (
    <div>
      <div className="bg-card shadow-sm border-b border-border px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <Link to="/" className="text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300 text-sm mb-1 inline-block">
              ← Back to Machines
            </Link>
            <h1 className="text-2xl font-bold text-foreground">{machine.machine_name}</h1>
            <p className="text-sm text-muted-foreground">
              {machine.machine_model} • {machine.machine_key}
            </p>
          </div>
        </div>
      </div>
      <Zvartnots key={machine.id} machine={machine} />
    </div>
  )
}
