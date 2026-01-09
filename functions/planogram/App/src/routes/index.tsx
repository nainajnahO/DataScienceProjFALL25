import { Link } from "react-router-dom"
import { useEffect, useState } from "react"
import { getAllMachines } from "@/lib/services/machines.ts"
import type { AppMachine } from "@/lib/types.ts"

export default function MachinesPage() {
  const [machines, setMachines] = useState<AppMachine[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchMachines() {
      try {
        setLoading(true)
        const data = await getAllMachines()
        setMachines(data)
      } catch (err) {
        console.error("Error fetching machines:", err)
        setError(err instanceof Error ? err.message : "Failed to fetch machines")
      } finally {
        setLoading(false)
      }
    }

    fetchMachines()
  }, [])

  return (
    <div className="min-h-screen bg-background">
      <div className="bg-card shadow-sm border-b border-border px-6 py-8">
        <div className="max-w-7xl mx-auto flex items-start justify-between">
          <div>
            <h1 className="text-4xl font-bold text-foreground mb-2">Vending Machines</h1>
            <p className="text-muted-foreground">Select a machine to view its planogram</p>
          </div>
          <Link
            to="/machines/new"
            className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white font-medium rounded-lg shadow-sm hover:shadow-md transition-all"
          >
            + Create New Machine
          </Link>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 dark:border-gray-100"></div>
              <p className="mt-4 text-muted-foreground">Loading machines...</p>
            </div>
          </div>
        ) : error ? (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">⚠️</div>
            <p className="text-xl text-red-500 dark:text-red-400">{error}</p>
          </div>
        ) : machines.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">📦</div>
            <p className="text-xl text-muted-foreground">No machines found</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {machines.map((machine) => (
              <Link
                key={machine.id}
                to={`/machines/${machine.id}`}
                className="bg-card rounded-lg shadow-md hover:shadow-lg transition-shadow p-6 border border-border hover:border-blue-300 dark:hover:border-blue-600"
              >
                <h3 className="text-xl font-bold text-card-foreground mb-2">{machine.machine_name}</h3>
                <div className="space-y-1 text-sm text-muted-foreground">
                  <p>
                    <span className="font-medium">Model:</span> {machine.machine_model}
                  </p>
                  <p>
                    <span className="font-medium">Key:</span> {machine.machine_key}
                  </p>
                  <p>
                    <span className="font-medium">Slots:</span> {machine.slots.length}
                  </p>
                  {machine.location && (
                    <p>
                      <span className="font-medium">Location:</span> {machine.location.address}
                    </p>
                  )}
                </div>
                <div className="mt-4 inline-flex items-center text-blue-500 dark:text-blue-400 font-medium">View Planogram →</div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
