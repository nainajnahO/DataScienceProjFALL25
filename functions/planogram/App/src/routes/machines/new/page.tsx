// import { Link } from "react-router-dom"
import { useState, useEffect, useMemo } from "react"
import { createEmptyPlanogram } from "@/lib/emptyPlanogram.ts"
import type { AppMachine } from "@/lib/types.ts"
import Zvartnots from "@/components/zvartnots/Zvartnots.tsx"
import { newMachineStore } from "@/lib/newMachineStore.ts"

export default function NewMachineEditorPage() {
  // Get machine from global store or create empty template
  const [newMachine, setNewMachine] = useState<AppMachine>(() => {
    // Check if we have a saved machine in the global store
    const storedMachine = newMachineStore.getMachine()
    if (storedMachine) {
      return storedMachine
    }

    // Otherwise create a fresh empty machine
    return {
      id: "new",
      machine_name: "New Machine",
      machine_model: "Template",
      machine_key: "NEW-TEMPLATE",
      machine_sub_group: "Draft",
      last_sale: new Date().toISOString().split("T")[0],
      n_sales: 0,
      refillers: [],
      slots: createEmptyPlanogram(),
    }
  })

  // Subscribe to store changes
  useEffect(() => {
    const unsubscribe = newMachineStore.subscribe(() => {
      const storedMachine = newMachineStore.getMachine()
      if (storedMachine) {
        setNewMachine(storedMachine)
      }
    })
    return unsubscribe
  }, [])

  // Create a unique key based on slots content to force remount when layout changes
  const machineKey = useMemo(() => {
    // Create hash including position, width, and product name
    const slotsHash = newMachine.slots
      .map(s => `${s.position}-${s.width}-${s.product_name || 'empty'}`)
      .join(',')
    return `${newMachine.id}-${slotsHash}`
  }, [newMachine])  // Depend on entire newMachine object

  return (
    <div>
      <div className="bg-card shadow-sm border-b border-border px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            {/* <Link
              to="/"
              className="text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300 text-sm mb-1 inline-block"
            >
              ← Back to Machines
            </Link> */}
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-foreground">{newMachine.machine_name}</h1>
              <span className="px-2 py-1 text-xs font-medium rounded-md bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                Preview Only - Not Saved
              </span>
            </div>
            <p className="text-sm text-muted-foreground">
              {newMachine.machine_model} • {newMachine.machine_key}
            </p>
          </div>
        </div>
      </div>
      <Zvartnots key={machineKey} machine={newMachine} />
    </div>
  )
}
