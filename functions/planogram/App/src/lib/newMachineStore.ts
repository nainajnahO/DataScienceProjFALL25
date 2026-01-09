import type { AppMachine } from "./types.ts"

type Listener = () => void

/**
 * Global store for the "new" machine template
 * This allows layout editor changes to persist across navigations
 */
class NewMachineStore {
  private machine: AppMachine | null = null
  private listeners: Set<Listener> = new Set()

  setMachine(machine: AppMachine) {
    this.machine = machine
    this.notifyListeners()
  }

  getMachine(): AppMachine | null {
    return this.machine
  }

  clearMachine() {
    this.machine = null
    this.notifyListeners()
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  private notifyListeners() {
    this.listeners.forEach(listener => listener())
  }
}

// Export a singleton instance
export const newMachineStore = new NewMachineStore()