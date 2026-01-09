import { useEffect } from 'react'
import { Outlet } from 'react-router-dom'
import { useDarkMode } from './hooks/useDarkMode.ts'
import { productStore } from '@/lib/productStore'
import { healthinessGradeStore } from '@/lib/healthinessGradeStore'

export default function App() {
  const { isDark, toggle } = useDarkMode()

  // Initialize product store and healthiness grade store on app mount
  useEffect(() => {
    productStore.initialize().catch(error => {
      console.error("Failed to initialize product store:", error)
      // Could show a toast notification here
    })
    
    healthinessGradeStore.initialize().catch(error => {
      console.error("Failed to initialize healthiness grade store:", error)
      // Could show a toast notification here
    })
  }, [])

  return (
    <div className="font-sans antialiased bg-background text-foreground min-h-screen">
      {/* Global Dark Mode Toggle */}
      <button
        onClick={toggle}
        className="fixed top-4 right-4 z-50 p-3 rounded-full bg-card border border-border shadow-lg hover:shadow-xl transition-all"
        aria-label="Toggle dark mode"
      >
        {isDark ? (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
          </svg>
        ) : (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
          </svg>
        )}
      </button>
      <Outlet />
    </div>
  )
}
