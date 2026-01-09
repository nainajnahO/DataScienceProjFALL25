import { useEffect, useState } from "react"

interface ErrorToastProps {
  message: string
  isVisible: boolean
  onClose: () => void
}

export default function ErrorToast({ message, isVisible, onClose }: ErrorToastProps) {
  const [shouldRender, setShouldRender] = useState(false)

  useEffect(() => {
    if (isVisible) {
      setShouldRender(true)
      const timer = setTimeout(() => {
        setShouldRender(false)
        setTimeout(onClose, 300) // Wait for fade-out animation
      }, 3000)
      return () => clearTimeout(timer)
    }
  }, [isVisible, onClose])

  if (!shouldRender && !isVisible) return null

  return (
    <div
      className={`fixed top-4 left-1/2 -translate-x-1/2 z-[100] transition-all duration-300 ${
        isVisible && shouldRender ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-2'
      }`}
    >
      <div className="bg-red-500 text-white px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 min-w-[300px] max-w-[500px]">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-5 w-5 flex-shrink-0"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
            clipRule="evenodd"
          />
        </svg>
        <p className="text-sm font-medium flex-1">{message}</p>
      </div>
    </div>
  )
}

