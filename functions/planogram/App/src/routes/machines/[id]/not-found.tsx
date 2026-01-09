import { Link } from "react-router-dom"

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center">
        <div className="text-6xl mb-4">⚠️</div>
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Machine Not Found</h1>
        <p className="text-gray-600 mb-6">Unable to load machine data</p>
        <Link
          to="/"
          className="inline-block px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
        >
          Back to Machine List
        </Link>
      </div>
    </div>
  )
}
