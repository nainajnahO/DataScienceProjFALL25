interface StatCardProps {
  title: string
  value: string | number
  change: string
  isPositive: boolean
}

export default function StatCard({ title, value, change, isPositive }: StatCardProps) {
  return (
    <div className="w-full h-full bg-card dark:bg-card border border-border p-6 rounded-lg shadow-sm flex flex-col items-center justify-center text-center">
      <p className="text-sm font-medium text-muted-foreground mb-2">{title}</p>
      <p className="text-4xl font-bold text-card-foreground mb-2">{value}</p>
      <p className={`text-sm ${isPositive ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}`}>{change}</p>
    </div>
  )
}

