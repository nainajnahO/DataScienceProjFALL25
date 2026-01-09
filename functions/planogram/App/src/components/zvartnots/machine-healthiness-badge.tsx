import { getGradeColor } from "./healthiness-utils"

interface MachineHealthinessBadgeProps {
  grade: string | null
}

export default function MachineHealthinessBadge({ grade }: MachineHealthinessBadgeProps) {
  return (
    <div className="flex-1 flex justify-start">
      {grade && (
        <span className={`px-5 py-3 text-3xl font-bold rounded-lg ${getGradeColor(grade)}`}>
          {grade}
        </span>
      )}
    </div>
  )
}

