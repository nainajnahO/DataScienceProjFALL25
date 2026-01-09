import { useState } from "react"
import LayoutEditorGrid from "./LayoutEditorGrid.tsx"
import LayoutConfirmationDialogs from "./layout-confirmation-dialogs.tsx"
import LayoutDisabledMessage from "./layout-disabled-message.tsx"
import { useMergeLogic } from "./use-merge-logic.ts"
import { useSlotAdjacency } from "./use-slot-adjacency.ts"
import { useDragHandlers } from "./use-drag-handlers.ts"
import { useSlotClickHandler } from "./use-slot-click-handler.ts"

export interface EditorProduct {
  id: string
  name: string
  price: number
  category: string
  image?: string
}

export interface MergedSlot {
  slots: number[]
  product?: EditorProduct
  width?: number
}

interface LayoutEditorProps {
  slots: Record<number, EditorProduct>
  mergedSlots: MergedSlot[]
  onUpdateSlots: (slots: Record<number, EditorProduct>, mergedSlots: MergedSlot[]) => void
  onBack: () => void
  rows: number
  cols: number
  machineName?: string
}

export default function LayoutEditor({
  slots,
  mergedSlots: initialMergedSlots,
  onUpdateSlots,
  onBack,
  rows: ROWS,
  cols: COLS,
  machineName,
}: LayoutEditorProps) {
  const [localSlots, setLocalSlots] = useState(slots)
  const [mergedSlots, setMergedSlots] = useState<MergedSlot[]>(initialMergedSlots)
  const [showConfirmation, setShowConfirmation] = useState(false)
  const [showDisabledMessage, setShowDisabledMessage] = useState(false)
  const [confirmationAction, setConfirmationAction] = useState<{
    type: "merge-double" | "merge-2-opt-3" | "create-triple-or-pair" | "split" | "split-triple" | "split-pair"
    data: any
  } | null>(null)

  const getPositionLabel = (index: number) => {
    const row = Math.floor(index / COLS)
    const col = index % COLS
    return `${String.fromCharCode(65 + row)}${col}`
  }

  const isMergedSlot = (index: number) => {
    const startMerge = mergedSlots.find((merged) => merged.slots[0] === index)
    if (startMerge) return startMerge
    return mergedSlots.find((merged) => merged.slots.includes(index))
  }

  const isFirstOfAnyMergedGroup = (index: number) => {
    return mergedSlots.some((merged) => merged.slots.includes(index) && merged.slots[0] === index)
  }

  const { areAdjacent, isAdjacentToMergedGroup, findExpansionNeighbor } = useSlotAdjacency(COLS)

  const findExpansionNeighborWithContext = (slot1: number, slot2: number) => {
    return findExpansionNeighbor(slot1, slot2, ROWS, localSlots, isMergedSlot)
    }

  const { createMerge, createPair1_5, createPair1_5_From3, splitMergedSlot } = useMergeLogic({
    mergedSlots,
    localSlots,
    setMergedSlots,
    setLocalSlots,
    isMergedSlot,
    ROWS,
    COLS,
  })

  const {
    draggedSlot,
    dragOverSlot,
    handleDragStart,
    handleDragOver,
    handleDragLeave,
    handleDrop,
  } = useDragHandlers({
    localSlots,
    mergedSlots,
    isMergedSlot,
    isAdjacentToMergedGroup,
    areAdjacent,
    findExpansionNeighbor: findExpansionNeighborWithContext,
    setShowDisabledMessage,
    setConfirmationAction,
    setShowConfirmation,
  })

  const { firstSelectedSlot, handleSlotClick } = useSlotClickHandler({
    mergedSlots,
    localSlots,
    isMergedSlot,
    areAdjacent,
    isAdjacentToMergedGroup,
    findExpansionNeighbor: findExpansionNeighborWithContext,
    setShowDisabledMessage,
    setConfirmationAction,
    setShowConfirmation,
  })

  const handleSave = () => {
    onUpdateSlots(localSlots, mergedSlots)
    onBack()
  }

  const handleAction = (actionChoice?: string) => {
    if (!confirmationAction) return

    if (confirmationAction.type === "merge-double") {
      const { draggedSlot, targetIndex } = confirmationAction.data
      createMerge(draggedSlot, targetIndex, undefined)
    } else if (confirmationAction.type === "merge-2-opt-3") {
      const { draggedSlot, targetIndex, extension } = confirmationAction.data
      if (actionChoice === "double") {
        createMerge(draggedSlot, targetIndex, undefined)
      } else if (actionChoice === "pair-1.5") {
        if (extension !== null) {
          createPair1_5_From3(draggedSlot, targetIndex, extension)
        }
      }
    } else if (confirmationAction.type === "create-triple-or-pair") {
      const { draggedSlot, targetIndex } = confirmationAction.data
      if (actionChoice === "triple") {
        createMerge(draggedSlot, targetIndex, undefined)
      } else if (actionChoice === "pair") {
        createPair1_5(draggedSlot, targetIndex)
      }
    } else if (confirmationAction.type === "split") {
      const { index } = confirmationAction.data
      splitMergedSlot(index)
    } else if (confirmationAction.type === "split-triple") {
      const { index } = confirmationAction.data
      const merged = isMergedSlot(index)
      if (!merged) return

      if (actionChoice === "singles") {
        splitMergedSlot(index)
      } else if (actionChoice === "1.5s") {
        const [s1, s2, s3] = merged.slots
        setMergedSlots(prev => prev.filter(m => !m.slots.includes(index)))
        const firstHalf: MergedSlot = { slots: [s1, s2], width: 1.5, product: merged.product }
        const secondHalf: MergedSlot = { slots: [s2, s3], width: 1.5, product: merged.product }
        setMergedSlots(prev => [...prev, firstHalf, secondHalf])
      }
    } else if (confirmationAction.type === "split-pair") {
      const { merged } = confirmationAction.data
      if (!merged) return
      const partner = mergedSlots.find(m => m !== merged && m.width === 1.5 && m.slots.some(s => merged.slots.includes(s)))
      setMergedSlots(prev => prev.filter(m => m !== merged && m !== partner))
    }

    setShowConfirmation(false)
    setConfirmationAction(null)
  }

  const handleCancel = () => {
    setShowConfirmation(false)
    setConfirmationAction(null)
  }

  return (
    <div className="min-h-screen bg-background p-4 md:p-6">
      <LayoutDisabledMessage
        show={showDisabledMessage}
        onClose={() => setShowDisabledMessage(false)}
      />

      <LayoutConfirmationDialogs
        showConfirmation={showConfirmation}
        confirmationAction={confirmationAction}
        onAction={handleAction}
        onCancel={handleCancel}
      />

      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-xl md:text-2xl font-semibold text-foreground mb-4">
            Layout for:
            <br />
            <span className="font-bold">{machineName || "Vending Machine"}</span>
          </h1>
          {firstSelectedSlot !== null && (
            <p className="text-sm text-blue-600 dark:text-blue-400 font-medium">
              Selected: {getPositionLabel(firstSelectedSlot)} - Click another slot to swap
            </p>
          )}
          <p className="text-sm text-muted-foreground mt-2">
            Drag a slot onto an adjacent slot to create double (2), triple (3), or 1.5-width spaces. Click on a merged slot to split it. Slots will be renumbered automatically.
          </p>
        </div>

        <LayoutEditorGrid
          rows={ROWS}
          cols={COLS}
          localSlots={localSlots}
          mergedSlots={mergedSlots}
          firstSelectedSlot={firstSelectedSlot}
          draggedSlot={draggedSlot}
          dragOverSlot={dragOverSlot}
          onSlotClick={handleSlotClick}
          onDragStart={handleDragStart}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          isMergedSlot={isMergedSlot}
          isFirstOfAnyMergedGroup={isFirstOfAnyMergedGroup}
          areAdjacent={areAdjacent}
        />

        <button
          onClick={handleSave}
          className="w-full bg-green-600 hover:bg-green-700 text-white py-6 text-lg rounded-md font-medium transition-colors"
        >
          Change Layout
        </button>
      </div>
    </div>
  )
}
