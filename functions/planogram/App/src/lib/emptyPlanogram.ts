import type { MachineSlot } from "./types.ts";

/**
 * Creates an empty planogram with the specified layout:
 * - Rows 1-3 (A, B, C): 2 slots of width 3, then 2 slots of width 1.5
 *   - Slot at col 0 (width 3) covers positions 0, 1, 2
 *   - Slot at col 3 (width 3) covers positions 3, 4, 5
 *   - Slot at col 6 (width 1.5) covers positions 6, 7 (visually A6-A7)
 *   - Slot at col 7 (width 1.5) covers positions 7, 8 (visually A7-A8)
 *   - Total: 3 + 3 + 1.5 + 1.5 = 9 positions (0-8)
 * - Row 4 (D): 9 slots of width 1 (positions 0-8)
 * - Rows 5-6 (E, F): 9 slots of width 1 (positions 0-8)
 */
export function createEmptyPlanogram(): MachineSlot[] {
  const slots: MachineSlot[] = [];

  // Rows 1-3 (A, B, C): 2 slots of width 3, then 2 slots of width 1.5
  for (let row = 0; row < 3; row++) {
    const rowLetter = String.fromCharCode(65 + row);

    // First two slots of width 3
    slots.push(createEmptySlot(`${rowLetter}0`, 3));
    slots.push(createEmptySlot(`${rowLetter}3`, 3));

    // Two slots of width 1.5 that overlap at position 7
    slots.push(createEmptySlot(`${rowLetter}6`, 1.5));
    slots.push(createEmptySlot(`${rowLetter}7`, 1.5));
  }

  // Row 4 (D): 9 slots of width 1 (positions D0-D8)
  // Total coverage: 9 positions, matching the rows above
  for (let col = 0; col < 9; col++) {
    slots.push(createEmptySlot(`D${col}`, 1));
  }

  // Rows 5-6 (E, F): 9 slots of width 1 (positions 0-8 to match grid)
  for (let row = 4; row < 6; row++) {
    const rowLetter = String.fromCharCode(65 + row);
    for (let col = 0; col < 9; col++) {
      slots.push(createEmptySlot(`${rowLetter}${col}`, 1));
    }
  }

  return slots;
}

function createEmptySlot(position: string, width: number): MachineSlot {
  return {
    position,
    product_name: "",
    image_url: "",
    price: 0,
    stock_current: 0,
    stock_max: 0,
    category: "",
    width,
    is_discount: false,
    discount: 0,
  };
}
