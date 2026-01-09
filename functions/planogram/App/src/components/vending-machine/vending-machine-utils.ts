// Utility functions for vending machine components

export function indexToPosition(index: number, cols: number): string {
  const row = Math.floor(index / cols);
  const col = index % cols;
  return String.fromCharCode(65 + row) + col;
}

export function getGridColumnSpan(width: number): number {
  if (width === 1.5) return 3;
  return width * 2;
}
