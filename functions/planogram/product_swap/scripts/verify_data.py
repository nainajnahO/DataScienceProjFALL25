"""Quick verification script to check the swap data."""
import pandas as pd

# Load the swap data
swaps = pd.read_parquet('data/swaps/product_swaps.parquet')

print("✓ Swap data loaded successfully!")
print(f"  Shape: {swaps.shape}")
print(f"  Total swaps: {len(swaps):,}")
print(f"  Date range: {swaps['swap_date'].min()} to {swaps['swap_date'].max()}")

print("\nColumns:")
for col in swaps.columns:
    print(f"  - {col}")

print("\n" + "="*70)
print("Example swap:")
print("="*70)
example = swaps.iloc[0]
for key, value in example.items():
    print(f"{key:25s}: {value}")

