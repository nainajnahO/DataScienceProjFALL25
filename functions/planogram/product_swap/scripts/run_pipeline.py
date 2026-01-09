"""
Product Swap Pipeline - Run All Setup Scripts
==============================================

This script runs all necessary setup scripts in the correct order to:
1. Process sales data with profit/revenue
2. Detect product swaps
3. Enrich swaps with financial metrics
4. Train prediction models

Usage:
    python scripts/run_pipeline.py [--skip-existing] [--skip-training]
"""

import subprocess
import sys
from pathlib import Path
import argparse
from datetime import datetime


# Script directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def check_output_exists(step_name, file_paths):
    """Check if output files already exist."""
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]
    
    all_exist = all(Path(f).exists() for f in file_paths)
    if all_exist:
        print(f"  ✓ Output files already exist for {step_name}")
        for f in file_paths:
            file_path = Path(f)
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"    - {file_path.name} ({size_mb:.1f} MB)")
    return all_exist


def run_script(script_name, description, output_files=None, skip_if_exists=False):
    """
    Run a script and handle errors.
    
    Args:
        script_name: Name of the script to run (relative to scripts/)
        description: Human-readable description
        output_files: List of output files to check for existence
        skip_if_exists: If True, skip if output files already exist
    """
    script_path = SCRIPT_DIR / script_name
    
    if not script_path.exists():
        print(f"❌ ERROR: Script not found: {script_path}")
        return False
    
    print("\n" + "="*70)
    print(f"STEP: {description}")
    print(f"Running: {script_name}")
    print("="*70)
    
    # Check if output already exists
    if skip_if_exists and output_files:
        if check_output_exists(description, output_files):
            print(f"  ⏭ Skipping {script_name} (output files already exist)")
            return True
    
    # Run the script
    try:
        start_time = datetime.now()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False  # Show output in real-time
        )
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n✓ {description} completed successfully ({elapsed:.1f}s)")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: {description} failed with exit code {e.returncode}")
        print(f"   Script: {script_name}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  INTERRUPTED: {description} was cancelled by user")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: Unexpected error running {script_name}: {e}")
        return False


def main():
    """Run the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="Run all product swap setup scripts in order",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py                    # Run all steps
  python scripts/run_pipeline.py --skip-existing    # Skip steps if output exists
  python scripts/run_pipeline.py --skip-training    # Skip model training
        """
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip steps if output files already exist'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training step'
    )
    parser.add_argument(
        '--only-training',
        action='store_true',
        help='Only run model training (assumes data is already prepared)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PRODUCT SWAP PIPELINE - COMPLETE SETUP")
    print("="*70)
    print(f"\nRunning from: {PROJECT_ROOT}")
    print(f"Script directory: {SCRIPT_DIR}")
    
    if args.skip_existing:
        print("\n⚠️  Mode: Skip steps if output files already exist")
    if args.skip_training:
        print("\n⚠️  Mode: Skip model training")
    if args.only_training:
        print("\n⚠️  Mode: Only run model training")
    
    pipeline_start = datetime.now()
    steps_completed = []
    steps_failed = []
    
    # Step 1: Aggregate sales data with profit/revenue
    if not args.only_training:
        output_files = [
            PROJECT_ROOT / 'data' / 'sales' / 'Sales_2020_with_profit.parquet',
            PROJECT_ROOT / 'data' / 'aggregates' / 'product_profit_revenue.parquet'
        ]
        success = run_script(
            'aggregate_sales_with_profit.py',
            'Aggregate Sales Data with Profit/Revenue',
            output_files=output_files,
            skip_if_exists=args.skip_existing
        )
        if success:
            steps_completed.append('1. Sales aggregation')
        else:
            steps_failed.append('1. Sales aggregation')
            print("\n❌ Pipeline stopped due to error in sales aggregation")
            return False
    
    # Step 2: Detect swaps
    if not args.only_training:
        output_files = [
            PROJECT_ROOT / 'data' / 'swaps' / 'product_swaps.parquet'
        ]
        success = run_script(
            'detect_swaps.py',
            'Detect Product Swaps',
            output_files=output_files,
            skip_if_exists=args.skip_existing
        )
        if success:
            steps_completed.append('2. Swap detection')
        else:
            steps_failed.append('2. Swap detection')
            print("\n❌ Pipeline stopped due to error in swap detection")
            return False
    
    # Step 3: Enrich swaps with profit/revenue
    if not args.only_training:
        output_files = [
            PROJECT_ROOT / 'data' / 'swaps' / 'product_swaps_enriched.parquet'
        ]
        success = run_script(
            'enrich_swaps_with_profit_revenue.py',
            'Enrich Swaps with Profit/Revenue Metrics',
            output_files=output_files,
            skip_if_exists=args.skip_existing
        )
        if success:
            steps_completed.append('3. Swap enrichment')
        else:
            steps_failed.append('3. Swap enrichment')
            print("\n❌ Pipeline stopped due to error in swap enrichment")
            return False
    
    # Step 4: Train models
    if not args.skip_training:
        output_files = [
            PROJECT_ROOT / 'models' / 'revenue_model.pkl',
            PROJECT_ROOT / 'models' / 'profit_model.pkl',
            PROJECT_ROOT / 'models' / 'success_model.pkl',
            PROJECT_ROOT / 'models' / 'encoders.pkl'
        ]
        success = run_script(
            'train_swap_model.py',
            'Train Prediction Models',
            output_files=output_files,
            skip_if_exists=False  # Always train models (can be time-consuming but important)
        )
        if success:
            steps_completed.append('4. Model training')
        else:
            steps_failed.append('4. Model training')
            print("\n⚠️  Model training failed, but data preparation completed")
    
    # Summary
    total_time = (datetime.now() - pipeline_start).total_seconds()
    
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"\n✓ Steps completed ({len(steps_completed)}):")
    for step in steps_completed:
        print(f"   {step}")
    
    if steps_failed:
        print(f"\n❌ Steps failed ({len(steps_failed)}):")
        for step in steps_failed:
            print(f"   {step}")
    
    if not steps_failed:
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE - All steps finished successfully!")
        print("="*70)
        print("\nYou can now use the prediction models in the frontend:")
        print("  from runtime.predict_swap_outcome import SwapPredictor")
        print("  predictor = SwapPredictor()")
        print("="*70)
        return True
    else:
        print("\n" + "="*70)
        print("⚠️  PIPELINE COMPLETED WITH ERRORS")
        print("="*70)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

