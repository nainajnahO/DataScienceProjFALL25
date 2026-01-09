"""Model saving utilities for production-ready artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import joblib
import json
from typing import Tuple
from src.models.multi_week_forecaster import MultiWeekForecaster


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_model_bundle(
    forecaster,
    strategy: str,
    feature_groups: Optional[list] = None,
    base_dir: Path | str = Path("../models/custom"),
    extra_metadata: Optional[Dict] = None,
) -> Path:
    """
    Save trained forecaster models and metadata into a timestamped directory.

    Structure:
      {base_dir}/{timestamp}_{strategy}/
        - model_h{horizon}.pkl (for each horizon)
        - metadata.json
    """
    base_dir = Path(base_dir)
    run_dir = base_dir / f"{_timestamp()}_{strategy}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save per-horizon estimators
    for horizon, model in getattr(forecaster, 'models', {}).items():
        joblib.dump(model, run_dir / f"model_h{horizon}.pkl")

    # Save metadata
    metadata = {
        "strategy": strategy,
        "feature_groups": feature_groups or [],
        "feature_names": getattr(forecaster, 'feature_names', []),
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "horizons": sorted(list(getattr(forecaster, 'models', {}).keys())),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return run_dir


def load_model_bundle(run_dir: Path | str) -> Tuple[MultiWeekForecaster, dict]:
    """
    Load a saved forecaster bundle from a run directory produced by save_model_bundle.
    Returns (forecaster, metadata).
    """
    run_dir = Path(run_dir)
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {run_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    strategy = metadata.get("strategy", "recursive_multi")
    feature_names = metadata.get("feature_names", [])
    horizons = metadata.get("horizons", [1, 2, 3, 4])

    forecaster = MultiWeekForecaster(strategy=strategy)
    forecaster.feature_names = feature_names
    forecaster.models = {}
    for h in horizons:
        model_path = run_dir / f"model_h{h}.pkl"
        if not model_path.exists():
            continue
        forecaster.models[h] = joblib.load(model_path)

    return forecaster, metadata


