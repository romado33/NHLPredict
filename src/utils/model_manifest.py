"""Record which model artifacts a run used (paths + hashes) for reproducibility."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import MODELS_DIR, PROJECT_ROOT


def sha256_file(path: Path, chunk: int = 1 << 20) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_v2_model_manifest() -> dict[str, Any]:
    """Fingerprints for V2 series-win models consumed by scoring + V2 predict."""
    files = {
        "v2_models.joblib": MODELS_DIR / "v2_models.joblib",
        "v2_features.joblib": MODELS_DIR / "v2_features.joblib",
        "v2_blend_weights.joblib": MODELS_DIR / "v2_blend_weights.joblib",
    }
    out: dict[str, Any] = {
        "kind": "v2_team_series_models",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models_dir": str(MODELS_DIR.resolve()),
        "files": {},
    }
    for label, p in files.items():
        out["files"][label] = {
            "path": str(p.resolve()),
            "sha256": sha256_file(p),
            "exists": p.is_file(),
        }
    return out


def build_scoring_model_manifest() -> dict[str, Any]:
    """Fingerprints for playoff scoring models."""
    p = MODELS_DIR / "scoring_models.joblib"
    return {
        "kind": "playoff_scoring_models",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "scoring_models.joblib": {
                "path": str(p.resolve()),
                "sha256": sha256_file(p),
                "exists": p.is_file(),
            }
        },
    }


def write_run_manifest(path: Path | None = None, extra: dict[str, Any] | None = None) -> Path:
    """Write combined manifest to data/processed/model_run_manifest.json."""
    from config.settings import PROCESSED_DIR

    path = path or (PROCESSED_DIR / "model_run_manifest.json")
    payload: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v2": build_v2_model_manifest(),
        "scoring": build_scoring_model_manifest(),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
