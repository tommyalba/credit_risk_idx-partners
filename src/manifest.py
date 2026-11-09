"""Reproducibility manifest: environment, split configuration and artifact hashes.

Generates `artifacts/manifest.json` describing exactly which code, environment
and data produced the serialized artifacts in data/interim, data/processed and
artifacts/. Run standalone (`python -m src.manifest`) or from notebook 2 after
the split/serialization cells.
"""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

TRACKED_SUFFIXES = {".pkl", ".joblib", ".parquet", ".csv"}
TRACKED_DIRS = ["data/interim", "data/processed", "artifacts"]

# Mirror of the constants used in notebooks/2 (keep in sync when they change).
SPLIT_CONFIG = {
    "random_state": 42,
    "performance_window_M_months": 33,
    "performance_window_coverage": 0.90,
    "snapshot_date": "2016-01-01",
    "oot_cutoff": "2013-01-01",
    "validation_scheme": "temporal: latest ~11% of historical issues -> valid",
    "tree_based_inner_split": "15% of train -> 7.5% early-stop + 7.5% calibration",
    "target": "0=Fully Paid variants, 1=Charged Off/Default/Late(31-120)",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pkg_version(name: str):
    try:
        return __import__(name).__version__
    except Exception:
        return None


def _git_commit(root: Path):
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def build_manifest(root=".") -> dict:
    root = Path(root)
    artifacts = {}
    for sub in TRACKED_DIRS:
        base = root / sub
        if not base.exists():
            continue
        for f in sorted(base.rglob("*")):
            if f.is_file() and f.suffix in TRACKED_SUFFIXES:
                artifacts[str(f.relative_to(root))] = {
                    "sha256": _sha256(f),
                    "bytes": f.stat().st_size,
                }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(root),
        "python": platform.python_version(),
        "packages": {
            "pandas": _pkg_version("pandas"),
            "numpy": _pkg_version("numpy"),
            "scikit-learn": _pkg_version("sklearn"),
            "xgboost": _pkg_version("xgboost"),
            "catboost": _pkg_version("catboost"),
            "lightgbm": _pkg_version("lightgbm"),
            "optbinning": _pkg_version("optbinning"),
            "mlflow": _pkg_version("mlflow"),
            "dvc": _pkg_version("dvc"),
        },
        "split_config": SPLIT_CONFIG,
        "artifacts": artifacts,
    }


def write_manifest(path="artifacts/manifest.json", root=".") -> Path:
    out = Path(root) / path
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(root)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written: {out} ({len(manifest['artifacts'])} artifacts hashed)")
    return out


if __name__ == "__main__":
    write_manifest()
