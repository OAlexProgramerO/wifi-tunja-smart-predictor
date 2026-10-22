#!/usr/bin/env python3
"""Generate synthetic runtime resources once in the Compose-owned named volumes."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from wifi_tunja_smart_predictor.config import (
    METRICS_DIR,
    PROJECT_ROOT,
    experiment_database_path,
)
from wifi_tunja_smart_predictor.deployment import (
    REQUIRED_DEMO_ASSETS,
    RUNTIME_READY_MARKER,
    missing_demo_assets,
    model_artifacts_available,
)
from wifi_tunja_smart_predictor.experiments.store import ExperimentStore
from wifi_tunja_smart_predictor.models.artifacts import (
    model_artifacts_available as model_bundle_artifacts_available,
)

STAGES = (
    "scripts/generate_synthetic_dataset.py",
    "scripts/prepare_data.py",
    "scripts/train_model.py",
    "scripts/evaluate_model.py",
)


def initialize_runtime(
    *,
    assets: tuple[Path, ...] = REQUIRED_DEMO_ASSETS,
    run_stage: Callable[[Sequence[str]], int] | None = None,
    validate_model_bundle: Callable[[], bool] | None = None,
    validate_model_artifacts: Callable[[], bool] | None = None,
    root: Path = PROJECT_ROOT,
    metrics_dir: Path = METRICS_DIR,
) -> int:
    """Create resources only in an empty volume; leave partial/existing data untouched."""
    experiment_path = experiment_database_path(root)
    ExperimentStore(experiment_path)
    invoke = run_stage or (lambda command: _run_stage(command, root=root))
    validate_bundle = validate_model_bundle or model_bundle_artifacts_available
    validate_runtime = validate_model_artifacts or model_artifacts_available
    marker = next(
        (asset for asset in assets if asset.name == RUNTIME_READY_MARKER.name),
        RUNTIME_READY_MARKER,
    )
    required_outputs = tuple(asset for asset in assets if asset != marker)
    missing = missing_demo_assets(assets)
    if not missing:
        if not _validate_or_refresh_diagnostics(
            validate_bundle=validate_bundle,
            validate_runtime=validate_runtime,
            invoke=invoke,
        ):
            return 1
        print("Synthetic dataset and model artifacts are already initialized.", flush=True)
        return 0
    if missing == (marker,) and not missing_demo_assets(required_outputs):
        if not _validate_or_refresh_diagnostics(
            validate_bundle=validate_bundle,
            validate_runtime=validate_runtime,
            invoke=invoke,
        ):
            return 1
        marker.touch()
        print("Validated existing resources and restored the runtime-ready marker.", flush=True)
        return 0
    if len(missing) != len(assets):
        print(
            "Runtime resources are incomplete; refusing to overwrite existing volume data. "
            "Inspect `docker compose logs runtime-init`, then reset only the demo volumes with "
            "`docker compose down -v` if the data can be regenerated.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    metrics_dir.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        print(f"Initializing synthetic resources: {Path(stage).name}", flush=True)
        if invoke((sys.executable, stage)) != 0:
            print(
                f"Runtime initialization failed in {Path(stage).name}; "
                "inspect `docker compose logs runtime-init`.",
                file=sys.stderr,
                flush=True,
            )
            return 1

    missing = missing_demo_assets(required_outputs)
    if missing:
        print(
            "Initialization finished without producing every required runtime resource; "
            "inspect `docker compose logs runtime-init`.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if not _validate_or_refresh_diagnostics(
        validate_bundle=validate_bundle,
        validate_runtime=validate_runtime,
        invoke=invoke,
    ):
        return 1
    marker.touch()
    print("Synthetic runtime resources initialized successfully.", flush=True)
    return 0


def _validate_or_refresh_diagnostics(
    *,
    validate_bundle: Callable[[], bool],
    validate_runtime: Callable[[], bool],
    invoke: Callable[[Sequence[str]], int],
) -> bool:
    """Refresh versioned reports only when the existing model bundle is intact."""
    try:
        if not validate_bundle():
            raise ValueError("persisted model bundle validation failed")
        if validate_runtime():
            return True
        print(
            "Existing model artifacts are valid but runtime reports are stale; "
            "refreshing evaluation reports without retraining models.",
            flush=True,
        )
        if invoke((sys.executable, "scripts/evaluate_model.py")) != 0:
            raise ValueError("evaluation report refresh failed")
        if not validate_runtime():
            raise ValueError("refreshed model diagnostics failed validation")
        return True
    except (OSError, ValueError):
        print(
            "Existing runtime model artifacts or diagnostics failed validation; "
            "refusing to overwrite persistent resources. Inspect the runtime-init and API logs.",
            file=sys.stderr,
            flush=True,
        )
        return False


def _run_stage(command: Sequence[str], *, root: Path) -> int:
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(initialize_runtime())
