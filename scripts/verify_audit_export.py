#!/usr/bin/env python3
"""Verify a downloaded policy audit JSON/CSV export and manifest offline."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from wifi_tunja_smart_predictor.experiments.ml_policy_export_manifest import main as verify_main

    return verify_main()


if __name__ == "__main__":
    raise SystemExit(main())
