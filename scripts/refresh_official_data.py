"""Refresh official municipal connectivity data into the local SQLite snapshot."""

from __future__ import annotations

import json
import logging

from wifi_tunja_smart_predictor.connectivity.mintic import refresh_official_data


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        snapshot = refresh_official_data()
    except Exception as exc:
        logging.error(
            "Official refresh failed and no valid local cache is available: %s", type(exc).__name__
        )
        return 1
    batch = snapshot["batch"]
    print(
        json.dumps(
            {
                "source_id": batch["source_id"],
                "retrieved_at_utc": batch["retrieved_at_utc"],
                "source_updated_at_utc": batch["source_updated_at_utc"],
                "checked_at_utc": batch.get("checked_at_utc"),
                "unchanged_source": batch.get("unchanged_source", False),
                "retrieved_count": batch["retrieved_count"],
                "accepted_count": batch["accepted_count"],
                "rejected_count": batch["rejected_count"],
                "content_sha256": batch["content_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
