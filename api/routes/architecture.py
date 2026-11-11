"""Read-only application architecture diagnostics."""

from __future__ import annotations

from fastapi import APIRouter

from wifi_tunja_smart_predictor.architecture import build_architecture_diagnostics

router = APIRouter(tags=["architecture"])


def _main_api_status() -> dict[str, str]:
    """Confirm the main process can respond to this route."""
    return {
        "status": "available",
        "check_status": "verified",
        "message": "This diagnostics response confirms the main API is responding.",
    }


@router.get("/architecture/diagnostics")
def architecture_diagnostics() -> dict:
    """Return component responsibilities and only statuses backed by available checks."""
    return build_architecture_diagnostics(main_api_status=_main_api_status())
