"""Registry for official Colombian connectivity data resources."""

from __future__ import annotations

from typing import Any

SOURCES: tuple[dict[str, Any], ...] = (
    {
        "source_id": "mintic_fixed_accesses",
        "institution": "Ministerio de Tecnologías de la Información y las Comunicaciones (MinTIC)",
        "dataset_name": "Accesos Fijos a Internet por Proveedor, Departamento, Municipio, Tecnología, Segmento y ancho de banda",
        "official_url": "https://www.datos.gov.co/Ciencia-Tecnolog-a-e-Innovaci-n/Accesos-Fijos-a-Internet-por-Proveedor-Departament/hwrp-3su3",
        "api_url": "https://www.datos.gov.co/resource/hwrp-3su3.json",
        "dataset_id": "hwrp-3su3",
        "geographic_coverage": "Colombia",
        "geographic_granularity": "Municipality, including Tunja (DIVIPOLA 15001); not neighborhood-level",
        "temporal_granularity": "Quarterly",
        "indicators": [
            "reported fixed internet access counts",
            "provider",
            "technology",
            "segment",
            "reported effective downstream and upstream bandwidth in Mbps",
        ],
        "license": "Creative Commons Attribution-ShareAlike 4.0 (catalog license CC_40_BY_SA)",
        "limitations": "Aggregated reported fixed-access records; not traffic, simultaneous use, coverage, latency, jitter, congestion, or saturation measurements.",
        "status": "registered",
        "last_source_update_utc": None,
        "last_successful_retrieval_utc": None,
    },
    {
        "source_id": "crc_postdata_qoe_fixed_reference",
        "institution": "Comisión de Regulación de Comunicaciones (CRC)",
        "dataset_name": "Data Flash 2025-016 QoE Fijo Jun.2025 (report reference)",
        "official_url": "https://www.postdata.gov.co/sites/default/files/dataflash/Data%20Flash%202025-016%20QoE%20Fijo%20Jun.2025.pdf",
        "api_url": None,
        "dataset_id": None,
        "geographic_coverage": "Capital-city comparison described by the report",
        "geographic_granularity": "City-level report reference; no granular machine-readable feed verified",
        "temporal_granularity": "June 2025 report period",
        "indicators": ["fixed internet quality-of-experience index described by report"],
        "license": None,
        "limitations": "Report/reference only; not treated as a regularly updated structured data feed. Direct PDF URL was unavailable during V7.0.0 source verification.",
        "status": "reference_unavailable",
        "last_source_update_utc": None,
        "last_successful_retrieval_utc": None,
    },
    {
        "source_id": "crc_postdata_quality_portal",
        "institution": "Comisión de Regulación de Comunicaciones (CRC)",
        "dataset_name": "Postdata quality dashboards: fixed and mobile user-experience measurements",
        "official_url": "https://www.postdata.gov.co/",
        "api_url": None,
        "dataset_id": None,
        "geographic_coverage": "National; fixed dashboard described by city and department; mobile dashboard by municipality and some localities/communes",
        "geographic_granularity": "Dashboard-level city/municipality and some large-city localities; data export/API not verified",
        "temporal_granularity": "Multiple chronological periods; exact availability varies by dashboard",
        "indicators": [
            "fixed and mobile user-experience quality indicators; exact fields are not ingested"
        ],
        "license": None,
        "limitations": "Official dashboard descriptions cite Ookla Speedtest crowdsourcing for fixed internet and operator-contracted crowdsourcing from user devices for mobile internet. This implementation does not ingest the underlying records or validate a stable data export/API.",
        "status": "reference_only",
        "last_source_update_utc": None,
        "last_successful_retrieval_utc": None,
    },
)


def source_registry(retrieval: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return a copy of the source registry with optional retrieval status fields."""
    retrieval = retrieval or {}
    sources = []
    for registered in SOURCES:
        source = dict(registered)
        if source["source_id"] in retrieval:
            source.update(retrieval[source["source_id"]])
        sources.append(source)
    return sources
