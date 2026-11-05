"""Deterministic assistant answers grounded in validated official snapshots."""

from __future__ import annotations

from typing import Any

from wifi_tunja_smart_predictor.connectivity.analysis import analyze_tunja
from wifi_tunja_smart_predictor.connectivity.mintic import load_snapshot

MINTIC_URL = "https://www.datos.gov.co/Ciencia-Tecnolog-a-e-Innovaci-n/Accesos-Fijos-a-Internet-por-Proveedor-Departament/hwrp-3su3"
POSTDATA_URL = "https://www.postdata.gov.co/"


def official_connectivity_tool(message: str) -> dict[str, Any]:
    """Answer explicitly real-connectivity questions without synthetic fallback."""
    spanish = any(
        term in message.casefold()
        for term in (
            "datos",
            "oficial",
            "conectividad",
            "proveedor",
            "tecnolog",
            "barrio",
            "satur",
            "calidad",
            "hay",
            "disponible",
            "último",
            "ultimo",
            "municipal",
        )
    )
    language = "es" if spanish else "en"
    normalized = message.casefold()
    try:
        snapshot = load_snapshot()
    except Exception:
        snapshot = None
    if snapshot is None:
        answer = (
            f"No hay una instantánea oficial válida guardada localmente, así que no puedo confirmar datos actuales de Tunja. La fuente registrada es el conjunto de accesos fijos MinTIC hwrp-3su3; actualízalo con `python scripts/refresh_official_data.py`. No usaré datos sintéticos como sustituto de evidencia real. Fuente: {MINTIC_URL}."
            if language == "es"
            else f"No validated official snapshot is stored locally, so I cannot confirm current data for Tunja. The registered source is MinTIC fixed-access dataset hwrp-3su3; refresh it with `python scripts/refresh_official_data.py`. I will not use synthetic data as a substitute for real evidence. Source: {MINTIC_URL}."
        )
        return {
            "answer": answer,
            "availability": "unavailable",
            "source_id": "mintic_fixed_accesses",
            "source_url": MINTIC_URL,
            "synthetic_fallback_used": False,
        }

    batch = snapshot["batch"]
    analysis = analyze_tunja(snapshot["records"])
    period = analysis["latest_reporting_period"] or "not recorded"
    freshness = (
        " La instantánea está obsoleta según el umbral operativo de 30 días."
        if snapshot["stale"] and language == "es"
        else (
            " This snapshot is stale under the 30-day operational warning threshold."
            if snapshot["stale"]
            else ""
        )
    )
    municipal_limit = (
        "La granularidad es municipal (Tunja, DIVIPOLA 15001); no hay observaciones por barrio."
        if language == "es"
        else "The geographic granularity is municipality-level (Tunja, DIVIPOLA 15001); no neighborhood observations are present."
    )
    if any(
        term in normalized
        for term in ("saturation", "congestion", "saturación", "saturacion", "congestión")
    ):
        answer = (
            f"No. La fuente MinTIC hwrp-3su3 informa accesos fijos y valores de velocidad por trimestre hasta {period}; no mide tráfico, utilización, disponibilidad ni saturación. Estos datos no confirman congestión real. {municipal_limit} Fuente: {MINTIC_URL}."
            if language == "es"
            else f"No. MinTIC source hwrp-3su3 reports quarterly fixed-access counts and speed values through {period}; it does not measure traffic, utilization, availability, or saturation. This evidence does not confirm real congestion. {municipal_limit} Source: {MINTIC_URL}."
        )
    elif any(
        term in normalized
        for term in ("neighborhood", "barrio", "barrial", "locality", "localidad")
    ):
        answer = (
            f"La instantánea disponible cubre Tunja por municipio, DIVIPOLA 15001, hasta {period}; no contiene datos por barrio y no distribuyo promedios municipales en zonas inventadas. Los escenarios de puntos de acceso siguen siendo sintéticos. Fuente: {MINTIC_URL}."
            if language == "es"
            else f"The available snapshot covers Tunja at municipality level, DIVIPOLA 15001, through {period}; it has no neighborhood data, and I do not distribute municipal averages across invented areas. Access-point scenarios remain synthetic. Source: {MINTIC_URL}."
        )
    elif any(
        term in normalized
        for term in ("quality", "calidad", "latency", "latencia", "jitter", "speedtest")
    ):
        answer = (
            f"La fuente MinTIC integrada proporciona conteos de accesos fijos, proveedor, tecnología, segmento y valores informados de bajada/subida para {period}. No se han integrado mediciones QoE de descarga de usuario, latencia ni jitter. CRC Postdata mantiene tableros QoE fijos y móviles, pero sus registros no están conectados aquí. Fuentes: {MINTIC_URL} y {POSTDATA_URL}."
            if language == "es"
            else f"The integrated MinTIC source provides fixed-access counts, provider, technology, segment, and reported downstream/upstream values through {period}. User-experience download measurements, latency, and jitter are not integrated. CRC Postdata has fixed and mobile QoE dashboards, but their records are not connected here. Sources: {MINTIC_URL} and {POSTDATA_URL}."
        )
    elif any(term in normalized for term in ("provider", "providers", "proveedor", "proveedores")):
        providers = analysis["latest_period_provider_accesses"]
        names = ", ".join(row["name"] for row in providers[:8]) or "ninguno"
        count = len(providers)
        answer = (
            f"En {period}, el conjunto MinTIC registra {count} proveedores en las filas únicas aceptadas de Tunja. Los ocho principales por accesos informados son: {names}. La cifra es un conteo trimestral informado, no usuarios simultáneos. Fuente: {MINTIC_URL}."
            if language == "es"
            else f"For {period}, the MinTIC dataset lists {count} providers in accepted unique Tunja rows. The top eight by reported accesses are: {names}. These are quarterly reported counts, not simultaneous users. Source: {MINTIC_URL}."
        )
    elif any(
        term in normalized for term in ("technology", "technologies", "tecnologia", "tecnología")
    ):
        technologies = analysis["latest_period_technology_accesses"]
        names = ", ".join(row["name"] for row in technologies)
        answer = (
            f"Las tecnologías informadas para Tunja en {period} son: {names}. La fuente registra categorías de acceso fijo a nivel municipal. Fuente: {MINTIC_URL}."
            if language == "es"
            else f"The technologies reported for Tunja in {period} are: {names}. The source records fixed-access categories at municipality level. Source: {MINTIC_URL}."
        )
    else:
        answer = (
            f"Hay una instantánea válida del conjunto oficial MinTIC de accesos fijos `hwrp-3su3` para Tunja, municipio DIVIPOLA 15001. Incluye {analysis['observation_count']:,} filas aceptadas en {len(analysis['quarterly_accesses'])} trimestres hasta {period}; se recuperó el {batch['retrieved_at_utc']}. Informa proveedor, tecnología, segmento, accesos y valores de bajada/subida, pero no mide saturación. {municipal_limit} Fuente: {MINTIC_URL}."
            if language == "es"
            else f"A validated snapshot of official MinTIC fixed-access dataset `hwrp-3su3` is available for Tunja, municipality DIVIPOLA 15001. It contains {analysis['observation_count']:,} accepted rows across {len(analysis['quarterly_accesses'])} quarters through {period}; retrieved {batch['retrieved_at_utc']}. It reports provider, technology, segment, access counts, and downstream/upstream values, but does not measure saturation. {municipal_limit} Source: {MINTIC_URL}."
        )
    answer += freshness
    return {
        "answer": answer,
        "availability": "stale" if snapshot["stale"] else "available",
        "source_id": "mintic_fixed_accesses",
        "source_url": MINTIC_URL,
        "reporting_period": period,
        "retrieved_at_utc": batch["retrieved_at_utc"],
        "source_updated_at_utc": batch["source_updated_at_utc"],
        "geographic_granularity": "municipality",
        "providers": [row["name"] for row in analysis["latest_period_provider_accesses"]],
        "technologies": [row["name"] for row in analysis["latest_period_technology_accesses"]],
        "reported_accesses_by_quarter": analysis["quarterly_accesses"],
        "available_connectivity_fields": [
            "reported fixed-access counts",
            "provider",
            "technology",
            "segment",
            "reported downstream and upstream values in Mbps",
        ],
        "saturation_ground_truth_available": False,
        "synthetic_fallback_used": False,
    }
