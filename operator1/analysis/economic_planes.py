"""Economic plane classification for the 5-plane Sudoku world model.

Maps companies to one of five economic planes (Supply, Manufacturing,
Consumption, Logistics, Financial Services) based on their sector and
industry classification from the equity provider.

The plane classification informs:
- How linked variables flow between entities
- Which inter-plane connections matter most
- Report narrative about the company's economic position

Spec ref: The_Apps_core_idea.pdf Section A
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from operator1.config_loader import load_config

logger = logging.getLogger(__name__)

_PLANES_CONFIG: dict[str, Any] | None = None


def _load_planes_config() -> dict[str, Any]:
    """Load and cache the economic planes config."""
    global _PLANES_CONFIG
    if _PLANES_CONFIG is None:
        try:
            _PLANES_CONFIG = load_config("economic_planes")
        except Exception:
            logger.warning("economic_planes.yml not found, using defaults")
            _PLANES_CONFIG = {"planes": {}, "default_plane": "manufacturing"}
    return _PLANES_CONFIG


def classify_economic_plane(
    sector: str | None,
    industry: str | None,
) -> dict[str, Any]:
    """Classify a company into the 5-plane economic model.

    Parameters
    ----------
    sector:
        GICS-style sector from the equity provider profile.
    industry:
        GICS-style industry from the equity provider profile.

    Returns
    -------
    dict with keys:
        - primary_plane: str (the best-matching plane name)
        - secondary_planes: list[str] (other planes that match)
        - plane_label: str (human-readable label)
        - plane_description: str
    """
    config = _load_planes_config()
    planes = config.get("planes", {})
    default = config.get("default_plane", "manufacturing")

    sector_lower = (sector or "").lower().strip()
    industry_lower = (industry or "").lower().strip()

    matches: list[tuple[str, int]] = []  # (plane_name, score)

    for plane_name, plane_def in planes.items():
        score = 0

        # Check sector match
        for s in plane_def.get("sectors", []):
            if s.lower() == sector_lower:
                score += 10
                break
            if s.lower() in sector_lower or sector_lower in s.lower():
                score += 5

        # Check industry match
        for ind in plane_def.get("industries", []):
            if ind.lower() == industry_lower:
                score += 8
                break
            if ind.lower() in industry_lower or industry_lower in ind.lower():
                score += 4

        if score > 0:
            matches.append((plane_name, score))

    # Sort by score descending
    matches.sort(key=lambda x: -x[1])

    if matches:
        primary = matches[0][0]
        secondary = [m[0] for m in matches[1:] if m[1] >= 4]
    else:
        primary = default
        secondary = []

    plane_def = planes.get(primary, {})

    result = {
        "primary_plane": primary,
        "secondary_planes": secondary,
        "plane_label": plane_def.get("label", primary.replace("_", " ").title()),
        "plane_description": plane_def.get("description", ""),
    }

    logger.info(
        "Economic plane: %s -> %s (%s)",
        sector or "unknown",
        primary,
        result["plane_label"],
    )

    return result


def get_relationship_plane(relationship_type: str) -> str:
    """Map a relationship type to its economic plane.

    Parameters
    ----------
    relationship_type:
        One of: competitors, suppliers, customers,
        financial_institutions, logistics, regulators.

    Returns
    -------
    The plane name, or "same_plane" for competitors.
    """
    config = _load_planes_config()
    mapping = config.get("relationship_to_plane", {})
    return mapping.get(relationship_type, "same_plane")
