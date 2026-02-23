"""Portfolio awareness: institutional overlap and user portfolio context.

Implements two dimensions of portfolio analysis:

1. **Institutional ownership overlap** -- shared large holders between
   the target and linked entities create correlated selling/buying
   pressure (systemic contagion channel).

2. **User portfolio context** -- if the user provides their existing
   holdings, compute correlation, concentration risk, and marginal
   Value-at-Risk contribution.

Spec ref: The_Apps_core_idea.pdf Section A (portfolio relationships)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class InstitutionalOverlapResult:
    """Institutional ownership overlap between target and linked entities."""

    available: bool = False
    target_top_holders: list[dict[str, Any]] = field(default_factory=list)
    overlap_scores: dict[str, float] = field(default_factory=dict)  # isin -> score
    portfolio_concentration_hhi: float = 0.0
    n_shared_holders: int = 0
    error: str | None = None


@dataclass
class PortfolioContextResult:
    """Analysis of how the target fits the user's existing portfolio."""

    available: bool = False
    n_holdings: int = 0
    correlation_with_portfolio: float = float("nan")
    marginal_var_contribution: float = float("nan")
    sector_concentration: float = float("nan")
    diversification_benefit: str = "unknown"
    recommendation_adjustment: str = ""
    holdings_detail: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Institutional ownership overlap
# ---------------------------------------------------------------------------


def compute_institutional_overlap(
    target_holders: list[dict[str, Any]],
    linked_holders: dict[str, list[dict[str, Any]]],
) -> InstitutionalOverlapResult:
    """Compute ownership overlap between target and linked entities.

    Parameters
    ----------
    target_holders:
        List of institutional holders for the target company.
        Each dict: {"holder": name, "shares": int, "dateReported": str, ...}
    linked_holders:
        {isin: [holder_dicts]} for each linked entity.

    Returns
    -------
    InstitutionalOverlapResult
    """
    result = InstitutionalOverlapResult(available=True)

    if not target_holders:
        result.available = False
        result.error = "No institutional holder data available"
        return result

    # Extract target holder names
    target_names = {
        h.get("holder", "").lower().strip()
        for h in target_holders
        if h.get("holder")
    }
    result.target_top_holders = target_holders[:10]

    # Compute overlap score per linked entity
    all_shared = set()
    for isin, holders in linked_holders.items():
        if not holders:
            continue
        linked_names = {
            h.get("holder", "").lower().strip()
            for h in holders
            if h.get("holder")
        }
        shared = target_names & linked_names
        if shared:
            overlap_score = len(shared) / max(len(target_names), 1)
            result.overlap_scores[isin] = round(overlap_score, 4)
            all_shared |= shared

    result.n_shared_holders = len(all_shared)

    # Portfolio concentration (Herfindahl of target holder shares)
    total_shares = sum(h.get("shares", 0) for h in target_holders)
    if total_shares > 0:
        shares_fractions = [
            (h.get("shares", 0) / total_shares) ** 2
            for h in target_holders
        ]
        result.portfolio_concentration_hhi = round(sum(shares_fractions), 4)

    return result


# ---------------------------------------------------------------------------
# User portfolio context
# ---------------------------------------------------------------------------


def parse_portfolio_string(portfolio_str: str) -> list[dict[str, Any]]:
    """Parse a portfolio string like 'AAPL:30,MSFT:25,GOOGL:20'.

    Returns list of {"symbol": str, "weight": float}.
    """
    holdings: list[dict[str, Any]] = []
    for item in portfolio_str.split(","):
        item = item.strip()
        if ":" in item:
            symbol, weight_str = item.split(":", 1)
            try:
                weight = float(weight_str.strip())
            except ValueError:
                weight = 0.0
            holdings.append({"symbol": symbol.strip().upper(), "weight": weight})
        elif item:
            holdings.append({"symbol": item.strip().upper(), "weight": 0.0})
    return holdings


def compute_portfolio_context(
    target_returns: pd.Series,
    target_sector: str,
    holdings: list[dict[str, Any]],
    holding_returns: dict[str, pd.Series],
    holding_sectors: dict[str, str],
) -> PortfolioContextResult:
    """Analyse how the target company fits the user's existing portfolio.

    Parameters
    ----------
    target_returns:
        Daily return series for the target company.
    target_sector:
        Sector of the target company.
    holdings:
        List of {"symbol": str, "weight": float} from user input.
    holding_returns:
        {symbol: daily_return_series} for each user holding.
    holding_sectors:
        {symbol: sector} for each user holding.

    Returns
    -------
    PortfolioContextResult
    """
    result = PortfolioContextResult(available=True, n_holdings=len(holdings))

    if not holdings or not holding_returns:
        result.available = False
        result.error = "No portfolio holdings provided"
        return result

    # Normalize weights to sum to 1
    total_weight = sum(h["weight"] for h in holdings)
    if total_weight <= 0:
        total_weight = len(holdings)
        for h in holdings:
            h["weight"] = 1.0 / len(holdings)
    else:
        for h in holdings:
            h["weight"] /= total_weight

    # Build portfolio return series (weighted sum)
    portfolio_returns = pd.Series(0.0, index=target_returns.index)
    for h in holdings:
        sym = h["symbol"]
        if sym in holding_returns:
            aligned = holding_returns[sym].reindex(target_returns.index).fillna(0)
            portfolio_returns += aligned * h["weight"]

    # Correlation with portfolio
    valid_mask = target_returns.notna() & portfolio_returns.notna()
    if valid_mask.sum() > 30:
        corr = float(target_returns[valid_mask].corr(portfolio_returns[valid_mask]))
        result.correlation_with_portfolio = round(corr, 4)

    # Sector concentration
    target_sector_lower = (target_sector or "").lower()
    same_sector_weight = sum(
        h["weight"]
        for h in holdings
        if (holding_sectors.get(h["symbol"], "").lower() == target_sector_lower)
    )
    # Adding the target as a hypothetical equal-weight addition
    hypothetical_target_weight = 1.0 / (len(holdings) + 1)
    result.sector_concentration = round(
        same_sector_weight + hypothetical_target_weight, 4
    )

    # Marginal VaR contribution (simplified)
    if valid_mask.sum() > 30:
        portfolio_vol = float(portfolio_returns[valid_mask].std()) * np.sqrt(252)
        target_vol = float(target_returns[valid_mask].std()) * np.sqrt(252)
        if portfolio_vol > 0:
            beta = corr * target_vol / portfolio_vol if corr == corr else 0
            result.marginal_var_contribution = round(
                beta * hypothetical_target_weight, 4
            )

    # Diversification benefit assessment
    corr_val = result.correlation_with_portfolio
    if corr_val != corr_val:  # NaN
        result.diversification_benefit = "insufficient data"
    elif corr_val < 0.3:
        result.diversification_benefit = "high"
        result.recommendation_adjustment = (
            "Good diversifier -- low correlation with existing holdings"
        )
    elif corr_val < 0.6:
        result.diversification_benefit = "moderate"
        result.recommendation_adjustment = (
            "Moderate diversification -- some overlap with existing risk"
        )
    elif corr_val < 0.8:
        result.diversification_benefit = "low"
        result.recommendation_adjustment = (
            "Limited diversification -- high correlation with portfolio"
        )
    else:
        result.diversification_benefit = "none"
        result.recommendation_adjustment = (
            "CAUTION -- very high correlation with existing holdings, "
            "adding this position increases concentration risk"
        )

    # Sector concentration warning
    if result.sector_concentration > 0.4:
        result.recommendation_adjustment += (
            f". Sector concentration would reach "
            f"{result.sector_concentration:.0%} -- consider rebalancing"
        )

    # Per-holding detail
    for h in holdings:
        sym = h["symbol"]
        detail: dict[str, Any] = {
            "symbol": sym,
            "weight": round(h["weight"] * 100, 1),
            "sector": holding_sectors.get(sym, "unknown"),
        }
        if sym in holding_returns:
            hr = holding_returns[sym].reindex(target_returns.index)
            pair_corr = target_returns.corr(hr)
            detail["correlation_with_target"] = (
                round(float(pair_corr), 4) if pair_corr == pair_corr else None
            )
        result.holdings_detail.append(detail)

    return result
