"""MOPS PIT client -- Taiwan (TWSE / TPEX).

Provides point-in-time financial data from Taiwan's Market Observation
Post System (MOPS), operated by the Taiwan Stock Exchange (TWSE).

Coverage: ~1,700+ listed companies, ~$1.2T market cap.
API: https://mops.twse.com.tw  (free, no key required)

MOPS provides structured financial statement data for all TWSE and
TPEX (OTC) listed companies.  Filing dates are tracked by the system.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, cached_post, HTTPError

logger = logging.getLogger(__name__)

_MOPS_BASE = "https://mops.twse.com.tw"


class MOPSError(Exception):
    """Raised on MOPS API failures."""

    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"MOPS error on {endpoint}: {detail}")


class MOPSClient:
    """Point-in-time client for MOPS (Taiwanese equities).

    Implements the ``PITClient`` protocol.

    Note: MOPS uses ROC (Republic of China) calendar years.
    Year 113 in ROC = 2024 CE.  Conversion: CE = ROC + 1911.
    """

    def __init__(self) -> None:
        self._headers = {
            "Accept": "application/json",
            "User-Agent": "Operator1/1.0",
        }

    @property
    def market_id(self) -> str:
        return "tw_mops"

    @property
    def market_name(self) -> str:
        return "Taiwan (TWSE / TPEX) -- MOPS"

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{_MOPS_BASE}{path}"
        try:
            return cached_get(url, params=params, headers=self._headers)
        except HTTPError as exc:
            raise MOPSError(path, str(exc)) from exc

    def _post(self, path: str, data: dict | None = None) -> Any:
        """POST request for MOPS endpoints (form submissions).

        MOPS uses form POST for most data endpoints, not GET.
        Routes through cached_post() for disk caching, rate limiting,
        and audit logging.
        """
        url = f"{_MOPS_BASE}{path}"
        try:
            return cached_post(
                url,
                form_data=data or {},
                headers={
                    "User-Agent": "Operator1/1.0",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
        except HTTPError as exc:
            raise MOPSError(path, str(exc)) from exc

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Taiwanese listed companies.

        MOPS provides company lists via the TWSE website.
        """
        try:
            data = self._post("/mops/web/ajax_t51sb01", data={
                "encodeURIComponent": 1,
                "step": 1,
                "firstin": 1,
                "TYPEK": "sii",  # Listed companies
            })
            # Parse response (HTML table or JSON depending on endpoint)
            companies: list[dict[str, Any]] = []
            if isinstance(data, list):
                for item in data:
                    companies.append({
                        "ticker": str(item.get("公司代號", "")),
                        "name": item.get("公司名稱", "") or item.get("company_name", ""),
                        "country": "TW",
                        "exchange": "TWSE",
                        "market_id": self.market_id,
                    })
        except Exception:
            companies = []

        if query:
            q = query.lower()
            companies = [
                c for c in companies
                if q in c["name"].lower() or q in c["ticker"].lower()
            ]

        return companies

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile, enriched with TWSE / OpenFIGI data."""
        companies = self.list_companies(query=identifier)
        if companies:
            c = companies[0]
            base_profile = {
                "name": c["name"],
                "ticker": c["ticker"],
                "isin": "",
                "country": "TW",
                "sector": "",
                "industry": "",
                "exchange": "TWSE",
                "currency": "TWD",
            }
            # Enrich with TWSE / OpenFIGI data
            from operator1.clients.supplement import enrich_profile
            from operator1.clients.canonical_translator import translate_profile
            enriched = enrich_profile(
                self.market_id,
                c["ticker"],
                existing_profile=base_profile,
                name=c["name"],
            )
            return translate_profile(enriched, self.market_id)
        raise MOPSError("get_profile", f"Company not found: {identifier}")

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        logger.warning("MOPS does not provide OHLCV data for %s.", identifier)
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financial statements from MOPS.

        MOPS provides financial data via form submissions.  The data
        includes filing dates (申報日期) for PIT alignment.
        """
        from datetime import date

        current_year = date.today().year
        roc_year = current_year - 1911  # Convert to ROC calendar

        type_map = {
            "income": "ajax_t164sb04",
            "balance": "ajax_t164sb03",
            "cashflow": "ajax_t164sb05",
        }
        endpoint = type_map.get(statement_type, "ajax_t164sb04")

        rows: list[dict] = []
        for year_offset in range(3):
            year = roc_year - year_offset
            try:
                data = self._post(f"/mops/web/{endpoint}", data={
                    "encodeURIComponent": 1,
                    "step": 1,
                    "firstin": 1,
                    "co_id": identifier,
                    "year": str(year),
                    "season": "",  # All seasons
                })

                if isinstance(data, list):
                    for item in data:
                        rows.append({
                            "concept": item.get("會計項目", ""),
                            "value": item.get("金額", ""),
                            "filing_date": "",  # Parsed from response headers
                            "report_date": f"{year + 1911}-12-31",
                            "fiscal_year": year + 1911,
                            "statement_type": statement_type,
                        })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Translate to canonical format for downstream processing
        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)
