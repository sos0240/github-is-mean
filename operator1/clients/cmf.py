"""CMF PIT client -- Chile (Santiago Stock Exchange).

Provides point-in-time financial data from Chile's Comision para el
Mercado Financiero (CMF), the Chilean securities regulator.

Coverage: ~200+ listed companies, $0.4T market cap.
API: https://www.cmfchile.cl/api  (free, no key required)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

_CMF_BASE = "https://www.cmfchile.cl"


class CMFError(Exception):
    """Raised on CMF API failures."""

    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"CMF error on {endpoint}: {detail}")


class CMFClient:
    """Point-in-time client for CMF (Chilean equities).

    Implements the ``PITClient`` protocol.
    """

    def __init__(self) -> None:
        self._headers = {
            "Accept": "application/json",
            "User-Agent": "Operator1/1.0",
        }

    @property
    def market_id(self) -> str:
        return "cl_cmf"

    @property
    def market_name(self) -> str:
        return "Chile (Santiago Stock Exchange) -- CMF"

    def _get(self, url: str, params: dict | None = None) -> Any:
        try:
            return cached_get(url, params=params, headers=self._headers)
        except HTTPError as exc:
            raise CMFError(url, str(exc)) from exc

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Chilean public companies registered with CMF."""
        try:
            data = self._get(
                f"{_CMF_BASE}/portal/informacion/entidades/busqueda",
                params={"tipo": "SA", "formato": "json"},
            )
            items = data if isinstance(data, list) else data.get("data", [])
        except Exception:
            items = []

        companies: list[dict[str, Any]] = []
        for item in items:
            companies.append({
                "ticker": item.get("rut", "") or item.get("nemo", ""),
                "name": item.get("razonSocial", "") or item.get("nombre", ""),
                "rut": item.get("rut", ""),
                "country": "CL",
                "exchange": "Santiago",
                "market_id": self.market_id,
            })

        if query:
            q = query.lower()
            companies = [
                c for c in companies
                if q in c["name"].lower() or q in c.get("ticker", "").lower()
            ]

        return companies

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile, enriched with CMF / OpenFIGI data."""
        companies = self.list_companies(query=identifier)
        if companies:
            c = companies[0]
            base_profile = {
                "name": c["name"],
                "ticker": c["ticker"],
                "rut": c.get("rut", ""),
                "isin": "",
                "country": "CL",
                "sector": "",
                "industry": "",
                "exchange": "Santiago",
                "currency": "CLP",
            }
            # Enrich with CMF / OpenFIGI data
            from operator1.clients.supplement import enrich_profile
            from operator1.clients.canonical_translator import translate_profile
            enriched = enrich_profile(
                self.market_id,
                c["ticker"],
                existing_profile=base_profile,
                rut=c.get("rut", ""),
                name=c["name"],
            )
            return translate_profile(enriched, self.market_id)
        raise CMFError("get_profile", f"Company not found: {identifier}")

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        logger.warning("CMF does not provide OHLCV data for %s.", identifier)
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

    def _fetch_financials(
        self,
        identifier: str,
        statement_type: str,
    ) -> pd.DataFrame:
        """Fetch financial statements from CMF XBRL filings.

        CMF requires IFRS-compliant filings from all listed companies.
        """
        try:
            data = self._get(
                f"{_CMF_BASE}/portal/estadisticas/fecu",
                params={
                    "rut": identifier,
                    "formato": "json",
                },
            )
            items = data if isinstance(data, list) else data.get("data", [])
        except Exception:
            return pd.DataFrame()

        rows: list[dict] = []
        for item in items:
            rows.append({
                "concept": item.get("cuenta", "") or item.get("nombre", ""),
                "value": item.get("monto", ""),
                "filing_date": item.get("fechaEnvio", ""),
                "report_date": item.get("periodo", ""),
                "statement_type": statement_type,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Translate to canonical format for downstream processing
        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)
