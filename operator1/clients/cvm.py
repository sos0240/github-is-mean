"""CVM PIT client -- Brazil (B3 Exchange).

Provides true point-in-time financial data from Brazil's Comissao de
Valores Mobiliarios (CVM), the Brazilian securities regulator.

Coverage: ~400+ listed companies on B3, $2.2T market cap.
API: https://dados.cvm.gov.br/api/v1  (free, no key required)

CVM provides structured data through their open data portal with
filing dates (dataReferencia / dataEntrega) for PIT alignment.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

_CVM_BASE = "https://dados.cvm.gov.br/api/v1"
_CVM_DATASET_BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA"


class CVMError(Exception):
    """Raised on CVM API failures."""

    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"CVM error on {endpoint}: {detail}")


class CVMClient:
    """Point-in-time client for CVM (Brazilian equities).

    Implements the ``PITClient`` protocol.  All filings include
    ``DT_REFER`` (reference date) and ``DT_RECEB`` (receipt/filing date)
    for true PIT alignment.
    """

    def __init__(self) -> None:
        self._headers = {
            "Accept": "application/json",
            "User-Agent": "Operator1/1.0",
        }

    @property
    def market_id(self) -> str:
        return "br_cvm"

    @property
    def market_name(self) -> str:
        return "Brazil (B3) -- CVM"

    def _get(self, url: str, params: dict | None = None) -> Any:
        try:
            return cached_get(url, params=params, headers=self._headers)
        except HTTPError as exc:
            raise CVMError(url, str(exc)) from exc

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Brazilian public companies registered with CVM."""
        try:
            data = self._get(
                f"{_CVM_BASE}/cia_aberta",
                params={"$top": 500, "$format": "json"},
            )
            items = data.get("value", []) if isinstance(data, dict) else []
        except Exception:
            items = []

        companies: list[dict[str, Any]] = []
        for item in items:
            companies.append({
                "ticker": item.get("CD_CVM", ""),
                "name": item.get("DENOM_SOCIAL", "") or item.get("DENOM_COMERC", ""),
                "cnpj": item.get("CNPJ_CIA", ""),
                "country": "BR",
                "exchange": "B3",
                "sit_reg": item.get("SIT_REG", ""),
                "market_id": self.market_id,
            })

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
        """Fetch company profile, enriched with CVM registry / OpenFIGI data."""
        companies = self.list_companies(query=identifier)
        if companies:
            c = companies[0]
            base_profile = {
                "name": c["name"],
                "ticker": c["ticker"],
                "cnpj": c.get("cnpj", ""),
                "isin": "",
                "country": "BR",
                "sector": "",
                "industry": "",
                "exchange": "B3",
                "currency": "BRL",
            }
            # Enrich with CVM registry / OpenFIGI data
            from operator1.clients.supplement import enrich_profile
            from operator1.clients.canonical_translator import translate_profile
            enriched = enrich_profile(
                self.market_id,
                c["ticker"],
                existing_profile=base_profile,
                cnpj=c.get("cnpj", ""),
                name=c["name"],
            )
            return translate_profile(enriched, self.market_id)
        raise CVMError("get_profile", f"Company not found: {identifier}")

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "DRE")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "BPA")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "DFC_MI")

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        logger.warning("CVM does not provide OHLCV data for %s.", identifier)
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

    def _fetch_financials(
        self,
        identifier: str,
        statement_code: str,
    ) -> pd.DataFrame:
        """Fetch financial statements from CVM open data.

        CVM publishes annual financial data as CSV datasets.
        Statement codes: DRE (income), BPA/BPP (balance), DFC_MI (cashflow).
        """
        from datetime import date

        current_year = date.today().year
        rows: list[dict] = []

        for year in range(current_year - 3, current_year + 1):
            try:
                # CVM annual datasets URL pattern
                url = (
                    f"{_CVM_DATASET_BASE}/DOC/DFP/"
                    f"DADOS/dfp_cia_aberta_{statement_code}_con_{year}.csv"
                )
                # Try to fetch as CSV
                import requests
                resp = requests.get(url, headers=self._headers, timeout=30)
                if resp.status_code == 200:
                    from io import StringIO
                    df_raw = pd.read_csv(
                        StringIO(resp.text),
                        sep=";",
                        encoding="latin-1",
                        on_bad_lines="skip",
                    )

                    # Filter for the requested company
                    if "CD_CVM" in df_raw.columns:
                        df_filtered = df_raw[
                            df_raw["CD_CVM"].astype(str) == str(identifier)
                        ]
                    else:
                        df_filtered = df_raw

                    for _, row in df_filtered.iterrows():
                        rows.append({
                            "concept": row.get("DS_CONTA", ""),
                            "value": row.get("VL_CONTA", ""),
                            "filing_date": row.get("DT_FIM_EXERC", ""),
                            "report_date": row.get("DT_FIM_EXERC", ""),
                            "fiscal_year": year,
                            "account_code": row.get("CD_CONTA", ""),
                        })
            except Exception:
                continue

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
        return translate_financials(df, self.market_id, statement_code)
