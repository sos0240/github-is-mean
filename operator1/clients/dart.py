"""DART PIT client -- South Korea (KOSPI / KOSDAQ).

Provides true point-in-time financial data from Korea's Data Analysis,
Retrieval and Transfer System (DART), operated by the Financial
Supervisory Service (FSS).

Coverage: ~2,500+ listed companies, $2.5T market cap.
API: https://opendart.fss.or.kr/api  (free, API key required)

To get an API key: https://opendart.fss.or.kr/
Registration is free and takes ~1 minute.

Key endpoints:
  - Company list: /company.json
  - Disclosures: /list.json
  - Financial statements: /fnlttSinglAcntAll.json (single company, all accounts)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

_DART_BASE = "https://opendart.fss.or.kr/api"

# In-memory caches
_corp_code_cache: dict[str, str] = {}


class DARTError(Exception):
    """Raised on DART API failures."""

    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"DART error on {endpoint}: {detail}")


class DARTClient:
    """Point-in-time client for DART (South Korean equities).

    Implements the ``PITClient`` protocol.  All disclosures include
    ``rcept_dt`` (receipt/filing date) for true PIT alignment.

    Parameters
    ----------
    api_key:
        DART Open API key (free registration at opendart.fss.or.kr).
    """

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key
        self._headers = {
            "Accept": "application/json",
            "User-Agent": "Operator1/1.0",
        }

    @property
    def market_id(self) -> str:
        return "kr_dart"

    @property
    def market_name(self) -> str:
        return "South Korea (KOSPI / KOSDAQ) -- DART"

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{_DART_BASE}{path}"
        if params is None:
            params = {}
        params["crtfc_key"] = self._api_key
        try:
            return cached_get(url, params=params, headers=self._headers)
        except HTTPError as exc:
            raise DARTError(path, str(exc)) from exc

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Korean companies from DART disclosure search.

        DART's /list.json returns recent disclosures; we deduplicate
        to build a company list.  For a full corp code list, the
        corpCode.xml endpoint would need to be downloaded separately.
        """
        params: dict[str, Any] = {
            "page_count": 100,
            "page_no": 1,
        }
        if query:
            params["corp_name"] = query

        try:
            data = self._get("/list.json", params=params)
            items = data.get("list", []) if isinstance(data, dict) else []
        except Exception:
            items = []

        seen: set[str] = set()
        companies: list[dict[str, Any]] = []
        for item in items:
            corp_code = item.get("corp_code", "")
            if corp_code and corp_code not in seen:
                seen.add(corp_code)
                companies.append({
                    "ticker": item.get("stock_code", ""),
                    "name": item.get("corp_name", ""),
                    "corp_code": corp_code,
                    "country": "KR",
                    "exchange": "KOSPI/KOSDAQ",
                    "market_id": self.market_id,
                })

        return companies

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from DART.

        Parameters
        ----------
        identifier:
            Corp code or stock code.
        """
        try:
            data = self._get("/company.json", params={"corp_code": identifier})
            raw_profile = {
                "name": data.get("corp_name", ""),
                "ticker": data.get("stock_code", identifier),
                "corp_code": data.get("corp_code", identifier),
                "isin": "",
                "country": "KR",
                "sector": data.get("induty_code", ""),
                "industry": data.get("induty_code", ""),
                "exchange": data.get("stock_name", "KOSPI"),
                "currency": "KRW",
                "ceo_name": data.get("ceo_nm", ""),
                "establishment_date": data.get("est_dt", ""),
            }
            # Translate to canonical profile format
            from operator1.clients.canonical_translator import translate_profile
            return translate_profile(raw_profile, self.market_id)
        except Exception:
            raise DARTError("get_profile", f"Company not found: {identifier}")

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "IS")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "BS")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "CF")

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        logger.warning("DART does not provide OHLCV data for %s.", identifier)
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

    # -- Internal helpers ----------------------------------------------------

    def _fetch_financials(
        self,
        identifier: str,
        fs_div: str,
    ) -> pd.DataFrame:
        """Fetch financial statements from DART fnlttSinglAcntAll endpoint.

        Parameters
        ----------
        identifier:
            Corp code.
        fs_div:
            Financial statement division: IS (income), BS (balance), CF (cashflow).
        """
        from datetime import date

        current_year = date.today().year
        rows: list[dict] = []

        # Fetch last 3 years of annual reports
        for year in range(current_year - 3, current_year + 1):
            try:
                data = self._get("/fnlttSinglAcntAll.json", params={
                    "corp_code": identifier,
                    "bsns_year": str(year),
                    "reprt_code": "11011",  # Annual report
                    "fs_div": "CFS",  # Consolidated
                })
                items = data.get("list", []) if isinstance(data, dict) else []

                for item in items:
                    sj_div = item.get("sj_div", "")
                    # Filter by statement type
                    if fs_div == "IS" and sj_div not in ("IS", "CIS"):
                        continue
                    if fs_div == "BS" and sj_div not in ("BS",):
                        continue
                    if fs_div == "CF" and sj_div not in ("CF",):
                        continue

                    rows.append({
                        "concept": item.get("account_nm", ""),
                        "value": item.get("thstrm_amount", ""),
                        "filing_date": item.get("rcept_dt", ""),
                        "report_date": f"{year}-12-31",
                        "fiscal_year": year,
                        "currency": item.get("currency", "KRW"),
                    })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Clean numeric values (remove commas)
        if "value" in df.columns:
            df["value"] = (
                df["value"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Translate to canonical format for downstream processing
        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, fs_div)
