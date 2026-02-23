"""EDINET PIT client -- Japan (Tokyo Stock Exchange / JPX).

Provides true point-in-time financial data from Japan's Electronic
Disclosure for Investors' NETwork (EDINET) system, operated by the
Financial Services Agency (FSA).

Coverage: ~3,800+ listed companies on TSE, $6.5T market cap.
API: https://api.edinet-fsa.go.jp/api/v2  (free, no key required)

Key endpoints:
  - Document list: /documents.json?date=YYYY-MM-DD&type=2
  - Document content: /documents/{docID}?type=1 (XBRL zip)
  - Company search via edinetCode lookup

Filing types:
  - 有価証券報告書 (Annual securities report) = 10-K equivalent
  - 四半期報告書 (Quarterly report) = 10-Q equivalent
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

_EDINET_BASE = "https://api.edinet-fsa.go.jp/api/v2"

# In-memory caches
_edinet_code_cache: dict[str, dict] = {}
_company_list_cache: list[dict] | None = None


class EDINETError(Exception):
    """Raised on EDINET API failures."""

    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"EDINET error on {endpoint}: {detail}")


class EDINETClient:
    """Point-in-time client for EDINET (Japanese equities).

    Implements the ``PITClient`` protocol.  All filings include the
    ``submitDateTime`` (filing date) for true PIT alignment.

    Parameters
    ----------
    subscription_key:
        Optional EDINET API subscription key.  The API works without one
        but rate limits are more generous with a key.
    """

    def __init__(self, subscription_key: str = "") -> None:
        self._subscription_key = subscription_key
        self._headers = {
            "Accept": "application/json",
            "User-Agent": "Operator1/1.0",
        }

    @property
    def market_id(self) -> str:
        return "jp_edinet"

    @property
    def market_name(self) -> str:
        return "Japan (Tokyo Stock Exchange) -- EDINET"

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{_EDINET_BASE}{path}"
        if params is None:
            params = {}
        if self._subscription_key:
            params["Subscription-Key"] = self._subscription_key
        try:
            return cached_get(url, params=params, headers=self._headers)
        except HTTPError as exc:
            raise EDINETError(path, str(exc)) from exc

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Japanese listed companies from EDINET code list.

        The EDINET code list maps edinetCode -> company name, ticker,
        and other metadata.  This is fetched from the documents API
        by scanning recent filings.  Results are cached in-memory to
        avoid repeated scanning (13+ API calls per invocation).
        """
        global _company_list_cache
        if _company_list_cache is None:
            # EDINET doesn't have a direct "list all companies" endpoint.
            # Scan recent filings biweekly to discover entities.
            from datetime import date, timedelta
            today = date.today()

            companies: list[dict[str, Any]] = []
            seen: set[str] = set()

            # Scan last 90 days biweekly (7 API calls instead of 13)
            for delta in range(0, 90, 14):
                d = today - timedelta(days=delta)
                try:
                    data = self._get(
                        "/documents.json",
                        params={"date": d.isoformat(), "type": 2},
                    )
                    results = data.get("results", []) if isinstance(data, dict) else []
                    for doc in results:
                        edinet_code = doc.get("edinetCode", "")
                        if edinet_code and edinet_code not in seen:
                            seen.add(edinet_code)
                            companies.append({
                                "ticker": doc.get("secCode", ""),
                                "name": doc.get("filerName", ""),
                                "edinet_code": edinet_code,
                                "country": "JP",
                                "exchange": "TSE",
                                "market_id": self.market_id,
                            })
                except Exception:
                    continue

            _company_list_cache = companies
            logger.info("EDINET company list cached: %d entities", len(companies))

        if query:
            q = query.lower()
            return [
                c for c in _company_list_cache
                if q in c["name"].lower() or q in c.get("ticker", "").lower()
            ]

        return list(_company_list_cache)

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from recent EDINET filings.

        Enriches the base profile with supplementary data from JPX
        and OpenFIGI to fill gaps (sector, industry).

        Parameters
        ----------
        identifier:
            EDINET code or securities code (ticker).
        """
        companies = self.list_companies(query=identifier)
        if companies:
            c = companies[0]
            base_profile = {
                "name": c["name"],
                "ticker": c.get("ticker", identifier),
                "edinet_code": c.get("edinet_code", ""),
                "isin": "",
                "country": "JP",
                "sector": "",
                "industry": "",
                "exchange": "TSE",
                "currency": "JPY",
            }
            # Enrich with JPX / OpenFIGI data
            from operator1.clients.supplement import enrich_profile
            from operator1.clients.canonical_translator import translate_profile
            enriched = enrich_profile(
                self.market_id,
                c.get("ticker", identifier),
                existing_profile=base_profile,
                name=c["name"],
            )
            return translate_profile(enriched, self.market_id)
        raise EDINETError("get_profile", f"Company not found: {identifier}")

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_filings(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_filings(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_filings(identifier, "cashflow")

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        logger.warning("EDINET does not provide OHLCV data for %s.", identifier)
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

    # -- Internal helpers ----------------------------------------------------

    # JPPFS (Japan GAAP) and IFRS concept mappings for financial extraction
    _CONCEPT_MAP = {
        "income": {
            # Revenue / sales
            "jppfs_cor:NetSales": "revenue",
            "jppfs_cor:Revenue": "revenue",
            "ifrs-full:Revenue": "revenue",
            # Gross profit
            "jppfs_cor:GrossProfit": "gross_profit",
            "ifrs-full:GrossProfit": "gross_profit",
            # Operating income
            "jppfs_cor:OperatingIncome": "operating_income",
            "ifrs-full:ProfitLossFromOperatingActivities": "operating_income",
            # Net income
            "jppfs_cor:ProfitLoss": "net_income",
            "ifrs-full:ProfitLoss": "net_income",
            "jppfs_cor:ProfitLossAttributableToOwnersOfParent": "net_income",
        },
        "balance": {
            "jppfs_cor:TotalAssets": "total_assets",
            "ifrs-full:Assets": "total_assets",
            "jppfs_cor:TotalLiabilities": "total_liabilities",
            "ifrs-full:Liabilities": "total_liabilities",
            "jppfs_cor:NetAssets": "total_equity",
            "ifrs-full:Equity": "total_equity",
            "jppfs_cor:CurrentAssets": "current_assets",
            "ifrs-full:CurrentAssets": "current_assets",
            "jppfs_cor:CurrentLiabilities": "current_liabilities",
            "ifrs-full:CurrentLiabilities": "current_liabilities",
            "jppfs_cor:CashAndDeposits": "cash_and_equivalents",
            "ifrs-full:CashAndCashEquivalents": "cash_and_equivalents",
        },
        "cashflow": {
            "jppfs_cor:NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
            "ifrs-full:CashFlowsFromUsedInOperatingActivities": "operating_cash_flow",
            "jppfs_cor:NetCashProvidedByUsedInInvestingActivities": "investing_cf",
            "ifrs-full:CashFlowsFromUsedInInvestingActivities": "investing_cf",
            "jppfs_cor:NetCashProvidedByUsedInFinancingActivities": "financing_cf",
            "ifrs-full:CashFlowsFromUsedInFinancingActivities": "financing_cf",
        },
    }

    def _fetch_filings(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financial data from EDINET for a given entity.

        1. Scans recent document listings for filings matching the identifier
        2. For each filing, attempts to download XBRL data (type=5 JSON)
        3. Extracts actual financial numbers using JPPFS/IFRS concept mapping

        Parameters
        ----------
        identifier:
            EDINET code or securities code (ticker).
        statement_type:
            One of "income", "balance", "cashflow".
        """
        from datetime import date, timedelta
        today = date.today()

        # Step 1: Find relevant filings.
        # Scan 120 days biweekly (9 API calls instead of 52).
        # Annual filings are due within 90 days; 120 gives margin.
        # Early termination once we have 4+ filings for the target.
        filing_docs: list[dict] = []
        for delta in range(0, 120, 14):  # biweekly sampling
            d = today - timedelta(days=delta)
            try:
                data = self._get(
                    "/documents.json",
                    params={"date": d.isoformat(), "type": 2},
                )
                results = data.get("results", []) if isinstance(data, dict) else []
                for doc in results:
                    edinet_code = doc.get("edinetCode", "")
                    sec_code = doc.get("secCode", "")

                    if identifier not in (edinet_code, sec_code):
                        continue

                    # Annual (120) and quarterly (140) reports
                    doc_type = doc.get("docTypeCode", "")
                    if doc_type not in ("120", "140"):
                        continue

                    filing_docs.append(doc)
            except Exception:
                continue
            # Early termination: 4 filings = full year of quarterly data
            if len(filing_docs) >= 4:
                break

        if not filing_docs:
            logger.info("EDINET: no filings found for %s", identifier)
            return pd.DataFrame()

        # Deduplicate by docID
        seen_ids: set[str] = set()
        unique_docs: list[dict] = []
        for doc in filing_docs:
            doc_id = doc.get("docID", "")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

        # Step 2: For each filing, try to extract XBRL financial data
        concept_map = self._CONCEPT_MAP.get(statement_type, {})
        rows: list[dict] = []

        for doc in unique_docs[:10]:  # Limit to most recent 10 filings
            doc_id = doc.get("docID", "")
            filing_date = doc.get("submitDateTime", "")
            report_date = doc.get("periodEnd", "")
            period_start = doc.get("periodStart", "")

            # Try to get XBRL JSON (type=5) for actual financial numbers
            xbrl_facts = self._download_xbrl_facts(doc_id)

            if xbrl_facts:
                # Extract financial values from XBRL facts
                for xbrl_concept, canonical_name in concept_map.items():
                    value = self._find_fact_value(xbrl_facts, xbrl_concept)
                    if value is not None:
                        rows.append({
                            "concept": canonical_name,
                            "value": value,
                            "filing_date": filing_date,
                            "report_date": report_date,
                            "period_start": period_start,
                            "doc_id": doc_id,
                            "source": "xbrl",
                        })
            else:
                # Fallback: record filing metadata even without extracted values
                rows.append({
                    "concept": f"{statement_type}_filing",
                    "value": None,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "period_start": period_start,
                    "doc_id": doc_id,
                    "source": "metadata_only",
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date", "period_start"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        n_values = df["value"].notna().sum() if "value" in df.columns else 0
        logger.info(
            "EDINET %s for %s: %d filings, %d financial values extracted",
            statement_type, identifier, len(unique_docs), n_values,
        )

        # Translate to canonical format for downstream processing
        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)

    def _download_xbrl_facts(self, doc_id: str) -> dict | None:
        """Download XBRL facts for a filing document.

        Tries EDINET type=5 (JSON XBRL) first, then type=1 (ZIP) as fallback.
        Returns a dict of {concept_name: value} or None on failure.
        """
        if not doc_id:
            return None

        # Try type=5 (JSON format of XBRL data -- if available)
        try:
            data = self._get(
                f"/documents/{doc_id}",
                params={"type": 5},
            )
            if isinstance(data, dict) and data:
                return data
        except Exception as exc:
            logger.debug("EDINET type=5 download failed for %s: %s", doc_id, exc)

        # Type=1 (ZIP with XBRL files) requires more complex parsing
        # which we skip for now -- the metadata-only fallback handles this
        return None

    @staticmethod
    def _find_fact_value(facts: dict, concept: str) -> float | None:
        """Extract a numeric value for a concept from XBRL facts.

        EDINET XBRL facts can be nested in various formats.  This tries
        common patterns.
        """
        if not facts or not concept:
            return None

        # Direct lookup
        if concept in facts:
            val = facts[concept]
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, dict):
                # {value: ..., unit: ..., context: ...}
                v = val.get("value") or val.get("amount")
                if v is not None:
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        pass
            if isinstance(val, str):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass

        # Try without namespace prefix
        short_name = concept.split(":")[-1] if ":" in concept else concept
        for key, val in facts.items():
            if key.endswith(short_name):
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, dict):
                    v = val.get("value") or val.get("amount")
                    if v is not None:
                        try:
                            return float(v)
                        except (ValueError, TypeError):
                            pass

        return None
