"""ESEF (European Single Electronic Format) PIT client.

Covers pan-EU filings: France (Euronext), Germany (Frankfurt/XETRA),
and all other EU member states that file under the ESEF regulation.

ESEF mandates that all EU-listed issuers publish annual reports in
XBRL-tagged XHTML format.  The filings.xbrl.org portal aggregates
these filings with immutable filing dates.

Coverage: ~5,000+ EU-listed companies, $8-9T combined market cap.
API: https://filings.xbrl.org/api  (free, no key required)

Each MarketInfo in the registry (eu_esef, fr_esef, de_esef) uses this
same client but filters by country code.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

_XBRL_API_BASE = "https://filings.xbrl.org/api"

# In-memory caches per country
_filing_list_cache: dict[str, list[dict]] = {}


class ESEFError(Exception):
    """Raised on ESEF/XBRL API failures."""

    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"ESEF API error on {endpoint}: {detail}")


class ESEFClient:
    """Point-in-time client for ESEF filings (EU equities).

    Implements the ``PITClient`` protocol.  Filters by country code
    to support per-country market entries (EU, FR, DE, etc.).

    Parameters
    ----------
    country_code:
        ISO-2 country code to filter filings.  Use "" for all EU.
    market_id:
        Registry market ID (e.g. "eu_esef", "fr_esef", "de_esef").
    """

    def __init__(
        self,
        country_code: str = "",
        market_id: str = "eu_esef",
    ) -> None:
        self._country_code = country_code.upper()
        self._market_id = market_id
        self._headers = {
            "Accept": "application/json",
            "User-Agent": "Operator1/1.0",
        }

    @property
    def market_id(self) -> str:
        return self._market_id

    @property
    def market_name(self) -> str:
        labels = {
            "eu_esef": "European Union (ESEF -- all EU)",
            "fr_esef": "France (Paris / Euronext -- ESEF)",
            "de_esef": "Germany (Frankfurt / XETRA -- ESEF)",
        }
        return labels.get(self._market_id, f"ESEF ({self._country_code})")

    def _get(self, url: str, params: dict | None = None) -> Any:
        try:
            return cached_get(url, params=params, headers=self._headers)
        except HTTPError as exc:
            raise ESEFError(url, str(exc)) from exc

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List companies with ESEF filings, optionally filtered."""
        cache_key = self._country_code or "ALL"
        if cache_key not in _filing_list_cache:
            params: dict[str, Any] = {"limit": 500}
            if self._country_code:
                params["country"] = self._country_code

            try:
                data = self._get(f"{_XBRL_API_BASE}/filings", params=params)
                filings = data if isinstance(data, list) else data.get("data", [])
            except Exception:
                filings = []

            # Deduplicate by entity name
            seen: set[str] = set()
            companies: list[dict[str, Any]] = []
            for f in filings:
                entity = f.get("entity_name", "") or f.get("entity", "")
                if entity and entity not in seen:
                    seen.add(entity)
                    companies.append({
                        "ticker": f.get("ticker", ""),
                        "name": entity,
                        "lei": f.get("lei", ""),
                        "country": f.get("country", self._country_code),
                        "exchange": f.get("exchange", ""),
                        "market_id": self._market_id,
                    })
            _filing_list_cache[cache_key] = companies

        results = _filing_list_cache[cache_key]
        if query:
            q = query.lower()
            results = [c for c in results if q in c["name"].lower() or q in c.get("ticker", "").lower()]
        return results

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from ESEF filings metadata.

        Enriches the base profile with supplementary data from Euronext
        and OpenFIGI to fill gaps (sector, industry, identifiers).
        """
        companies = self.list_companies(query=identifier)
        if companies:
            c = companies[0]
            base_profile = {
                "name": c["name"],
                "ticker": c.get("ticker", identifier),
                "isin": "",
                "lei": c.get("lei", ""),
                "country": c.get("country", self._country_code),
                "sector": "",
                "industry": "",
                "exchange": c.get("exchange", ""),
                "currency": "EUR",
            }
            # Enrich with Euronext / OpenFIGI data
            from operator1.clients.supplement import enrich_profile
            from operator1.clients.canonical_translator import translate_profile
            enriched = enrich_profile(
                self._market_id,
                c.get("ticker", identifier),
                existing_profile=base_profile,
                name=c["name"],
            )
            return translate_profile(enriched, self._market_id)
        raise ESEFError("get_profile", f"Company not found: {identifier}")

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch income statement data from ESEF XBRL filings."""
        return self._fetch_statements(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_statements(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_statements(identifier, "cashflow")

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """ESEF does not provide price data."""
        logger.warning("ESEF does not provide OHLCV data for %s.", identifier)
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

    # -- Internal helpers ----------------------------------------------------

    # IFRS concept mapping for canonical financial variables
    _CONCEPT_MAP = {
        "income": {
            "ifrs-full:Revenue": "revenue",
            "ifrs-full:GrossProfit": "gross_profit",
            "ifrs-full:ProfitLossFromOperatingActivities": "operating_income",
            "ifrs-full:ProfitLoss": "net_income",
            "ifrs-full:IncomeTaxExpenseContinuingOperations": "taxes",
        },
        "balance": {
            "ifrs-full:Assets": "total_assets",
            "ifrs-full:Liabilities": "total_liabilities",
            "ifrs-full:Equity": "total_equity",
            "ifrs-full:CurrentAssets": "current_assets",
            "ifrs-full:CurrentLiabilities": "current_liabilities",
            "ifrs-full:CashAndCashEquivalents": "cash_and_equivalents",
        },
        "cashflow": {
            "ifrs-full:CashFlowsFromUsedInOperatingActivities": "operating_cash_flow",
            "ifrs-full:CashFlowsFromUsedInInvestingActivities": "investing_cf",
            "ifrs-full:CashFlowsFromUsedInFinancingActivities": "financing_cf",
        },
    }

    def _fetch_statements(
        self,
        identifier: str,
        statement_type: str,
    ) -> pd.DataFrame:
        """Fetch financial statements from ESEF XBRL filing data.

        The filings.xbrl.org API provides filing metadata and may include
        structured XBRL facts.  The response format varies:
        - The ``/filings`` endpoint returns filing metadata (dates, entity)
        - Facts may be nested under ``facts``, ``data``, or require a
          separate ``/filings/{id}/facts`` call

        We extract whatever financial data is available and map IFRS
        concepts to canonical variable names.
        """
        params: dict[str, Any] = {"q": identifier, "limit": 50}
        if self._country_code:
            params["country"] = self._country_code

        try:
            data = self._get(f"{_XBRL_API_BASE}/filings", params=params)
        except Exception as exc:
            logger.warning("ESEF statement fetch failed for %s: %s", identifier, exc)
            return pd.DataFrame()

        # Handle various response formats from filings.xbrl.org
        filings: list[dict] = []
        if isinstance(data, list):
            filings = data
        elif isinstance(data, dict):
            # Could be {data: [...]} or {filings: [...]} or paginated
            filings = (
                data.get("data", [])
                or data.get("filings", [])
                or data.get("results", [])
            )
            if not filings and "entity_name" in data:
                # Single filing returned directly
                filings = [data]

        concept_map = self._CONCEPT_MAP.get(statement_type, {})
        rows: list[dict] = []

        for filing in filings[:20]:  # Limit to 20 most recent
            # Extract dates from various possible field names
            filing_date = (
                filing.get("date_added", "")
                or filing.get("filing_date", "")
                or filing.get("processed", "")
                or filing.get("accepted", "")
            )
            report_date = (
                filing.get("period_end", "")
                or filing.get("report_date", "")
                or filing.get("period", "")
            )
            entity = (
                filing.get("entity_name", "")
                or filing.get("entity", "")
                or filing.get("filer", "")
            )
            filing_id = filing.get("id", "") or filing.get("filing_id", "")

            # Try to get facts from the filing response
            facts = filing.get("facts", {})

            # If no facts in the listing, try a separate facts endpoint
            if not facts and filing_id:
                try:
                    facts_data = self._get(
                        f"{_XBRL_API_BASE}/filings/{filing_id}/facts",
                    )
                    if isinstance(facts_data, dict):
                        facts = facts_data.get("facts", facts_data)
                    elif isinstance(facts_data, list):
                        # Convert list of fact objects to dict
                        for f in facts_data:
                            concept = f.get("concept", "") or f.get("name", "")
                            if concept:
                                facts[concept] = f
                except Exception:
                    pass

            # Extract financial values from facts
            found_values = False
            for xbrl_concept, canonical_name in concept_map.items():
                value = self._extract_fact_value(facts, xbrl_concept)
                if value is not None:
                    rows.append({
                        "concept": canonical_name,
                        "value": value,
                        "filing_date": filing_date,
                        "report_date": report_date,
                        "entity": entity,
                        "source": "xbrl_facts",
                    })
                    found_values = True

            # If no facts extracted, record filing metadata
            if not found_values:
                rows.append({
                    "concept": f"{statement_type}_filing",
                    "value": None,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "entity": entity,
                    "filing_id": filing_id,
                    "source": "metadata_only",
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        n_values = df["value"].notna().sum() if "value" in df.columns else 0
        logger.info(
            "ESEF %s for %s: %d filings, %d financial values extracted",
            statement_type, identifier, len(filings[:20]), n_values,
        )

        # Translate to canonical format for downstream processing
        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self._market_id, statement_type)

    @staticmethod
    def _extract_fact_value(facts: dict, concept: str) -> float | None:
        """Extract a numeric value for a concept from XBRL facts dict."""
        if not facts:
            return None

        # Direct lookup
        val = facts.get(concept)
        if val is not None:
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, dict):
                v = val.get("value") or val.get("amount") or val.get("numericValue")
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
        short = concept.split(":")[-1] if ":" in concept else concept
        for key, val in facts.items():
            if key.endswith(short):
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, dict):
                    v = val.get("value") or val.get("amount") or val.get("numericValue")
                    if v is not None:
                        try:
                            return float(v)
                        except (ValueError, TypeError):
                            pass
        return None
