"""SEC EDGAR PIT client -- United States (NYSE / NASDAQ).

Provides true point-in-time financial data from the SEC's EDGAR system.
All filings have immutable filing dates (``acceptedDate``), making this
the gold standard for PIT data.

Coverage: ~10,000+ public companies, $50T+ market cap.
API: https://efts.sec.gov/LATEST  (free, no key required)
     https://data.sec.gov          (bulk XBRL data)

Key endpoints:
  - Company search: /submissions/CIK{cik}.json
  - Full-text search: efts.sec.gov/LATEST/search-index?q=...
  - XBRL companyfacts: data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
  - Filings: data.sec.gov/submissions/CIK{cik}.json

Important: SEC requires a User-Agent header with contact info.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

# SEC requires identifying User-Agent
_SEC_USER_AGENT = "Operator1/1.0 (https://github.com/Abdu0200/Operator-one)"

_EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
_EDGAR_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts"
_EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# In-memory caches
_company_list_cache: list[dict[str, Any]] | None = None
_companyfacts_cache: dict[str, dict] = {}


class SECEdgarError(Exception):
    """Raised on SEC EDGAR API failures."""

    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"SEC EDGAR error on {endpoint}: {detail}")


class SECEdgarClient:
    """Point-in-time client for SEC EDGAR (US equities).

    Implements the ``PITClient`` protocol.  All financial statements
    include ``filing_date`` (the SEC acceptance date) alongside the
    fiscal period ``report_date``.

    Parameters
    ----------
    user_agent:
        SEC requires a User-Agent with your name/email.  Override for
        production use.
    """

    def __init__(self, user_agent: str = _SEC_USER_AGENT) -> None:
        self._user_agent = user_agent
        self._headers = {
            "User-Agent": self._user_agent,
            "Accept": "application/json",
        }

    # -- Protocol properties -------------------------------------------------

    @property
    def market_id(self) -> str:
        return "us_sec_edgar"

    @property
    def market_name(self) -> str:
        return "United States (NYSE / NASDAQ) -- SEC EDGAR"

    # -- Internal helpers ----------------------------------------------------

    def _get(self, url: str, params: dict | None = None) -> Any:
        """GET with SEC-required headers."""
        try:
            return cached_get(url, params=params, headers=self._headers)
        except HTTPError as exc:
            raise SECEdgarError(url, str(exc)) from exc

    @staticmethod
    def _pad_cik(cik: str | int) -> str:
        """Zero-pad a CIK to 10 digits as SEC expects."""
        return str(cik).zfill(10)

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List all SEC-registered companies, optionally filtered by query.

        Uses the SEC company_tickers.json bulk file (~13K entries).
        """
        global _company_list_cache
        if _company_list_cache is None:
            data = self._get(_EDGAR_COMPANY_TICKERS_URL)
            if isinstance(data, dict):
                _company_list_cache = [
                    {
                        "ticker": v.get("ticker", ""),
                        "name": v.get("title", ""),
                        "cik": str(v.get("cik_str", "")),
                        "exchange": "",
                        "market_id": self.market_id,
                    }
                    for v in data.values()
                ]
            else:
                _company_list_cache = []

        if not query:
            return _company_list_cache

        q = query.lower()
        return [
            c for c in _company_list_cache
            if q in c["name"].lower() or q in c["ticker"].lower()
        ]

    def search_company(self, name: str) -> list[dict[str, Any]]:
        """Search for a company by name using the EDGAR full-text search."""
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from SEC submissions endpoint.

        Parameters
        ----------
        identifier:
            CIK number (with or without zero-padding) or ticker symbol.
        """
        # Resolve ticker -> CIK if needed
        cik = self._resolve_cik(identifier)
        padded = self._pad_cik(cik)

        url = f"{_EDGAR_SUBMISSIONS_URL}/CIK{padded}.json"
        data = self._get(url)

        raw_profile = {
            "name": data.get("name", ""),
            "ticker": data.get("tickers", [""])[0] if data.get("tickers") else identifier,
            "cik": cik,
            "isin": "",  # SEC doesn't natively provide ISINs
            "country": "US",
            "sector": data.get("sicDescription", ""),
            "industry": data.get("sicDescription", ""),
            "exchange": data.get("exchanges", [""])[0] if data.get("exchanges") else "",
            "currency": "USD",
            "fiscal_year_end": data.get("fiscalYearEnd", ""),
            "ein": data.get("ein", ""),
            "sic": data.get("sic", ""),
            "filing_count": len(data.get("filings", {}).get("recent", {}).get("accessionNumber", [])),
        }
        # Translate to canonical profile format
        from operator1.clients.canonical_translator import translate_profile
        return translate_profile(raw_profile, self.market_id)

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch income statement data from XBRL companyfacts.

        Returns DataFrame with columns including ``filing_date`` and
        ``report_date`` for true PIT alignment.
        """
        facts = self._get_companyfacts(identifier)
        return self._extract_statement(facts, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        """Fetch balance sheet data from XBRL companyfacts."""
        facts = self._get_companyfacts(identifier)
        return self._extract_statement(facts, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch cash flow statement data from XBRL companyfacts."""
        facts = self._get_companyfacts(identifier)
        return self._extract_statement(facts, "cashflow")

    # -- Price data ----------------------------------------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """SEC EDGAR does not provide OHLCV price data.

        Price data should be sourced separately (e.g. Yahoo Finance,
        or a dedicated price feed).  Returns an empty DataFrame.
        """
        logger.warning(
            "SEC EDGAR does not provide OHLCV data. "
            "Price data for %s should be sourced separately.",
            identifier,
        )
        return pd.DataFrame()

    # -- Peers / executives --------------------------------------------------

    def get_peers(self, identifier: str) -> list[str]:
        """Return peer companies based on SIC code matching.

        Finds companies with the same 4-digit SIC code (exact industry
        match).  Falls back to 2-digit SIC (broad sector match) if fewer
        than 5 exact matches are found.  Returns up to 10 peer tickers.
        """
        profile = self.get_profile(identifier)
        sic = profile.get("sic", "")
        target_ticker = profile.get("ticker", identifier).upper()
        if not sic:
            return []

        # Load the extended company tickers list with SIC codes
        try:
            data = self._get("https://www.sec.gov/files/company_tickers_exchange.json")
            companies = data.get("data", []) if isinstance(data, dict) else []
            # data format: [cik, name, ticker, exchange, sic]
        except Exception:
            # Fallback: can't get SIC-enhanced list
            return []

        # Find companies with same 4-digit SIC
        sic_4 = str(sic).zfill(4)
        sic_2 = sic_4[:2]
        exact_peers: list[str] = []
        broad_peers: list[str] = []

        for row in companies:
            if not isinstance(row, (list, tuple)) or len(row) < 5:
                continue
            peer_ticker = str(row[2]).upper()
            peer_sic = str(row[4]).zfill(4) if row[4] else ""

            # Skip the target itself
            if peer_ticker == target_ticker:
                continue

            if peer_sic == sic_4:
                exact_peers.append(peer_ticker)
            elif peer_sic[:2] == sic_2 and len(broad_peers) < 20:
                broad_peers.append(peer_ticker)

        # Prefer exact SIC matches, pad with broad if needed
        peers = exact_peers[:10]
        if len(peers) < 5:
            remaining = 10 - len(peers)
            peers.extend(broad_peers[:remaining])

        logger.info(
            "SEC EDGAR peers for %s (SIC %s): %d exact, %d broad -> %d returned",
            target_ticker, sic_4, len(exact_peers), len(broad_peers), len(peers),
        )
        return peers[:10]

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        """SEC doesn't have a direct executives endpoint; returns empty."""
        return []

    # -- Internal extraction helpers -----------------------------------------

    def _resolve_cik(self, identifier: str) -> str:
        """Resolve a ticker symbol to a CIK, or return as-is if already CIK."""
        if identifier.isdigit():
            return identifier

        # Search company list for ticker match
        companies = self.list_companies()
        for c in companies:
            if c["ticker"].upper() == identifier.upper():
                return c["cik"]

        raise SECEdgarError(
            "resolve_cik",
            f"Could not resolve '{identifier}' to a CIK number",
        )

    def _get_companyfacts(self, identifier: str) -> dict:
        """Fetch and cache the full XBRL companyfacts for an entity."""
        cik = self._resolve_cik(identifier)
        padded = self._pad_cik(cik)

        if padded in _companyfacts_cache:
            return _companyfacts_cache[padded]

        url = f"{_EDGAR_COMPANYFACTS_URL}/CIK{padded}.json"
        data = self._get(url)
        _companyfacts_cache[padded] = data
        return data

    def _extract_statement(
        self,
        facts: dict,
        statement_type: str,
    ) -> pd.DataFrame:
        """Extract financial statement rows from XBRL companyfacts.

        The companyfacts JSON nests data under us-gaap taxonomy concepts.
        Each concept has units (e.g. USD) with individual filing entries
        that include ``filed`` (filing date) and ``end`` (period end).
        """
        # Concept mapping by statement type
        concept_map = {
            "income": [
                "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "GrossProfit", "OperatingIncomeLoss",
                "NetIncomeLoss", "EarningsPerShareBasic",
                "EarningsPerShareDiluted", "InterestExpense",
                "IncomeTaxExpenseBenefit",
                "SellingGeneralAndAdministrativeExpense",
                "ResearchAndDevelopmentExpense",
            ],
            "balance": [
                "Assets", "Liabilities", "StockholdersEquity",
                "AssetsCurrent", "LiabilitiesCurrent",
                "CashAndCashEquivalentsAtCarryingValue",
                "ShortTermBorrowings", "LongTermDebt",
                "AccountsReceivableNetCurrent",
            ],
            "cashflow": [
                "NetCashProvidedByUsedInOperatingActivities",
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "NetCashProvidedByUsedInInvestingActivities",
                "NetCashProvidedByUsedInFinancingActivities",
                "PaymentsOfDividends",
                "PaymentsForRepurchaseOfCommonStock",
            ],
        }

        concepts = concept_map.get(statement_type, [])
        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        rows: list[dict] = []
        for concept_name in concepts:
            concept_data = us_gaap.get(concept_name, {})
            units = concept_data.get("units", {})

            # Try USD first, then shares
            for unit_key in ("USD", "USD/shares", "shares"):
                entries = units.get(unit_key, [])
                for entry in entries:
                    # Only include 10-K (annual) and 10-Q (quarterly) filings
                    form = entry.get("form", "")
                    if form not in ("10-K", "10-Q", "20-F"):
                        continue

                    rows.append({
                        "concept": concept_name,
                        "value": entry.get("val"),
                        "filing_date": entry.get("filed", ""),
                        "report_date": entry.get("end", ""),
                        "period_start": entry.get("start", ""),
                        "form": form,
                        "fiscal_year": entry.get("fy"),
                        "fiscal_period": entry.get("fp", ""),
                        "unit": unit_key,
                    })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Parse dates
        for col in ("filing_date", "report_date", "period_start"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Translate to canonical format for downstream processing
        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)
