"""Companies House PIT client -- United Kingdom (LSE).

Provides true point-in-time financial data from the UK Companies House
filing system.  All filings have immutable ``date_of_receipt`` timestamps.

Coverage: ~4,000+ listed companies on LSE, $3.18T market cap.
API: https://api.company-information.service.gov.uk  (free, key required)

To get an API key: https://developer.company-information.service.gov.uk/
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)

_CH_BASE_URL = "https://api.company-information.service.gov.uk"


class CompaniesHouseError(Exception):
    """Raised on Companies House API failures."""

    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"Companies House error on {endpoint}: {detail}")


class CompaniesHouseClient:
    """Point-in-time client for UK Companies House.

    Implements the ``PITClient`` protocol.

    Parameters
    ----------
    api_key:
        Companies House API key (HTTP Basic auth with key as username).
    """

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key
        self._headers = {
            "Accept": "application/json",
            "User-Agent": "Operator1/1.0",
        }

    @property
    def market_id(self) -> str:
        return "uk_companies_house"

    @property
    def market_name(self) -> str:
        return "United Kingdom (LSE) -- Companies House"

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{_CH_BASE_URL}{path}"
        headers = dict(self._headers)
        # Companies House uses HTTP Basic auth (key as username, empty password).
        # Since cached_get doesn't support auth, we encode it in the header.
        if self._api_key:
            import base64
            encoded = base64.b64encode(f"{self._api_key}:".encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        try:
            return cached_get(url, params=params, headers=headers)
        except HTTPError as exc:
            raise CompaniesHouseError(path, str(exc)) from exc

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """Search Companies House for companies matching query."""
        if not query:
            return []  # CH requires a search term

        data = self._get("/search/companies", params={"q": query, "items_per_page": 50})
        items = data.get("items", []) if isinstance(data, dict) else []

        return [
            {
                "ticker": "",  # CH doesn't use tickers
                "name": item.get("title", ""),
                "company_number": item.get("company_number", ""),
                "country": "GB",
                "status": item.get("company_status", ""),
                "type": item.get("company_type", ""),
                "date_of_creation": item.get("date_of_creation", ""),
                "market_id": self.market_id,
            }
            for item in items
        ]

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile by company number.

        Translates raw Companies House response to canonical profile format.
        """
        data = self._get(f"/company/{identifier}")
        raw_profile = {
            "name": data.get("company_name", ""),
            "ticker": "",
            "company_number": identifier,
            "isin": "",
            "country": "GB",
            "sector": data.get("sic_codes", [""])[0] if data.get("sic_codes") else "",
            "industry": ", ".join(data.get("sic_codes", [])),
            "exchange": "LSE",
            "currency": "GBP",
            "status": data.get("company_status", ""),
            "type": data.get("type", ""),
            "date_of_creation": data.get("date_of_creation", ""),
        }
        from operator1.clients.canonical_translator import translate_profile
        return translate_profile(raw_profile, self.market_id)

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch filing history and extract accounts data."""
        return self._fetch_accounts(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_accounts(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_accounts(identifier, "cashflow")

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        logger.warning("Companies House does not provide OHLCV data for %s.", identifier)
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch officers (directors) for a company."""
        try:
            data = self._get(f"/company/{identifier}/officers")
            items = data.get("items", []) if isinstance(data, dict) else []
            return [
                {
                    "name": o.get("name", ""),
                    "role": o.get("officer_role", ""),
                    "appointed_on": o.get("appointed_on", ""),
                    "resigned_on": o.get("resigned_on", ""),
                }
                for o in items
            ]
        except Exception:
            return []

    # -- Internal helpers ----------------------------------------------------

    # UK GAAP and IFRS iXBRL concept mapping
    _CONCEPT_MAP = {
        "income": {
            "uk-gaap:Turnover": "revenue",
            "uk-gaap:TurnoverRevenue": "revenue",
            "ifrs-full:Revenue": "revenue",
            "uk-gaap:GrossProfitLoss": "gross_profit",
            "ifrs-full:GrossProfit": "gross_profit",
            "uk-gaap:OperatingProfitLoss": "operating_income",
            "ifrs-full:ProfitLossFromOperatingActivities": "operating_income",
            "uk-gaap:ProfitLossOnOrdinaryActivitiesBeforeTax": "pretax_income",
            "ifrs-full:ProfitLossBeforeTax": "pretax_income",
            "uk-gaap:ProfitLossForPeriod": "net_income",
            "ifrs-full:ProfitLoss": "net_income",
        },
        "balance": {
            "uk-gaap:FixedAssets": "total_assets",
            "uk-gaap:TotalAssetsLessCurrentLiabilities": "total_assets",
            "ifrs-full:Assets": "total_assets",
            "uk-gaap:Creditors": "total_liabilities",
            "ifrs-full:Liabilities": "total_liabilities",
            "uk-gaap:ShareholderFunds": "total_equity",
            "ifrs-full:Equity": "total_equity",
            "uk-gaap:CurrentAssets": "current_assets",
            "ifrs-full:CurrentAssets": "current_assets",
            "uk-gaap:CreditorsDueWithinOneYear": "current_liabilities",
            "ifrs-full:CurrentLiabilities": "current_liabilities",
            "uk-gaap:CashBankInHand": "cash_and_equivalents",
            "ifrs-full:CashAndCashEquivalents": "cash_and_equivalents",
        },
        "cashflow": {
            "uk-gaap:NetCashInflowOutflowFromOperatingActivities": "operating_cash_flow",
            "ifrs-full:CashFlowsFromUsedInOperatingActivities": "operating_cash_flow",
            "uk-gaap:NetCashInflowOutflowFromInvestingActivities": "investing_cf",
            "ifrs-full:CashFlowsFromUsedInInvestingActivities": "investing_cf",
            "uk-gaap:NetCashInflowOutflowFromFinancingActivities": "financing_cf",
            "ifrs-full:CashFlowsFromUsedInFinancingActivities": "financing_cf",
        },
    }

    def _fetch_accounts(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch filing history and extract financial data from accounts.

        1. Gets the filing history for accounts category
        2. For each filing, attempts to download the iXBRL document
        3. Parses financial values from iXBRL tags using concept mapping

        Companies House filing documents are HTML with embedded iXBRL
        tags containing the actual financial numbers.
        """
        try:
            data = self._get(
                f"/company/{identifier}/filing-history",
                params={"category": "accounts", "items_per_page": 50},
            )
            items = data.get("items", []) if isinstance(data, dict) else []
        except Exception as exc:
            logger.warning("CH filing history fetch failed: %s", exc)
            return pd.DataFrame()

        concept_map = self._CONCEPT_MAP.get(statement_type, {})
        rows: list[dict] = []

        for item in items[:10]:  # Limit to most recent 10 filings
            filing_date = item.get("date", "")
            report_date = item.get("description_values", {}).get("made_up_date", "")
            transaction_id = item.get("transaction_id", "")
            doc_links = item.get("links", {})

            # Try to download and parse the iXBRL document
            xbrl_values = {}
            if transaction_id and self._api_key:
                xbrl_values = self._download_ixbrl_values(
                    identifier, transaction_id, doc_links, concept_map,
                )

            if xbrl_values:
                for canonical_name, value in xbrl_values.items():
                    rows.append({
                        "concept": canonical_name,
                        "value": value,
                        "filing_date": filing_date,
                        "report_date": report_date,
                        "transaction_id": transaction_id,
                        "source": "ixbrl",
                    })
            else:
                # Fallback: record filing metadata
                rows.append({
                    "concept": f"{statement_type}_filing",
                    "value": None,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "category": item.get("category", ""),
                    "description": item.get("description", ""),
                    "type": item.get("type", ""),
                    "transaction_id": transaction_id,
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
            "Companies House %s for %s: %d filings, %d financial values extracted",
            statement_type, identifier, len(items[:10]), n_values,
        )

        # Translate to canonical format for downstream processing
        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)

    def _download_ixbrl_values(
        self,
        company_number: str,
        transaction_id: str,
        doc_links: dict,
        concept_map: dict[str, str],
    ) -> dict[str, float]:
        """Download an iXBRL accounts document and extract financial values.

        Returns a dict of {canonical_name: value} for found concepts.
        """
        # Get document metadata to find the download URL
        try:
            doc_meta_url = doc_links.get("document_metadata", "")
            if not doc_meta_url:
                doc_meta_url = f"/company/{company_number}/filing-history/{transaction_id}"

            meta = self._get(doc_meta_url)
            resources = meta.get("resources", {}) if isinstance(meta, dict) else {}

            # Find the iXBRL document (usually the first .html resource)
            doc_url = None
            for res_key, res_info in resources.items():
                content_type = ""
                if isinstance(res_info, dict):
                    content_type = res_info.get("content_type", "")
                if "html" in content_type.lower() or res_key.endswith(".html"):
                    doc_url = f"/document/{transaction_id}/{res_key}"
                    break

            if not doc_url:
                return {}

            # Download the iXBRL HTML document
            import requests
            import base64

            url = f"https://document-api.company-information.service.gov.uk{doc_url}"
            headers = {"Accept": "text/html"}
            if self._api_key:
                encoded = base64.b64encode(f"{self._api_key}:".encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"

            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                return {}

            # Parse iXBRL tags from the HTML
            return self._parse_ixbrl_html(resp.text, concept_map)

        except Exception as exc:
            logger.debug("CH iXBRL download failed for %s: %s", transaction_id, exc)
            return {}

    @staticmethod
    def _parse_ixbrl_html(html: str, concept_map: dict[str, str]) -> dict[str, float]:
        """Extract financial values from iXBRL-tagged HTML.

        iXBRL embeds XBRL facts in HTML using tags like:
        ``<ix:nonFraction name="uk-gaap:Turnover" ...>1,234,567</ix:nonFraction>``
        """
        import re

        values: dict[str, float] = {}

        # Match ix:nonFraction tags (which contain numeric financial data)
        # Pattern: <ix:nonFraction name="concept:Name" ...>value</ix:nonFraction>
        pattern = re.compile(
            r'<ix:nonFraction[^>]*name=["\']([^"\']+)["\'][^>]*'
            r'(?:sign=["\']([^"\']*)["\'])?[^>]*>'
            r'([^<]*)</ix:nonFraction>',
            re.IGNORECASE,
        )

        for match in pattern.finditer(html):
            concept_name = match.group(1)
            sign = match.group(2) or ""
            raw_value = match.group(3).strip()

            # Check if this concept is one we care about
            canonical = concept_map.get(concept_name)
            if not canonical:
                # Try matching by local name (without namespace)
                for mapped_concept, mapped_canonical in concept_map.items():
                    if concept_name.endswith(mapped_concept.split(":")[-1]):
                        canonical = mapped_canonical
                        break

            if not canonical:
                continue

            # Parse the numeric value (remove commas, handle negatives)
            try:
                cleaned = raw_value.replace(",", "").replace(" ", "")
                value = float(cleaned)
                if sign == "-":
                    value = -value
                # Keep the first (most recent) value found
                if canonical not in values:
                    values[canonical] = value
            except (ValueError, TypeError):
                continue

        return values
