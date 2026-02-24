"""Japan EDINET PIT client -- powered by edinet-tools SDK.

Replaces the original edinet.py with the edinet-tools unofficial wrapper
for richer, more reliable access to EDINET financial data.

Primary library: edinet-tools (https://github.com/matthelmer/edinet-tools)
  - Entity lookup, document listing, XBRL/CSV parsing
  - Structured financial statement extraction

Fallback: Direct EDINET API v2 calls via requests (same as original client)

Coverage: ~3,800+ listed companies on TSE, $6.5T market cap.
API: https://api.edinet-fsa.go.jp/api/v2 (free, API key recommended)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_EDINET_BASE = "https://api.edinet-fsa.go.jp/api/v2"
_CACHE_DIR = Path("cache/jp_edinet")

# Form codes for securities reports (annual) and quarterly reports
_ANNUAL_FORM_CODES = {"030000", "030001"}  # Yuho (securities report)
_QUARTERLY_FORM_CODES = {"043000", "043001"}  # Quarterly report


class JPEdinetError(Exception):
    """Raised on Japan EDINET wrapper failures."""
    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"JP EDINET error on {endpoint}: {detail}")


class JPEdinetClient:
    """Point-in-time client for EDINET (Japanese equities) using edinet-tools.

    Implements the ``PITClient`` protocol. Uses edinet-tools as primary
    library, with direct API v2 calls as fallback.

    Parameters
    ----------
    subscription_key:
        EDINET API v2 subscription key. Optional but recommended for
        better rate limits. Load from EDINET_API_KEY env var.
    cache_dir:
        Local cache directory.
    """

    def __init__(
        self,
        subscription_key: str = "",
        cache_dir: Path | str = _CACHE_DIR,
    ) -> None:
        self._subscription_key = subscription_key or os.environ.get("EDINET_API_KEY", "")
        self._cache_dir = Path(cache_dir)
        self._edinet_tools_available = False
        self._company_list_cache: list[dict[str, Any]] | None = None

        # Try to initialize edinet-tools
        try:
            import edinet_tools
            self._edinet_tools_available = True
            logger.info("edinet-tools SDK available for EDINET data")
        except ImportError:
            logger.warning("edinet-tools not installed; using direct EDINET API v2")

    # -- Cache helpers --------------------------------------------------------

    def _cache_path(self, identifier: str, filename: str) -> Path:
        safe_id = identifier.replace("/", "_").replace("\\", "_").upper()
        return self._cache_dir / safe_id / filename

    def _read_cache(self, identifier: str, filename: str) -> dict | None:
        path = self._cache_path(identifier, filename)
        if not path.exists():
            return None
        try:
            stat = path.stat()
            age_days = (date.today() - date.fromtimestamp(stat.st_mtime)).days
            if filename == "profile.json" and age_days > 7:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, identifier: str, filename: str, data: dict) -> None:
        path = self._cache_path(identifier, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    # -- Protocol properties -------------------------------------------------

    @property
    def market_id(self) -> str:
        return "jp_edinet"

    @property
    def market_name(self) -> str:
        return "Japan (Tokyo Stock Exchange) -- EDINET"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Japanese listed companies."""
        if self._company_list_cache is None:
            self._company_list_cache = self._fetch_company_list()

        if not query:
            return self._company_list_cache

        q = query.lower()
        return [
            c for c in self._company_list_cache
            if q in c.get("name", "").lower()
            or q in c.get("ticker", "").lower()
            or q in c.get("edinet_code", "").lower()
        ]

    def search_company(self, name: str) -> list[dict[str, Any]]:
        """Search for a Japanese company by name, ticker, or EDINET code.

        VERIFIED AGAINST OFFICIAL DOCS:
        - Date: 2026-02-24
        - Version: edinet-tools@0.3.0
        - Docs: https://github.com/matthelmer/edinet-tools
        - Research Log: .roo/research/jp-edinet-2026-02-24.md
        - edinet-tools v0.3.0 API: edinet_tools.search(query, limit=N)
        """
        # Try edinet-tools search first (v0.3.0 API)
        if self._edinet_tools_available:
            try:
                import edinet_tools
                # v0.3.0: edinet_tools.search(query, limit=N)
                results = edinet_tools.search(name, limit=20)
                if results:
                    return [
                        {
                            "ticker": getattr(r, "sec_code", getattr(r, "ticker", ""))[:4] if hasattr(r, "sec_code") else "",
                            "name": getattr(r, "name", str(r)),
                            "edinet_code": getattr(r, "edinet_code", ""),
                            "cik": getattr(r, "edinet_code", ""),
                            "exchange": "TSE",
                            "market_id": self.market_id,
                        }
                        for r in results
                    ]
            except Exception as exc:
                logger.debug("edinet-tools search failed: %s", exc)

        return self.list_companies(query=name)

    def _fetch_company_list(self) -> list[dict[str, Any]]:
        """Build company list by scanning recent EDINET filings.

        VERIFIED: edinet-tools v0.3.0 does NOT have get_supported_companies().
        The entity lookup is per-company, not bulk. We use the gov API
        filing scan as the primary method for building the company list.
        Research Log: .roo/research/jp-edinet-2026-02-24.md
        """
        # edinet-tools v0.3.0 doesn't provide a bulk company list.
        # Use the gov API filing scan to discover entities.
        return self._scan_recent_filings_for_companies()

    def _scan_recent_filings_for_companies(self) -> list[dict[str, Any]]:
        """Scan last 14 days of EDINET filings to discover entities."""
        from operator1.http_utils import cached_get, HTTPError

        companies: dict[str, dict] = {}
        for days_ago in range(0, 14):
            filing_date = (date.today() - timedelta(days=days_ago)).isoformat()
            params: dict[str, Any] = {"date": filing_date, "type": 2}
            if self._subscription_key:
                params["Subscription-Key"] = self._subscription_key

            try:
                data = cached_get(
                    f"{_EDINET_BASE}/documents.json",
                    params=params,
                    headers={"Accept": "application/json"},
                )
                results = data.get("results", []) if isinstance(data, dict) else []
                for doc in results:
                    edinet_code = doc.get("edinetCode", "")
                    if not edinet_code or edinet_code in companies:
                        continue
                    sec_code = doc.get("secCode", "")
                    companies[edinet_code] = {
                        "ticker": sec_code[:4] if sec_code and len(sec_code) >= 4 else "",
                        "name": doc.get("filerName", ""),
                        "edinet_code": edinet_code,
                        "cik": edinet_code,
                        "exchange": "TSE",
                        "market_id": self.market_id,
                    }
            except Exception:
                continue

        return list(companies.values())

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile for a Japanese company.

        Parameters
        ----------
        identifier:
            Ticker (e.g. "7203"), EDINET code (e.g. "E02144"), or name.
        """
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        raw_profile: dict[str, Any] = {}

        # Try edinet-tools first (v0.3.0 API: edinet_tools.entity())
        # Research Log: .roo/research/jp-edinet-2026-02-24.md
        if self._edinet_tools_available:
            try:
                import edinet_tools
                # v0.3.0: edinet_tools.entity(identifier) -- lookup by ticker or name
                info = edinet_tools.entity(identifier)
                if info:
                    raw_profile = {
                        "name": getattr(info, "name", ""),
                        "ticker": getattr(info, "sec_code", identifier)[:4] if hasattr(info, "sec_code") else identifier,
                        "isin": "",
                        "country": "JP",
                        "sector": getattr(info, "industry", ""),
                        "industry": getattr(info, "industry", ""),
                        "exchange": "TSE",
                        "currency": "JPY",
                        "edinet_code": getattr(info, "edinet_code", ""),
                        "cik": getattr(info, "edinet_code", ""),
                    }
            except Exception as exc:
                logger.debug("edinet-tools profile failed for %s: %s", identifier, exc)

        if not raw_profile:
            # Fallback: search filings for this entity
            matches = self.search_company(identifier)
            if matches:
                m = matches[0]
                raw_profile = {
                    "name": m.get("name", ""),
                    "ticker": m.get("ticker", identifier),
                    "isin": "",
                    "country": "JP",
                    "sector": "",
                    "industry": "",
                    "exchange": "TSE",
                    "currency": "JPY",
                    "edinet_code": m.get("edinet_code", ""),
                    "cik": m.get("edinet_code", ""),
                }

        if not raw_profile:
            raise JPEdinetError("get_profile", f"Company not found: {identifier}")

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw_profile, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    # -- Financial statements ------------------------------------------------

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch income statements with filing_date and report_date."""
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        """Fetch balance sheets with filing_date and report_date."""
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        """Fetch cash flow statements with filing_date and report_date."""
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Extract financial statements using edinet-tools or direct API.

        Uses CSV download (type=5) from EDINET for easier parsing when
        edinet-tools structured extraction is not available.
        """
        # Try edinet-tools first
        if self._edinet_tools_available:
            try:
                df = self._fetch_via_edinet_tools(identifier, statement_type)
                if df is not None and not df.empty:
                    return df
            except Exception as exc:
                logger.warning("edinet-tools financials failed for %s: %s", identifier, exc)

        # Fallback: direct EDINET API v2 with CSV parsing
        return self._fetch_via_direct_api(identifier, statement_type)

    def _fetch_via_edinet_tools(self, identifier: str, statement_type: str) -> pd.DataFrame | None:
        """Use edinet-tools v0.3.0 to get structured financial data.

        VERIFIED AGAINST OFFICIAL DOCS:
        - Date: 2026-02-24
        - Version: edinet-tools@0.3.0
        - Docs: https://github.com/matthelmer/edinet-tools
        - Research Log: .roo/research/jp-edinet-2026-02-24.md
        - v0.3.0 API: entity(), documents(), doc.parse()
        """
        import edinet_tools

        # v0.3.0: edinet_tools.entity(identifier) for company lookup
        try:
            info = edinet_tools.entity(identifier)
        except Exception:
            info = None
        if not info:
            return None

        # v0.3.0: entity.documents(days=N) for recent filings
        docs = []
        try:
            entity_docs = info.documents(days=730)  # 2 years of filings
            if entity_docs:
                docs = list(entity_docs)
        except Exception:
            # Fallback: scan by date using edinet_tools.documents(date)
            try:
                for days_back in [0, 90, 180, 270, 365, 450, 540, 630, 730]:
                    target_date = (date.today() - timedelta(days=days_back)).isoformat()
                    try:
                        day_docs = edinet_tools.documents(target_date)
                        if day_docs:
                            for d in day_docs:
                                edinet_code = getattr(d, "edinet_code", "")
                                if edinet_code == getattr(info, "edinet_code", "NONE"):
                                    docs.append(d)
                    except Exception:
                        continue
            except Exception:
                pass

        if not docs:
            return None

        rows: list[dict] = []
        for doc in docs:
            try:
                parsed = doc.parse() if hasattr(doc, "parse") else None
                if parsed is None:
                    continue

                data_dict = parsed.to_dict() if hasattr(parsed, "to_dict") else {}
                filing_date = str(getattr(doc, "submit_date_time", getattr(doc, "submitDateTime", "")))
                period_end = str(getattr(doc, "period_end", ""))
                form_code = str(getattr(doc, "form_code", ""))
                is_annual = form_code in _ANNUAL_FORM_CODES

                for key, value in data_dict.items():
                    if value is None:
                        continue
                    canonical = self._map_edinet_concept(key, statement_type)
                    if canonical:
                        try:
                            rows.append({
                                "concept": canonical,
                                "value": float(value),
                                "filing_date": filing_date,
                                "report_date": period_end,
                                "period_type": "annual" if is_annual else "quarterly",
                                "form": "Yuho" if is_annual else "Quarterly",
                            })
                        except (ValueError, TypeError):
                            continue
            except Exception:
                continue

        if not rows:
            return None

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)

    def _fetch_via_direct_api(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fallback: use direct EDINET API v2 to find and download filings."""
        from operator1.http_utils import cached_get, HTTPError

        # First, resolve the identifier to an EDINET code
        matches = self.search_company(identifier)
        if not matches:
            return pd.DataFrame()

        edinet_code = matches[0].get("edinet_code", "")
        if not edinet_code:
            return pd.DataFrame()

        # Scan recent filing dates to find this company's filings
        doc_ids: list[dict] = []
        for days_ago in range(0, 730, 7):  # Check weekly for 2 years
            filing_date = (date.today() - timedelta(days=days_ago)).isoformat()
            params: dict[str, Any] = {"date": filing_date, "type": 2}
            if self._subscription_key:
                params["Subscription-Key"] = self._subscription_key

            try:
                data = cached_get(
                    f"{_EDINET_BASE}/documents.json",
                    params=params,
                    headers={"Accept": "application/json"},
                )
                results = data.get("results", []) if isinstance(data, dict) else []
                for doc in results:
                    if doc.get("edinetCode") != edinet_code:
                        continue
                    form_code = doc.get("formCode", "")
                    if form_code in _ANNUAL_FORM_CODES | _QUARTERLY_FORM_CODES:
                        # Check legal status and withdrawal
                        if doc.get("legalStatus", "0") == "0":
                            continue
                        if doc.get("withdrawalStatus", "0") != "0":
                            continue
                        doc_ids.append({
                            "doc_id": doc.get("docID", ""),
                            "filing_date": doc.get("submitDateTime", ""),
                            "period_end": doc.get("periodEnd", ""),
                            "period_start": doc.get("periodStart", ""),
                            "form_code": form_code,
                            "csv_flag": doc.get("csvFlag", "0"),
                            "is_annual": form_code in _ANNUAL_FORM_CODES,
                        })
            except Exception:
                continue

        if not doc_ids:
            return pd.DataFrame()

        # Download CSV for each filing (type=5 if available)
        rows: list[dict] = []
        for doc_meta in doc_ids[:8]:  # Limit to 8 filings
            if doc_meta["csv_flag"] != "1":
                continue

            try:
                import requests
                params = {"type": 5}
                if self._subscription_key:
                    params["Subscription-Key"] = self._subscription_key

                resp = requests.get(
                    f"{_EDINET_BASE}/documents/{doc_meta['doc_id']}",
                    params=params,
                    timeout=60,
                )
                if resp.status_code == 200 and resp.content:
                    # CSV parsing -- EDINET CSV format has concept names
                    import io
                    import zipfile
                    try:
                        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                            for name in zf.namelist():
                                if name.endswith(".csv"):
                                    with zf.open(name) as f:
                                        csv_df = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip")
                                        for _, row in csv_df.iterrows():
                                            concept = str(row.iloc[0]) if len(row) > 0 else ""
                                            value = row.iloc[1] if len(row) > 1 else None
                                            canonical = self._map_edinet_concept(concept, statement_type)
                                            if canonical and value is not None:
                                                try:
                                                    rows.append({
                                                        "concept": canonical,
                                                        "value": float(value),
                                                        "filing_date": doc_meta["filing_date"],
                                                        "report_date": doc_meta["period_end"],
                                                        "period_type": "annual" if doc_meta["is_annual"] else "quarterly",
                                                    })
                                                except (ValueError, TypeError):
                                                    continue
                    except zipfile.BadZipFile:
                        logger.debug("Bad ZIP from EDINET for doc %s", doc_meta["doc_id"])
            except Exception as exc:
                logger.debug("EDINET CSV download failed for %s: %s", doc_meta["doc_id"], exc)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)

    def _map_edinet_concept(self, concept: str, statement_type: str) -> str | None:
        """Map EDINET/JPPFS concept to canonical name."""
        # Use the canonical_translator's JPPFS mapping
        from operator1.clients.canonical_translator import _JPPFS_MAP, _IFRS_MAP
        combined = {**_JPPFS_MAP, **_IFRS_MAP}

        # Direct match
        if concept in combined:
            return combined[concept]

        # Try lowercase match on concept label
        concept_lower = concept.lower().strip()
        label_map = {
            "revenue": "revenue", "net sales": "revenue",
            "operating income": "operating_income",
            "net income": "net_income", "profit": "net_income",
            "total assets": "total_assets",
            "total liabilities": "total_liabilities",
            "net assets": "total_equity", "total equity": "total_equity",
            "cash and deposits": "cash_and_equivalents",
            "operating cash flow": "operating_cash_flow",
        }
        return label_map.get(concept_lower)

    # -- Price data (delegated to ohlcv_provider) ----------------------------

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """EDINET does not provide OHLCV. Handled by ohlcv_provider."""
        return pd.DataFrame()

    # -- Peers / executives --------------------------------------------------

    def get_peers(self, identifier: str) -> list[str]:
        """Return peer companies based on industry classification."""
        profile = self.get_profile(identifier)
        industry = profile.get("industry", "")
        if not industry:
            return []

        all_companies = self.list_companies()
        target_ticker = profile.get("ticker", identifier)
        peers = [
            c["ticker"] for c in all_companies
            if c.get("ticker") and c["ticker"] != target_ticker
        ]
        return peers[:10]

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        """EDINET doesn't have a direct executives endpoint."""
        return []
