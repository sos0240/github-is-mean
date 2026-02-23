"""India BSE PIT client -- uses BSE India API.

Primary: BSE India API (https://api.bseindia.com/BseIndiaAPI/api)
Coverage: ~5,500+ listed companies on BSE, ~$4T market cap.
"""
from __future__ import annotations
import json, logging, os
from datetime import date
from pathlib import Path
from typing import Any
import pandas as pd
from operator1.http_utils import cached_get, HTTPError

logger = logging.getLogger(__name__)
_BSE_BASE = "https://api.bseindia.com/BseIndiaAPI/api"
_CACHE_DIR = Path("cache/in_bse")

class INBseClient:
    """PIT client for Indian BSE equities."""
    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._headers = {"Accept": "application/json", "User-Agent": "Operator1/1.0",
                         "Referer": "https://www.bseindia.com/"}

    def _cache_path(self, identifier: str, fn: str) -> Path:
        return self._cache_dir / identifier.upper() / fn
    def _read_cache(self, identifier: str, fn: str) -> dict | None:
        p = self._cache_path(identifier, fn)
        if not p.exists(): return None
        try:
            if fn == "profile.json" and (date.today() - date.fromtimestamp(p.stat().st_mtime)).days > 7: return None
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception: return None
    def _write_cache(self, identifier: str, fn: str, data: dict) -> None:
        p = self._cache_path(identifier, fn)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    @property
    def market_id(self) -> str: return "in_bse"
    @property
    def market_name(self) -> str: return "India (BSE) -- BSE India"

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        if not query: return []
        try:
            data = cached_get(f"{_BSE_BASE}/Suggest/Getstockdata/{query}", headers=self._headers)
            items = data if isinstance(data, list) else []
            return [{"ticker": i.get("scrip_cd", ""), "name": i.get("scripname", ""),
                     "cik": i.get("scrip_cd", ""), "exchange": "BSE", "country": "IN",
                     "market_id": self.market_id} for i in items]
        except Exception: return []

    def search_company(self, name: str) -> list[dict[str, Any]]: return self.list_companies(query=name)

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached: return cached
        try:
            data = cached_get(f"{_BSE_BASE}/StockReachGraph/stockData/{identifier}", headers=self._headers)
            raw = {"name": data.get("comp_name", ""), "ticker": identifier, "isin": data.get("isin_code", ""),
                   "country": "IN", "sector": data.get("sector_name", ""), "industry": data.get("industry", ""),
                   "exchange": "BSE", "currency": "INR", "cik": identifier}
        except Exception:
            raw = {"name": "", "ticker": identifier, "isin": "", "country": "IN",
                   "sector": "", "industry": "", "exchange": "BSE", "currency": "INR", "cik": identifier}
        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def get_income_statement(self, identifier: str) -> pd.DataFrame: return pd.DataFrame()
    def get_balance_sheet(self, identifier: str) -> pd.DataFrame: return pd.DataFrame()
    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame: return pd.DataFrame()
    def get_quotes(self, identifier: str) -> pd.DataFrame: return pd.DataFrame()
    def get_peers(self, identifier: str) -> list[str]: return []
    def get_executives(self, identifier: str) -> list[dict[str, Any]]: return []
