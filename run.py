#!/usr/bin/env python3
"""Operator 1 -- Interactive Terminal Launcher.

A user-friendly interface that guides you through running the
financial analysis pipeline step by step.

Flow:
  1. System checks (Python version, internet)
  2. Dependency check
  3. LLM provider & model selection (always prompt first)
  4. Data source mode: wrappers-only vs API + wrappers
  5. If API + wrappers: prompt for market-specific API keys
  6. Company + country input
  7. LLM resolves the right market/client for the company
  8. Pipeline options + confirmation
  9. Run pipeline

Usage:
    python run.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import socket
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _color(text: str, code: str) -> str:
    """Wrap text in ANSI color codes (no-op on Windows without colorama)."""
    if os.name == "nt":
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(t: str) -> str:
    return _color(t, "32")


def _yellow(t: str) -> str:
    return _color(t, "33")


def _red(t: str) -> str:
    return _color(t, "31")


def _cyan(t: str) -> str:
    return _color(t, "36")


def _bold(t: str) -> str:
    return _color(t, "1")


def _dim(t: str) -> str:
    return _color(t, "2")


def _banner():
    print("")
    print(_bold(_cyan("  ================================================================")))
    print(_bold(_cyan("     OPERATOR 1 -- Point-in-Time Financial Analysis")))
    print(_bold(_cyan("  ================================================================")))
    print(_dim("  Institutional-grade equity research from free government filings"))
    print(_dim("  10 markets | $91T+ coverage | 25+ math models | PIT-audited data"))
    print("")


def _separator():
    print(_dim("  " + "-" * 60))


def _step(num: int, title: str):
    print("")
    print(_bold(f"  [{num}] {title}"))
    print("")


def _ok(msg: str):
    print(f"  {_green('[OK]')} {msg}")


def _warn(msg: str):
    print(f"  {_yellow('[!]')} {msg}")


def _err(msg: str):
    print(f"  {_red('[ERROR]')} {msg}")


def _info(msg: str):
    print(f"  {_dim('[i]')} {msg}")


def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"  > {msg}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("")
        sys.exit(0)
    return value if value else default


def _yes_no(msg: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        value = input(f"  > {msg} {suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("")
        sys.exit(0)
    if not value:
        return default
    return value in ("y", "yes")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_python_version() -> bool:
    v = sys.version_info
    if v.major >= 3 and v.minor >= 10:
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    elif v.major >= 3 and v.minor >= 8:
        _warn(f"Python {v.major}.{v.minor}.{v.micro} (3.10+ recommended)")
        return True
    else:
        _err(f"Python {v.major}.{v.minor}.{v.micro} -- need 3.8+")
        return False


def check_internet() -> bool:
    """Check internet connectivity."""
    hosts = [
        ("data.sec.gov", 443),
        ("api.stlouisfed.org", 443),
        ("1.1.1.1", 53),
    ]
    for host, port in hosts:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            _ok(f"Internet connection active (reached {host})")
            return True
        except (socket.timeout, socket.error, OSError):
            continue
    _err("No internet connection detected")
    _info("This pipeline requires internet to fetch financial data.")
    return False


def check_dependencies() -> tuple[bool, list[str]]:
    """Check which key Python packages are installed."""
    required = {
        "requests": "HTTP requests",
        "pandas": "Data processing",
        "numpy": "Numerical computing",
        "yaml": "Config loading (pyyaml)",
    }
    optional = {
        "statsmodels": "Kalman filter, VAR models",
        "arch": "GARCH volatility models",
        "sklearn": "Tree ensembles, imputation",
        "torch": "LSTM deep learning",
        "hmmlearn": "Hidden Markov Models",
        "ruptures": "Structural break detection",
        "xgboost": "XGBoost tree ensemble",
        "matplotlib": "Chart generation",
        "dotenv": "python-dotenv (.env loading)",
    }

    missing_required: list[str] = []

    for pkg, desc in required.items():
        try:
            __import__(pkg)
            _ok(f"{pkg} -- {desc}")
        except ImportError:
            _err(f"{pkg} -- {desc} [MISSING]")
            missing_required.append(pkg)

    for pkg, desc in optional.items():
        try:
            __import__(pkg)
            _ok(f"{pkg} -- {desc}")
        except ImportError:
            _warn(f"{pkg} -- {desc} [not installed, some features limited]")

    return len(missing_required) == 0, missing_required


def _mask_key(key: str) -> str:
    """Show first 4 and last 4 chars of a key, mask the rest."""
    if len(key) <= 8:
        return key[:2] + "..." + key[-2:]
    return key[:4] + "..." + key[-4:]


# ---------------------------------------------------------------------------
# LLM key setup (always prompt first)
# ---------------------------------------------------------------------------

def _select_llm_model(provider: str) -> str:
    """Show available models for the chosen provider and let the user pick.

    Returns the model name string, or empty string to use the default.
    """
    try:
        from operator1.clients.llm_factory import get_available_models
        models = get_available_models(provider)
    except Exception:
        return ""

    if not models:
        return ""

    print("")
    print(_bold(f"  Available {provider.title()} models:"))
    print("")
    for idx, m in enumerate(models, 1):
        ctx = m["context_window"]
        out = m["max_output_tokens"]
        tier = m["tier"]
        # Human-readable context/output sizes
        ctx_str = f"{ctx // 1_000_000}M" if ctx >= 1_000_000 else f"{ctx // 1_000}K"
        out_str = f"{out // 1_000}K" if out >= 1_000 else str(out)
        default_marker = " (default)" if idx == 1 else ""
        print(f"    {_bold(str(idx))}. {m['name']}")
        print(f"       {_dim(f'{tier} | {ctx_str} context | {out_str} output')}{_green(default_marker)}")
    print("")

    choice = _prompt(f"Choose model (1-{len(models)})", "1")
    try:
        sel = int(choice) - 1
        if 0 <= sel < len(models):
            selected = models[sel]["name"]
            _ok(f"Model: {selected}")
            return selected
    except ValueError:
        # Try matching by name
        for m in models:
            if choice.lower() in m["name"].lower():
                _ok(f"Model: {m['name']}")
                return m["name"]

    # Default: first model
    _ok(f"Model: {models[0]['name']} (default)")
    return models[0]["name"]


def setup_llm_keys() -> dict[str, str]:
    """Prompt for LLM API keys. Always runs at the start of the flow.

    Returns a dict of all loaded keys (LLM + any from .env).
    """
    keys: dict[str, str] = {}
    env_path = Path(__file__).resolve().parent / ".env"

    # Load existing keys from .env if present
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'").strip()
                if k and v and not v.startswith("your_"):
                    keys[k] = v

    # Also pull from environment variables
    for key_name in ["GEMINI_API_KEY", "ANTHROPIC_API_KEY"]:
        if key_name not in keys:
            env_val = os.environ.get(key_name)
            if env_val and env_val.strip():
                keys[key_name] = env_val.strip()

    has_gemini = "GEMINI_API_KEY" in keys
    has_claude = "ANTHROPIC_API_KEY" in keys

    if has_gemini:
        _ok(f"GEMINI_API_KEY: {_mask_key(keys['GEMINI_API_KEY'])} (loaded from .env)")
    if has_claude:
        _ok(f"ANTHROPIC_API_KEY: {_mask_key(keys['ANTHROPIC_API_KEY'])} (loaded from .env)")

    if has_gemini and has_claude:
        _ok("Both LLM providers available")
    elif has_gemini or has_claude:
        _ok("LLM provider available")
    else:
        # No LLM keys found -- prompt the user
        print(_dim("  An LLM API key is required for smart market routing and"))
        print(_dim("  AI-generated report narratives."))
        print("")
        print(f"    {_bold('1')}. Google Gemini  -- https://aistudio.google.com/apikey")
        print(f"    {_bold('2')}. Anthropic Claude -- https://console.anthropic.com/")
        print(f"    {_bold('3')}. Skip (limited functionality, template reports only)")
        print("")

        llm_choice = _prompt("Choose LLM provider (1/2/3)", "1")
        if llm_choice == "1":
            value = _prompt("Enter GEMINI_API_KEY")
            if value:
                keys["GEMINI_API_KEY"] = value.strip()
                _save_key_to_env(env_path, "GEMINI_API_KEY", value.strip())
                _ok("GEMINI_API_KEY saved")
        elif llm_choice == "2":
            value = _prompt("Enter ANTHROPIC_API_KEY")
            if value:
                keys["ANTHROPIC_API_KEY"] = value.strip()
                _save_key_to_env(env_path, "ANTHROPIC_API_KEY", value.strip())
                _ok("ANTHROPIC_API_KEY saved")
        else:
            _warn("Skipping LLM setup. Smart routing disabled, template reports only.")

    # Determine which LLM provider to use
    has_gemini = "GEMINI_API_KEY" in keys
    has_claude = "ANTHROPIC_API_KEY" in keys
    llm_provider = ""

    if has_gemini and has_claude:
        print("")
        print(_bold("  Which LLM provider to use for this session?"))
        print(f"    {_bold('1')}. Google Gemini")
        print(f"    {_bold('2')}. Anthropic Claude")
        print("")
        prov = _prompt("Choose (1/2)", "1")
        llm_provider = "claude" if prov == "2" else "gemini"
    elif has_claude:
        llm_provider = "claude"
    elif has_gemini:
        llm_provider = "gemini"

    # --- Model selection ---
    llm_model = ""
    if llm_provider:
        _ok(f"LLM provider: {llm_provider}")
        os.environ["LLM_PROVIDER"] = llm_provider
        keys["_llm_provider"] = llm_provider

        llm_model = _select_llm_model(llm_provider)
        if llm_model:
            os.environ["LLM_MODEL"] = llm_model
            keys["_llm_model"] = llm_model

    # Set keys in environment
    for k, v in keys.items():
        if not k.startswith("_"):
            os.environ[k] = v

    return keys


def _save_key_to_env(env_path: Path, key_name: str, value: str) -> None:
    """Append a key to the .env file."""
    try:
        with open(env_path, "a") as f:
            f.write(f"\n{key_name}={value}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data source mode selection
# ---------------------------------------------------------------------------

# Country -> market_id mapping for LLM routing
_COUNTRY_TO_MARKET: dict[str, str] = {
    "us": "us_sec_edgar",
    "united states": "us_sec_edgar",
    "usa": "us_sec_edgar",
    "america": "us_sec_edgar",
    "uk": "uk_companies_house",
    "united kingdom": "uk_companies_house",
    "britain": "uk_companies_house",
    "england": "uk_companies_house",
    "japan": "jp_edinet",
    "jp": "jp_edinet",
    "south korea": "kr_dart",
    "korea": "kr_dart",
    "kr": "kr_dart",
    "taiwan": "tw_mops",
    "tw": "tw_mops",
    "brazil": "br_cvm",
    "br": "br_cvm",
    "chile": "cl_cmf",
    "cl": "cl_cmf",
    "germany": "de_esef",
    "de": "de_esef",
    "france": "fr_esef",
    "fr": "fr_esef",
    "eu": "eu_esef",
    "europe": "eu_esef",
    "australia": "au_asx",
    "au": "au_asx",
    "canada": "ca_sedar",
    "ca": "ca_sedar",
    "china": "cn_sse",
    "cn": "cn_sse",
    "hong kong": "hk_hkex",
    "hk": "hk_hkex",
    "india": "in_bse",
    "in": "in_bse",
    "singapore": "sg_sgx",
    "sg": "sg_sgx",
    "mexico": "mx_bmv",
    "mx": "mx_bmv",
    "south africa": "za_jse",
    "za": "za_jse",
    "switzerland": "ch_six",
    "ch": "ch_six",
    "saudi arabia": "sa_tadawul",
    "sa": "sa_tadawul",
    "uae": "ae_dfm",
    "ae": "ae_dfm",
}

# Well-known companies to market mappings for fallback
_KNOWN_COMPANIES: dict[str, str] = {
    "aapl": "us_sec_edgar", "apple": "us_sec_edgar",
    "msft": "us_sec_edgar", "microsoft": "us_sec_edgar",
    "googl": "us_sec_edgar", "google": "us_sec_edgar", "alphabet": "us_sec_edgar",
    "amzn": "us_sec_edgar", "amazon": "us_sec_edgar",
    "tsla": "us_sec_edgar", "tesla": "us_sec_edgar",
    "nvda": "us_sec_edgar", "nvidia": "us_sec_edgar",
    "meta": "us_sec_edgar",
    "7203": "jp_edinet", "toyota": "jp_edinet",
    "9984": "jp_edinet", "softbank": "jp_edinet",
    "005930": "kr_dart", "samsung": "kr_dart",
    "2330": "tw_mops", "tsmc": "tw_mops",
    "vale3": "br_cvm", "vale": "br_cvm",
    "petrobras": "br_cvm", "petr4": "br_cvm",
    "bp": "uk_companies_house", "hsbc": "uk_companies_house",
    "shell": "uk_companies_house", "shel": "uk_companies_house",
    "sap": "de_esef", "siemens": "de_esef",
    "lvmh": "fr_esef", "totalenergies": "fr_esef",
}


def choose_data_source_mode() -> str:
    """Let the user choose between wrappers-only or API + wrappers.

    Returns 'wrappers' or 'api_and_wrappers'.
    """
    print(_bold("  Choose data source mode:"))
    print("")
    print(f"    {_bold('1')}. {_green('Wrappers only')} (recommended)")
    print("       Uses unofficial wrapper libraries (edgartools, dart-fss, etc.)")
    print(f"       {_dim('No extra API keys needed. Simplest setup.')}")
    print("")
    print(f"    {_bold('2')}. {_yellow('API + Wrappers together')}")
    print("       Uses raw API endpoints alongside wrappers for richer data")
    print(f"       {_dim('May require market-specific API keys (all free).')}")
    print("")

    choice = _prompt("Select mode (1/2)", "1")
    if choice == "2":
        _ok("Mode: API + Wrappers (enhanced data)")
        return "api_and_wrappers"
    else:
        _ok("Mode: Wrappers only (recommended)")
        return "wrappers"


def setup_market_api_keys(keys: dict[str, str]) -> dict[str, str]:
    """Prompt for market-specific API keys when using API + wrappers mode."""
    env_path = Path(__file__).resolve().parent / ".env"

    market_keys = [
        ("COMPANIES_HOUSE_API_KEY", "UK Companies House", "https://developer.company-information.service.gov.uk/"),
        ("DART_API_KEY", "South Korea DART", "https://opendart.fss.or.kr/"),
        ("FRED_API_KEY", "US FRED (macro data)", "https://fred.stlouisfed.org/docs/api/api_key.html"),
        ("ESTAT_API_KEY", "Japan e-Stat", "https://www.e-stat.go.jp/en"),
        ("KOSIS_API_KEY", "South Korea KOSIS", "https://kosis.kr/openapi/"),
        ("INSEE_API_KEY", "France INSEE", "https://api.insee.fr/"),
        ("BCCH_API_KEY", "Chile BCCh", "https://si3.bcentral.cl/"),
        ("ALPHA_VANTAGE_API_KEY", "Alpha Vantage (OHLCV)", "https://www.alphavantage.co/support/#api-key"),
    ]

    print(_dim("  Market-specific API keys (all free registration):"))
    print("")

    for key_name, desc, url in market_keys:
        if key_name in keys:
            _ok(f"{key_name}: {_mask_key(keys[key_name])} ({desc})")
        else:
            _info(f"{key_name}: not set -- {desc}")
            _dim(f"       Register: {url}")

    print("")
    if _yes_no("Enter any market API keys now?", default=False):
        for key_name, desc, url in market_keys:
            if key_name not in keys:
                value = _prompt(f"{desc} key (or press Enter to skip)")
                if value:
                    keys[key_name] = value.strip()
                    os.environ[key_name] = value.strip()
                    _save_key_to_env(env_path, key_name, value.strip())
                    _ok(f"Saved {key_name}")

    return keys


# ---------------------------------------------------------------------------
# LLM-driven market routing
# ---------------------------------------------------------------------------

def _resolve_market_with_llm(
    company: str,
    country: str,
    llm_provider: str,
    keys: dict[str, str],
) -> str | None:
    """Use the configured LLM to determine the right market for a company.

    Returns a market_id string or None if the LLM can't determine it.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from operator1.clients.pit_registry import MARKETS

    market_list = "\n".join(
        f"- {mid}: {m.country} ({m.exchange}) -- {m.pit_api_name}"
        for mid, m in sorted(MARKETS.items())
    )

    prompt = (
        f"Given the following company and country, determine which market/exchange "
        f"this company is listed on and return ONLY the market_id from the list below.\n\n"
        f"Company: {company}\n"
        f"Country: {country}\n\n"
        f"Available markets:\n{market_list}\n\n"
        f"Return ONLY the market_id (e.g. 'us_sec_edgar'). No explanation."
    )

    try:
        if llm_provider == "gemini" and keys.get("GEMINI_API_KEY"):
            from operator1.clients.gemini import GeminiClient
            client = GeminiClient(api_key=keys["GEMINI_API_KEY"])
            response = client.generate(prompt)
        elif llm_provider == "claude" and keys.get("ANTHROPIC_API_KEY"):
            from operator1.clients.claude import ClaudeClient
            client = ClaudeClient(api_key=keys["ANTHROPIC_API_KEY"])
            response = client.generate(prompt)
        else:
            return None

        if response:
            # Extract market_id from the response
            candidate = response.strip().lower().replace("'", "").replace('"', '')
            if candidate in MARKETS:
                return candidate
            # Try to find it in the response text
            for mid in MARKETS:
                if mid in response.lower():
                    return mid
    except Exception as exc:
        _warn(f"LLM routing failed: {exc}")

    return None


def resolve_market(company: str, country: str, keys: dict[str, str]) -> str:
    """Determine the right market for a company using multiple strategies.

    Strategy:
    1. Try direct country lookup
    2. Try well-known company lookup
    3. Try LLM-based routing (if LLM key available)
    4. Fall back to manual region/market selection
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from operator1.clients.pit_registry import MARKETS

    # Strategy 1: Country lookup
    country_lower = country.lower().strip()
    if country_lower in _COUNTRY_TO_MARKET:
        market_id = _COUNTRY_TO_MARKET[country_lower]
        if market_id in MARKETS:
            market = MARKETS[market_id]
            _ok(f"Market resolved from country: {market.country} ({market.pit_api_name})")
            return market_id

    # Strategy 2: Well-known company lookup
    company_lower = company.lower().strip()
    if company_lower in _KNOWN_COMPANIES:
        market_id = _KNOWN_COMPANIES[company_lower]
        if market_id in MARKETS:
            market = MARKETS[market_id]
            _ok(f"Market resolved from company: {market.country} ({market.pit_api_name})")
            return market_id

    # Strategy 3: LLM routing
    llm_provider = keys.get("_llm_provider", "")
    if llm_provider:
        _info("Using LLM to determine the right market...")
        market_id = _resolve_market_with_llm(company, country, llm_provider, keys)
        if market_id and market_id in MARKETS:
            market = MARKETS[market_id]
            _ok(f"Market resolved by LLM: {market.country} ({market.pit_api_name})")
            return market_id
        _warn("LLM could not determine the market. Falling back to manual selection.")

    # Strategy 4: Manual fallback
    _info("Could not auto-detect market. Please select manually:")
    return _manual_market_selection()


def _manual_market_selection() -> str:
    """Fall back to the traditional region -> market selection."""
    from operator1.clients.pit_registry import get_regions, get_markets_by_region

    regions = get_regions()
    print("")
    print(_bold("  Choose a region:"))
    print("")
    for i, region in enumerate(regions, 1):
        markets = get_markets_by_region(region)
        countries = ", ".join(m.country for m in markets)
        print(f"    {_bold(str(i))}. {region}")
        print(f"       {_dim(countries)}")
        print("")

    choice = _prompt("Select region (1-{})".format(len(regions)), "1")
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(regions):
            region = regions[idx]
        else:
            region = regions[0]
    except ValueError:
        region = regions[0]

    markets = get_markets_by_region(region)
    print("")
    print(_bold(f"  Markets in {region}:"))
    print("")
    for i, m in enumerate(markets, 1):
        key_note = _dim(" (API key required)") if m.requires_api_key else _dim(" (no key needed)")
        print(f"    {_bold(str(i))}. {m.country} -- {m.exchange}")
        print(f"       Data source: {m.pit_api_name}{key_note}")
        print("")

    choice = _prompt("Select market (1-{})".format(len(markets)), "1")
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(markets):
            return markets[idx].market_id
    except ValueError:
        pass

    return markets[0].market_id


# ---------------------------------------------------------------------------
# Pipeline options
# ---------------------------------------------------------------------------

def _check_user_input_pii(company: str, country: str, keys: dict[str, str]) -> None:
    """Use LLM + regex to check if user input contains personal data."""
    try:
        from operator1.clients.personal_data_guard import (
            check_user_input_for_pii,
            format_pii_warning,
        )

        # Build a lightweight LLM client for the check
        llm_client = None
        llm_provider = keys.get("_llm_provider", "")
        if llm_provider == "gemini" and keys.get("GEMINI_API_KEY"):
            from operator1.clients.gemini import GeminiClient
            llm_client = GeminiClient(api_key=keys["GEMINI_API_KEY"])
        elif llm_provider == "claude" and keys.get("ANTHROPIC_API_KEY"):
            from operator1.clients.claude import ClaudeClient
            llm_client = ClaudeClient(api_key=keys["ANTHROPIC_API_KEY"])

        result = check_user_input_for_pii(company, country, llm_client)
        if result.has_personal_data:
            warning_text = format_pii_warning(result)
            print("")
            print(_red("  " + "-" * 56))
            for line in warning_text.splitlines():
                print(_red(f"  {line}"))
            print(_red("  " + "-" * 56))
            print("")
            if not _yes_no("Your input may contain personal data. Continue anyway?", default=False):
                print(_dim("  Cancelled. Please re-enter with a company name or ticker."))
                sys.exit(0)
    except Exception:
        pass  # Non-blocking -- never prevent the pipeline from running


def _check_market_pii(market_id: str, market) -> None:
    """Warn the user if the resolved market's API registration requires personal data."""
    try:
        from operator1.clients.personal_data_guard import (
            check_wrapper_personal_data,
            format_pii_warning,
        )

        result = check_wrapper_personal_data(market_id)
        if result.has_personal_data:
            warning_text = format_pii_warning(result)
            print("")
            print(_yellow("  " + "-" * 56))
            print(_yellow("  PRIVACY NOTICE: API Registration Requirements"))
            print(_yellow("  " + "-" * 56))
            for line in warning_text.splitlines():
                print(_yellow(f"  {line}"))
            print("")
            _info(f"Market: {market.country} ({market.pit_api_name})")
            if hasattr(market, "input_requirements") and market.input_requirements:
                _info(f"Registration requires: {market.input_requirements}")
            print("")
            if not _yes_no("This market requires personal data for API registration. Continue?", default=True):
                _info("You can choose a different market or use wrappers-only mode.")
                sys.exit(0)
        elif result.market_personal_data_level == "low":
            _info(f"Note: {market.pit_api_name} requires basic registration ({result.details})")
    except Exception:
        pass  # Non-blocking


def estimate_runtime(skip_linked: bool, skip_models: bool) -> str:
    """Rough time estimate based on options."""
    if skip_models and skip_linked:
        return "~2-5 minutes (data fetch + features only)"
    elif skip_models:
        return "~5-10 minutes (data + features + linked entities)"
    elif skip_linked:
        return "~15-30 minutes (models without linked entities)"
    else:
        return "~30-60 minutes (full analysis with all models)"


# ---------------------------------------------------------------------------
# Main interactive flow
# ---------------------------------------------------------------------------

def main() -> int:
    _clear()
    _banner()

    # ------------------------------------------------------------------
    # Step 1: System checks
    # ------------------------------------------------------------------
    _step(1, "System Checks")

    if not check_python_version():
        return 1

    _separator()

    if not check_internet():
        if not _yes_no("Continue without internet? (pipeline will likely fail)"):
            return 1

    # ------------------------------------------------------------------
    # Step 2: Dependencies
    # ------------------------------------------------------------------
    _step(2, "Checking Dependencies")

    deps_ok, missing = check_dependencies()

    if not deps_ok:
        print("")
        _err(f"Missing required packages: {', '.join(missing)}")
        if _yes_no("Install missing packages with pip?"):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
                check=False,
            )
            print("")
            _info("Re-checking dependencies...")
            deps_ok, missing = check_dependencies()
            if not deps_ok:
                _err("Still missing packages. Install manually: pip install -r requirements.txt")
                return 1
        else:
            _info("Run: pip install -r requirements.txt")
            return 1

    # ------------------------------------------------------------------
    # Step 3: LLM Provider & Model Selection (always prompt first)
    # ------------------------------------------------------------------
    _step(3, "LLM Provider & Model Selection")

    keys = setup_llm_keys()
    llm_provider = keys.get("_llm_provider", "")

    # ------------------------------------------------------------------
    # Step 4: Data Source Mode
    # ------------------------------------------------------------------
    _step(4, "Data Source Mode")

    data_mode = choose_data_source_mode()

    # ------------------------------------------------------------------
    # Step 5: Market-specific API keys (only if API + wrappers)
    # ------------------------------------------------------------------
    if data_mode == "api_and_wrappers":
        _step(5, "Market API Keys")
        keys = setup_market_api_keys(keys)
    else:
        # Even in wrappers-only mode, OHLCV price data is needed since
        # most PIT sources (SEC EDGAR, ESEF, EDINET, etc.) do not provide
        # price data.  Without OHLCV, the pipeline loses all price-derived
        # features (returns, volatility, drawdown).
        _step(5, "OHLCV Price Data Key")
        env_path = Path(__file__).resolve().parent / ".env"
        if "ALPHA_VANTAGE_API_KEY" not in keys:
            av_env = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
            if av_env:
                keys["ALPHA_VANTAGE_API_KEY"] = av_env.strip()

        if "ALPHA_VANTAGE_API_KEY" in keys:
            _ok(f"ALPHA_VANTAGE_API_KEY: {_mask_key(keys['ALPHA_VANTAGE_API_KEY'])}")
        else:
            _warn(
                "ALPHA_VANTAGE_API_KEY not set. Most PIT sources don't provide "
                "price data, so OHLCV features (returns, volatility, drawdown) "
                "will be unavailable."
            )
            _info("Register for free: https://www.alphavantage.co/support/#api-key")
            value = _prompt("Enter ALPHA_VANTAGE_API_KEY (or press Enter to skip)")
            if value:
                keys["ALPHA_VANTAGE_API_KEY"] = value.strip()
                os.environ["ALPHA_VANTAGE_API_KEY"] = value.strip()
                _save_key_to_env(env_path, "ALPHA_VANTAGE_API_KEY", value.strip())
                _ok("ALPHA_VANTAGE_API_KEY saved")

    # ------------------------------------------------------------------
    # Step 6: Company + Country Input
    # ------------------------------------------------------------------
    _step(6, "Company Selection")

    print(_bold("  Enter the company you want to analyze:"))
    print(f"  {_dim('Type a ticker (AAPL) or company name (Apple Inc)')}")
    print("")
    company = _prompt("Company ticker or name")
    if not company:
        _err("Company is required.")
        return 1
    _ok(f"Company: {company}")

    print("")
    print(_bold("  Enter the country where this company is listed:"))
    print(f"  {_dim('e.g. US, Japan, UK, South Korea, Brazil, etc.')}")
    print("")
    country = _prompt("Country", "US")
    _ok(f"Country: {country}")

    # --- Personal data check on user input ---
    _check_user_input_pii(company, country, keys)

    # ------------------------------------------------------------------
    # Step 7: LLM resolves market/client
    # ------------------------------------------------------------------
    _step(7, "Market Resolution")

    _info(f"Determining the right data source for {company} in {country}...")
    market_id = resolve_market(company, country, keys)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from operator1.clients.pit_registry import get_market, get_macro_api_for_market
    market = get_market(market_id)
    macro = get_macro_api_for_market(market_id)

    if market:
        _ok(f"Market: {market.country} ({market.pit_api_name})")
        if macro:
            _info(f"Macro data: {macro.api_name}")
        # --- Personal data check on wrapper requirements ---
        _check_market_pii(market_id, market)
    else:
        _err(f"Unknown market: {market_id}")
        return 1

    # ------------------------------------------------------------------
    # Step 8: Pipeline Options
    # ------------------------------------------------------------------
    _step(8, "Pipeline Options")

    skip_linked = not _yes_no(
        "Discover linked entities? (competitors, suppliers)", default=True
    )
    skip_models = not _yes_no(
        "Run temporal models? (forecasting, burn-out)", default=True
    )
    gen_pdf = _yes_no("Generate PDF report? (requires pandoc)", default=False)

    # ------------------------------------------------------------------
    # Step 9: Confirmation & Run
    # ------------------------------------------------------------------
    _step(9, "Confirmation")

    estimate = estimate_runtime(skip_linked, skip_models)

    print(f"  Market:           {_bold(market.country)} ({market.pit_api_name})")
    print(f"  Company:          {_bold(company)}")
    print(f"  Country:          {country}")
    print(f"  Data mode:        {'API + Wrappers' if data_mode == 'api_and_wrappers' else 'Wrappers only'}")
    if macro:
        print(f"  Macro source:     {macro.api_name}")
    print(f"  Linked entities:  {'Yes' if not skip_linked else 'Skip'}")
    print(f"  Temporal models:  {'Yes' if not skip_models else 'Skip'}")
    print(f"  PDF output:       {'Yes' if gen_pdf else 'No'}")
    llm_model = keys.get("_llm_model", "")
    if llm_provider == "gemini":
        _llm_label = f"Gemini / {llm_model or 'default'} (AI-generated)"
    elif llm_provider == "claude":
        _llm_label = f"Claude / {llm_model or 'default'} (AI-generated)"
    else:
        _llm_label = "Template fallback (no LLM key)"
    print(f"  Report engine:    {_llm_label}")
    print(f"  Estimated time:   {_yellow(estimate)}")
    print("")

    if not _yes_no("Start the pipeline?"):
        _info("Cancelled.")
        return 0

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    print("")
    _step(10, "Running Pipeline")

    start_time = time.time()

    cmd = [
        sys.executable, "main.py",
        "--market", market_id,
        "--company", company,
    ]
    if skip_linked:
        cmd.append("--skip-linked")
    if skip_models:
        cmd.append("--skip-models")
    if gen_pdf:
        cmd.append("--pdf")
    if llm_provider:
        cmd.extend(["--llm-provider", llm_provider])
    llm_model = keys.get("_llm_model", "")
    if llm_model:
        cmd.extend(["--llm-model", llm_model])

    _info(f"Command: {' '.join(cmd)}")
    if llm_provider:
        _info(f"LLM: {llm_provider} / {llm_model or 'default'}")
    print("")
    _separator()
    print("")

    result = subprocess.run(cmd, check=False)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("")
    _separator()
    print("")

    if result.returncode == 0:
        _ok(f"Pipeline completed successfully in {minutes}m {seconds}s")
        print("")
        _info("Output files:")

        output_dir = Path("cache")
        if output_dir.exists():
            for f in sorted(output_dir.rglob("*")):
                if f.is_file():
                    size = f.stat().st_size
                    size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
                    print(f"    {f.relative_to('.')}  ({size_str})")

        print("")
        _info("Key files:")
        for key_file in [
            "cache/company_profile.json",
            "cache/report/analysis_report.md",
            "cache/report/analysis_report.pdf",
        ]:
            p = Path(key_file)
            if p.exists():
                print(f"    {_green('[exists]')} {key_file}")
            else:
                print(f"    {_dim('[  --  ]')} {key_file}")
    else:
        _err(f"Pipeline failed (exit code {result.returncode}) after {minutes}m {seconds}s")
        _info("Check the log output above for details.")
        return 1

    print("")
    _bold("  Done! Review the report in cache/report/analysis_report.md")
    print("")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("")
        print(_dim("  Interrupted."))
        sys.exit(130)
