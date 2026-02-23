#!/usr/bin/env python3
"""Operator 1 -- Interactive Terminal Launcher.

A user-friendly interface that guides you through running the
financial analysis pipeline step by step.

Usage:
    python run.py
"""

from __future__ import annotations

import os
import sys
import time
import socket
import shutil
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
# Region / Market / Company selection
# ---------------------------------------------------------------------------

def choose_region() -> str:
    """Let the user pick a region."""
    # Import registry
    sys.path.insert(0, str(Path(__file__).resolve().parent))
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
            _ok(f"Region: {regions[idx]}")
            return regions[idx]
    except ValueError:
        # Try name match
        for r in regions:
            if choice.lower() in r.lower():
                _ok(f"Region: {r}")
                return r

    _info("Defaulting to North America")
    return "North America"


def choose_market(region: str) -> str:
    """Let the user pick a market/exchange within a region."""
    from operator1.clients.pit_registry import get_markets_by_region

    markets = get_markets_by_region(region)
    print("")
    print(_bold(f"  Markets in {region}:"))
    print("")

    for i, m in enumerate(markets, 1):
        key_note = _dim(" (API key required)") if m.requires_api_key else _dim(" (no key needed)")
        print(f"    {_bold(str(i))}. {m.country} -- {m.exchange}")
        print(f"       Data source: {m.pit_api_name}{key_note}")
        print(f"       Market cap: {m.market_cap}")
        print("")

    choice = _prompt("Select market (1-{})".format(len(markets)), "1")
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(markets):
            _ok(f"Market: {markets[idx].country} ({markets[idx].pit_api_name})")
            return markets[idx].market_id
    except ValueError:
        for m in markets:
            if choice.lower() in m.country.lower():
                _ok(f"Market: {m.country} ({m.pit_api_name})")
                return m.market_id

    _ok(f"Market: {markets[0].country}")
    return markets[0].market_id


def choose_company() -> str:
    """Let the user enter a company ticker or name."""
    print("")
    print(_bold("  Enter a company to analyze:"))
    print(f"  {_dim('Type a ticker (AAPL) or company name (Apple Inc)')}")
    print("")

    company = _prompt("Company ticker or name")
    if not company:
        _err("Company is required.")
        sys.exit(1)

    _ok(f"Company: {company}")
    return company


def check_optional_keys() -> dict[str, str]:
    """Load optional API keys from .env or environment."""
    env_path = Path(__file__).resolve().parent / ".env"

    keys: dict[str, str] = {}
    if env_path.exists():
        _info(f"Found .env file at {env_path}")
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

    # Check environment variables
    optional_keys = [
        "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
        "COMPANIES_HOUSE_API_KEY", "DART_API_KEY",
        "FRED_API_KEY", "ESTAT_API_KEY", "KOSIS_API_KEY",
        "INSEE_API_KEY", "BCCH_API_KEY",
    ]
    for key_name in optional_keys:
        if key_name not in keys:
            env_val = os.environ.get(key_name)
            if env_val and env_val.strip():
                keys[key_name] = env_val.strip()

    # Show key status
    print(_dim("  Optional API keys (all data sources are free):"))

    # LLM keys (Gemini or Claude)
    has_gemini = "GEMINI_API_KEY" in keys
    has_claude = "ANTHROPIC_API_KEY" in keys
    if has_gemini:
        _ok(f"GEMINI_API_KEY: {_mask_key(keys['GEMINI_API_KEY'])} (Gemini report generation)")
    else:
        _info("GEMINI_API_KEY: not set")
    if has_claude:
        _ok(f"ANTHROPIC_API_KEY: {_mask_key(keys['ANTHROPIC_API_KEY'])} (Claude report generation)")
    else:
        _info("ANTHROPIC_API_KEY: not set")

    if not has_gemini and not has_claude:
        _warn("No LLM key set (reports will use template fallback)")

    if "FRED_API_KEY" in keys:
        _ok(f"FRED_API_KEY: {_mask_key(keys['FRED_API_KEY'])} (US macro data)")
    else:
        _info("FRED_API_KEY: not set (US macro data limited)")

    # Count other optional keys
    _llm_keys = ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "FRED_API_KEY")
    other_keys = [k for k in keys if k not in _llm_keys]
    if other_keys:
        _ok(f"{len(other_keys)} other optional keys configured")
    print("")

    # If no LLM key at all, offer to enter one
    if not has_gemini and not has_claude:
        print(_dim("  An LLM API key enables AI-generated report narratives."))
        print(_dim("  Choose a provider:"))
        print(f"    {_bold('1')}. Google Gemini  -- https://aistudio.google.com/apikey")
        print(f"    {_bold('2')}. Anthropic Claude -- https://console.anthropic.com/")
        print(f"    {_bold('3')}. Skip (use template fallback)")
        print("")

        llm_choice = _prompt("LLM provider (1/2/3)", "3")
        if llm_choice == "1":
            value = _prompt("Enter GEMINI_API_KEY")
            if value:
                keys["GEMINI_API_KEY"] = value.strip()
                try:
                    with open(env_path, "a") as f:
                        f.write(f"\nGEMINI_API_KEY={value.strip()}\n")
                    _ok("Saved GEMINI_API_KEY to .env")
                except Exception:
                    pass
        elif llm_choice == "2":
            value = _prompt("Enter ANTHROPIC_API_KEY")
            if value:
                keys["ANTHROPIC_API_KEY"] = value.strip()
                try:
                    with open(env_path, "a") as f:
                        f.write(f"\nANTHROPIC_API_KEY={value.strip()}\n")
                    _ok("Saved ANTHROPIC_API_KEY to .env")
                except Exception:
                    pass
        else:
            _info("Skipping LLM setup. Reports will use template fallback.")

    # Set in environment
    for k, v in keys.items():
        os.environ[k] = v

    return keys


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
    # Step 3: Region & Market Selection
    # ------------------------------------------------------------------
    _step(3, "Select Market")

    _info("Operator 1 uses free government filing APIs (Point-in-Time data).")
    _info("No expensive data subscriptions required.")
    print("")

    region = choose_region()
    market_id = choose_market(region)

    # Show macro source
    from operator1.clients.pit_registry import get_macro_api_for_market, get_market
    market = get_market(market_id)
    macro = get_macro_api_for_market(market_id)
    if macro:
        _info(f"Macro data: {macro.api_name}")

    # ------------------------------------------------------------------
    # Step 4: Company Selection
    # ------------------------------------------------------------------
    _step(4, "Select Company")

    company = choose_company()

    # ------------------------------------------------------------------
    # Step 5: API Keys (optional)
    # ------------------------------------------------------------------
    _step(5, "API Keys")

    keys = check_optional_keys()

    # ------------------------------------------------------------------
    # Step 6: Options
    # ------------------------------------------------------------------
    _step(6, "Pipeline Options")

    skip_linked = not _yes_no(
        "Discover linked entities? (competitors, suppliers)", default=True
    )
    skip_models = not _yes_no(
        "Run temporal models? (forecasting, burn-out)", default=True
    )
    gen_pdf = _yes_no("Generate PDF report? (requires pandoc)", default=False)

    # LLM provider selection
    has_gemini = "GEMINI_API_KEY" in keys
    has_claude = "ANTHROPIC_API_KEY" in keys
    llm_provider = ""  # empty = auto-detect

    if has_gemini and has_claude:
        # Both keys available -- let the user choose
        print("")
        print(_bold("  LLM Provider for report generation:"))
        print(f"    {_bold('1')}. Google Gemini  (gemini-2.0-flash)")
        print(f"    {_bold('2')}. Anthropic Claude (claude-sonnet-4)")
        print("")
        prov_choice = _prompt("Choose LLM provider (1/2)", "1")
        if prov_choice == "2":
            llm_provider = "claude"
            _ok("Using Claude for AI features")
        else:
            llm_provider = "gemini"
            _ok("Using Gemini for AI features")
    elif has_claude:
        llm_provider = "claude"
        _info("Using Claude for AI features (ANTHROPIC_API_KEY found)")
    elif has_gemini:
        llm_provider = "gemini"
        _info("Using Gemini for AI features (GEMINI_API_KEY found)")
    else:
        _info("No LLM key configured. Reports will use template fallback.")

    # ------------------------------------------------------------------
    # Step 7: Confirmation
    # ------------------------------------------------------------------
    _step(7, "Confirmation")

    estimate = estimate_runtime(skip_linked, skip_models)

    print(f"  Market:           {_bold(market.country)} ({market.pit_api_name})")
    print(f"  Company:          {_bold(company)}")
    if macro:
        print(f"  Macro source:     {macro.api_name}")
    print(f"  Linked entities:  {'Yes' if not skip_linked else 'Skip'}")
    print(f"  Temporal models:  {'Yes' if not skip_models else 'Skip'}")
    print(f"  PDF output:       {'Yes' if gen_pdf else 'No'}")
    if llm_provider == "gemini":
        _llm_label = "Gemini (AI-generated)"
    elif llm_provider == "claude":
        _llm_label = "Claude (AI-generated)"
    else:
        _llm_label = "Template fallback (no LLM key)"
    print(f"  Report engine:    {_llm_label}")
    print(f"  Estimated time:   {_yellow(estimate)}")
    print("")

    if not _yes_no("Start the pipeline?"):
        _info("Cancelled.")
        return 0

    # ------------------------------------------------------------------
    # Step 8: Run pipeline
    # ------------------------------------------------------------------
    _step(8, "Running Pipeline")

    start_time = time.time()

    # Pass LLM provider choice via environment variable so the factory
    # in main.py picks it up without needing a CLI flag.
    if llm_provider:
        os.environ["LLM_PROVIDER"] = llm_provider

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

    _info(f"Command: {' '.join(cmd)}")
    if llm_provider:
        _info(f"LLM_PROVIDER={llm_provider}")
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
