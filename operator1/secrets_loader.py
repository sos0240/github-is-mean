"""Load API secrets from Kaggle, .env file, or environment variables.

On Kaggle, keys are stored via the Kaggle Secrets client.  For local
development, falls back to a ``.env`` file in the project root, then
to environment variables.

The new PIT architecture only requires GEMINI_API_KEY (for report
generation).  Market-specific keys are optional and only needed for
certain PIT APIs (Companies House, DART).
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Attempt to load .env file for local development
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Load variables from .env file if python-dotenv is available."""
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        load_dotenv(env_path)
        logger.info("Loaded environment from %s", env_path)
    except ImportError:
        # Manual fallback: parse simple KEY=VALUE lines
        try:
            with open(env_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ.setdefault(key, value)
            logger.info("Loaded .env file manually (python-dotenv not installed)")
        except Exception as exc:
            logger.warning("Failed to parse .env file: %s", exc)


# Required keys -- pipeline will halt if any are missing.
# NOTE: No keys are strictly required.  All PIT data sources are free.
# GEMINI_API_KEY is optional (used for report generation only).
_REQUIRED_KEYS: tuple[str, ...] = ()

# Optional keys -- logged as info if absent.
_OPTIONAL_KEYS = (
    "GEMINI_API_KEY",           # Report generation (Gemini AI)
    "ANTHROPIC_API_KEY",        # Report generation (Claude AI -- alternative to Gemini)
    "COMPANIES_HOUSE_API_KEY",  # UK Companies House (free registration)
    "JQUANTS_API_KEY",          # Japan J-Quants (free email registration at jpx-jquants.com)
    "DART_API_KEY",             # South Korea DART (free registration)
    "EDGAR_IDENTITY",           # US SEC EDGAR email identity (required by SEC regulation)
    "openfigi_key",             # OpenFIGI (optional, for higher rate limits)
    "FRED_API_KEY",             # US FRED macro data (free registration)
)


def _load_from_kaggle() -> dict[str, str]:
    """Attempt to read secrets via Kaggle UserSecretsClient."""
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-untyped]
        client = UserSecretsClient()
        secrets: dict[str, str] = {}
        for key in (*_REQUIRED_KEYS, *_OPTIONAL_KEYS):
            try:
                value = client.get_secret(key)
                if value:
                    secrets[key] = value.strip()
            except Exception:
                pass
        return secrets
    except ImportError:
        return {}


def _load_from_env() -> dict[str, str]:
    """Fallback: read from OS environment variables."""
    secrets: dict[str, str] = {}
    for key in (*_REQUIRED_KEYS, *_OPTIONAL_KEYS):
        value = os.environ.get(key)
        if value:
            secrets[key] = value.strip()
    return secrets


def load_secrets() -> dict[str, str]:
    """Return a dict of API secrets.

    Priority order:
    1. Kaggle Secrets client (if running on Kaggle)
    2. ``.env`` file in project root
    3. OS environment variables

    Raises ``SystemExit`` with a clear message if any required key is missing.
    """
    # Try .env file first (for local development)
    _load_dotenv()

    # Try Kaggle secrets first
    secrets = _load_from_kaggle()
    if not secrets:
        secrets = _load_from_env()

    # Validate required keys (currently none are required)
    missing = [k for k in _REQUIRED_KEYS if not secrets.get(k)]
    if missing:
        raise SystemExit(
            f"Missing required API keys: {', '.join(missing)}\n"
            f"Create a .env file from .env.example.\n"
            f"Note: Most PIT data sources (SEC EDGAR, ESEF, CVM, etc.) "
            f"are free and do not require API keys.\n"
            f"J-Quants (Japan), DART (Korea), and Companies House (UK) "
            f"require free registration for API keys."
        )

    # Log optional keys
    for key in _OPTIONAL_KEYS:
        if not secrets.get(key):
            label = {
                "GEMINI_API_KEY": "report generation (Gemini)",
                "ANTHROPIC_API_KEY": "report generation (Claude)",
                "COMPANIES_HOUSE_API_KEY": "UK market",
                "JQUANTS_API_KEY": "Japan market (J-Quants)",
                "DART_API_KEY": "South Korea market",
                "EDGAR_IDENTITY": "US SEC EDGAR identity",
                "openfigi_key": "OpenFIGI higher rate limits",
                "FRED_API_KEY": "US FRED macro data",
            }.get(key, "specific features")
            logger.info(
                "Optional key %s not set (needed for %s).",
                key, label,
            )

    return secrets
