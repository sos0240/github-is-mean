"""Personal data detection guard for Operator 1.

Uses the configured LLM to analyze user input and wrapper API
registration requirements to detect whether personal identifiable
information (PII) is involved. Warns the user before proceeding.

This module addresses a key concern: some government filing APIs
(EDINET, DART, etc.) require personal information during API key
registration (name, phone, affiliation). The guard ensures users
are aware of this before they interact with these services.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PersonalDataCheckResult:
    """Result of a personal data / PII check."""

    has_personal_data: bool = False
    warnings: list[str] = field(default_factory=list)
    details: str = ""
    market_personal_data_level: str = "none"


# ---------------------------------------------------------------------------
# Known PII patterns (regex-based, no LLM needed)
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email address", re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)),
    ("phone number", re.compile(r"(?:\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}")),
    ("social security / national ID", re.compile(r"\b\d{3}[\-\s]?\d{2}[\-\s]?\d{4}\b")),
    ("credit card number", re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b")),
    ("IP address", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("date of birth pattern", re.compile(r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b")),
]


def _check_pii_regex(text: str) -> list[str]:
    """Check text for obvious PII patterns using regex.

    Returns a list of warning strings for each type of PII detected.
    """
    warnings: list[str] = []
    for label, pattern in _PII_PATTERNS:
        if pattern.search(text):
            warnings.append(f"Input appears to contain a {label}")
    return warnings


# ---------------------------------------------------------------------------
# LLM-based PII detection
# ---------------------------------------------------------------------------

_PII_CHECK_PROMPT = """\
You are a privacy guard for a financial analysis tool. Analyze the following
user input and determine if it contains any personal identifiable information
(PII) that should NOT be entered into a financial data pipeline.

User input:
  Company: {company}
  Country: {country}

Check for:
1. Personal names (not company names)
2. Personal addresses or home addresses
3. Personal phone numbers
4. Personal email addresses
5. Government ID numbers (SSN, passport, national ID)
6. Personal financial account numbers

Respond with EXACTLY one of:
- "CLEAN" if the input contains only company/market information
- "PERSONAL: <reason>" if the input contains personal information

Examples:
- "Apple Inc" in "US" -> CLEAN
- "John Smith" in "US" -> PERSONAL: Input appears to be a personal name, not a company
- "123-45-6789" in "US" -> PERSONAL: Input contains what appears to be an SSN
"""


def check_user_input_for_pii(
    company: str,
    country: str,
    llm_client: Any | None = None,
) -> PersonalDataCheckResult:
    """Check user-provided company/country input for personal data.

    Uses regex patterns first (fast, no API call), then optionally
    uses the LLM for more nuanced detection (e.g., distinguishing
    personal names from company names).

    Parameters
    ----------
    company:
        The company name or ticker entered by the user.
    country:
        The country entered by the user.
    llm_client:
        Optional LLM client for AI-based detection. If None, only
        regex-based checks are performed.

    Returns
    -------
    PersonalDataCheckResult with warnings if PII is detected.
    """
    result = PersonalDataCheckResult()
    combined_input = f"{company} {country}"

    # Step 1: Regex-based checks (fast, no API call)
    regex_warnings = _check_pii_regex(combined_input)
    if regex_warnings:
        result.has_personal_data = True
        result.warnings.extend(regex_warnings)
        result.details = "Regex-based PII detection flagged potential personal data."
        return result

    # Step 2: LLM-based check (if available)
    if llm_client is not None:
        try:
            prompt = _PII_CHECK_PROMPT.format(company=company, country=country)
            response = llm_client.generate(prompt)
            if response:
                response_clean = response.strip()
                if response_clean.upper().startswith("PERSONAL"):
                    reason = response_clean.split(":", 1)[-1].strip() if ":" in response_clean else "LLM flagged personal data"
                    result.has_personal_data = True
                    result.warnings.append(reason)
                    result.details = "LLM-based PII detection flagged potential personal data."
                elif response_clean.upper().startswith("CLEAN"):
                    result.details = "LLM confirmed input is clean."
                else:
                    logger.debug("Unexpected LLM PII response: %s", response_clean[:100])
        except Exception as exc:
            logger.debug("LLM PII check failed (non-blocking): %s", exc)

    return result


# ---------------------------------------------------------------------------
# Wrapper input requirement checks
# ---------------------------------------------------------------------------

# Per-market API registration requirements, sourced from .roo/research/ logs.
# personal_data_level: none | low | medium | high
# - none: no registration or no personal data needed
# - low: email-only registration
# - medium: name + phone or affiliation required
# - high: government ID or extensive personal data required

_MARKET_REGISTRATION_INFO: dict[str, dict[str, str]] = {
    "us_sec_edgar": {
        "personal_data_level": "low",
        "registration_requires": "Email address (as User-Agent header)",
        "registration_url": "https://www.sec.gov/os/accessing-edgar-data",
        "note": "SEC EDGAR requires a User-Agent with your email for API access.",
    },
    "uk_companies_house": {
        "personal_data_level": "low",
        "registration_requires": "Email, name, application description",
        "registration_url": "https://developer.company-information.service.gov.uk/",
        "note": "Companies House API key registration requires a developer account.",
    },
    "eu_esef": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://filings.xbrl.org/api",
        "note": "ESEF / filings.xbrl.org is free with no registration.",
    },
    "fr_esef": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://filings.xbrl.org/api",
        "note": "Uses same ESEF API as EU.",
    },
    "de_esef": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://filings.xbrl.org/api",
        "note": "Uses same ESEF API as EU.",
    },
    "jp_edinet": {
        "personal_data_level": "medium",
        "registration_requires": "Name, affiliation, phone number (may need Japanese-compatible), email",
        "registration_url": "https://api.edinet-fsa.go.jp/api/auth/index.aspx?mode=1",
        "note": (
            "EDINET API key registration requires personal information including "
            "phone number and affiliation. Multi-factor auth may require a "
            "Japanese-compatible phone number."
        ),
    },
    "kr_dart": {
        "personal_data_level": "medium",
        "registration_requires": "Email, name, usage purpose, IP address (if corporate)",
        "registration_url": "https://opendart.fss.or.kr/",
        "note": (
            "DART API key registration collects personal information. "
            "Consent to personal info collection is required (re-sought every 2 years)."
        ),
    },
    "tw_mops": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://mops.twse.com.tw",
        "note": "MOPS uses form POST scraping with no API key.",
    },
    "br_cvm": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://dados.cvm.gov.br",
        "note": "CVM open data portal is free with no registration.",
    },
    "cl_cmf": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://www.cmfchile.cl/",
        "note": "CMF Chile data is free with no registration.",
    },
    # Tier 2 markets
    "au_asx": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://www.asx.com.au/asx/1",
        "note": "ASX public API requires no registration.",
    },
    "ca_sedar": {
        "personal_data_level": "low",
        "registration_requires": "Email for SEDAR+ account",
        "registration_url": "https://www.sedarplus.ca/",
        "note": "SEDAR+ may require account for full access.",
    },
    "cn_sse": {
        "personal_data_level": "low",
        "registration_requires": "Email (for akshare/tushare API tokens)",
        "registration_url": "https://tushare.pro/register",
        "note": "Chinese market data via community wrappers may require registration.",
    },
    "hk_hkex": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://www.hkex.com.hk/",
        "note": "HKEX public data is free.",
    },
    "in_bse": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://www.bseindia.com/",
        "note": "BSE India public data is free.",
    },
    "sg_sgx": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://www.sgx.com/",
        "note": "SGX public data is free.",
    },
    "mx_bmv": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://www.bmv.com.mx/",
        "note": "BMV public data is free.",
    },
    "za_jse": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://www.jse.co.za/",
        "note": "JSE public data is free.",
    },
    "ch_six": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://www.six-group.com/",
        "note": "SIX public data is free.",
    },
    "sa_tadawul": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://www.saudiexchange.sa/",
        "note": "Tadawul public data is free.",
    },
    "ae_dfm": {
        "personal_data_level": "none",
        "registration_requires": "None",
        "registration_url": "https://www.dfm.ae/",
        "note": "DFM public data is free.",
    },
}


def get_market_registration_info(market_id: str) -> dict[str, str]:
    """Return the API registration requirements for a market.

    Returns a dict with keys: personal_data_level, registration_requires,
    registration_url, note.
    """
    return _MARKET_REGISTRATION_INFO.get(market_id, {
        "personal_data_level": "none",
        "registration_requires": "Unknown",
        "registration_url": "",
        "note": "",
    })


def check_wrapper_personal_data(
    market_id: str,
    llm_client: Any | None = None,
) -> PersonalDataCheckResult:
    """Check if a market's API wrapper requires personal data for registration.

    Parameters
    ----------
    market_id:
        The market identifier (e.g., 'us_sec_edgar', 'jp_edinet').
    llm_client:
        Optional LLM client. Currently unused (static lookup), but
        reserved for future dynamic analysis.

    Returns
    -------
    PersonalDataCheckResult with warnings about personal data requirements.
    """
    info = get_market_registration_info(market_id)
    level = info.get("personal_data_level", "none")

    result = PersonalDataCheckResult(
        market_personal_data_level=level,
    )

    if level in ("medium", "high"):
        result.has_personal_data = True
        result.warnings.append(
            f"This market's API registration requires personal information: "
            f"{info.get('registration_requires', 'unknown details')}"
        )
        note = info.get("note", "")
        if note:
            result.warnings.append(note)
        result.details = (
            f"Registration URL: {info.get('registration_url', 'N/A')}. "
            f"Personal data level: {level}."
        )
    elif level == "low":
        result.details = (
            f"This market requires basic registration ({info.get('registration_requires', 'email')}). "
            f"Personal data level: low."
        )

    return result


def format_pii_warning(result: PersonalDataCheckResult) -> str:
    """Format a PersonalDataCheckResult into a human-readable warning string.

    Returns empty string if no warnings.
    """
    if not result.has_personal_data and not result.warnings:
        return ""

    lines = []
    if result.has_personal_data:
        lines.append("WARNING: Personal data detected")
        lines.append("")
    for w in result.warnings:
        lines.append(f"  - {w}")
    if result.details:
        lines.append("")
        lines.append(f"  {result.details}")

    return "\n".join(lines)
