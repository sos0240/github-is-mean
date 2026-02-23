"""API client modules for external data sources.

Point-in-Time (PIT) data providers -- region-based selection:

  - ``us_edgar``         -- US (SEC EDGAR, NYSE/NASDAQ)
  - ``eu_esef_wrapper``  -- EU (ESEF/XBRL, pan-European)
  - ``uk_ch_wrapper``    -- UK (Companies House, LSE)
  - ``jp_edinet_wrapper``-- Japan (EDINET, TSE)
  - ``kr_dart_wrapper``  -- South Korea (DART, KOSPI/KOSDAQ)
  - ``tw_mops_wrapper``  -- Taiwan (MOPS, TWSE/TPEX)
  - ``br_cvm_wrapper``   -- Brazil (CVM, B3)
  - ``cl_cmf_wrapper``   -- Chile (CMF, Santiago)

Supplementary data providers (for partial-coverage regions):
  - ``supplement``       -- OpenFIGI, Euronext, JPX, TWSE, B3, Santiago
  - ``ohlcv_provider``   -- Alpha Vantage (global), TWSE (Taiwan)

Data normalization:
  - ``canonical_translator`` -- Maps all API data to canonical field format

AI / Report generation (choose one via ``llm_provider`` config):
  - ``gemini``           -- Google Gemini
  - ``claude``           -- Anthropic Claude (alternative to Gemini)
  - ``llm_base``         -- Abstract base class for LLM providers
  - ``llm_factory``      -- Factory to create the configured LLM client

Use :func:`equity_provider.create_pit_client` to get the right
backend based on the user's market selection.

See :mod:`pit_registry` for the full market catalog.
"""
