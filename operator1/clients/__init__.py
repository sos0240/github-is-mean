"""API client modules for external data sources.

Point-in-Time (PIT) data providers -- region-based selection:

  - ``sec_edgar``        -- US (SEC EDGAR, NYSE/NASDAQ)
  - ``esef``             -- EU (ESEF/XBRL, pan-European)
  - ``companies_house``  -- UK (Companies House, LSE)
  - ``edinet``           -- Japan (EDINET, TSE)
  - ``dart``             -- South Korea (DART, KOSPI/KOSDAQ)
  - ``mops``             -- Taiwan (MOPS, TWSE/TPEX)
  - ``cvm``              -- Brazil (CVM, B3)
  - ``cmf``              -- Chile (CMF, Santiago)

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
