"""API client modules for external data sources.

Point-in-Time (PIT) data providers -- region-based selection
(all use free community unofficial wrapper libraries):

  - ``us_edgar``         -- US (SEC EDGAR via edgartools + sec-edgar-api)
  - ``eu_esef_wrapper``  -- EU (ESEF/XBRL via pyesef)
  - ``uk_ch_wrapper``    -- UK (Companies House wrapper with caching)
  - ``jp_jquants_wrapper``-- Japan (J-Quants via jquants-api-client)
  - ``kr_dart_wrapper``  -- South Korea (DART via dart-fss)
  - ``tw_mops_wrapper``  -- Taiwan (MOPS scraper wrapper)
  - ``br_cvm_wrapper``   -- Brazil (CVM via pycvm)
  - ``cl_cmf_wrapper``   -- Chile (CMF wrapper with caching)

Supplementary data providers (for partial-coverage regions):
  - ``supplement``       -- OpenFIGI, Euronext, JPX, TWSE, B3, Santiago

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
