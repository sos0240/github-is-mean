# Add Claude as Alternative LLM Provider

## Overview
The project currently uses Google Gemini as its sole LLM provider for report generation, entity discovery, macro indicator mapping, and news sentiment scoring. This plan adds Anthropic Claude as an alternative provider that users can select via configuration.

## Architecture

### Current State
- `GeminiClient` in `operator1/clients/gemini.py` handles all LLM interactions
- Used in: entity discovery, macro mapping, report generation, sentiment scoring
- Instantiated directly in `main.py` with `GeminiClient(api_key=...)`

### Target State
- Abstract `LLMClient` base class defines the shared interface
- `GeminiClient` and `ClaudeClient` both implement `LLMClient`
- Factory function `create_llm_client(secrets, config)` returns the correct provider
- User selects provider via `llm_provider` in `global_config.yml` or env var

## Files to Create
1. **`operator1/clients/llm_base.py`** - Abstract base class with shared prompts and JSON parsing
2. **`operator1/clients/claude.py`** - Claude client using Anthropic Messages API
3. **`operator1/clients/llm_factory.py`** - Factory function for provider selection

## Files to Modify
1. **`operator1/constants.py`** - Add `CLAUDE_BASE_URL`
2. **`operator1/secrets_loader.py`** - Add `ANTHROPIC_API_KEY` to optional keys
3. **`.env.example`** - Add Claude configuration section
4. **`config/global_config.yml`** - Add `llm_provider` setting
5. **`main.py`** - Use factory instead of direct `GeminiClient` instantiation
6. **`operator1/clients/__init__.py`** - Update docstring

## Interface Contract
Both clients expose:
- `propose_linked_entities(target_profile, sector_hints)` -> dict
- `propose_macro_indicator_mappings(country, sector)` -> dict
- `generate_report(company_profile_json, ...)` -> str
- `score_sentiment(headlines, batch_size)` -> list[float]
