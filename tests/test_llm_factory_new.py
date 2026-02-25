"""Tests for operator1.clients.llm_factory."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestLLMFactory:
    def test_get_available_models_gemini(self):
        from operator1.clients.llm_factory import get_available_models
        models = get_available_models("gemini")
        assert len(models) > 0
        assert all("name" in m for m in models)

    def test_get_available_models_claude(self):
        from operator1.clients.llm_factory import get_available_models
        models = get_available_models("claude")
        assert len(models) > 0
        assert all("name" in m for m in models)

    def test_create_llm_client_gemini(self):
        from operator1.clients.llm_factory import create_llm_client
        client = create_llm_client({"GEMINI_API_KEY": "test_key"}, provider="gemini")
        assert client is not None
        assert client.provider_name == "Gemini"

    def test_create_llm_client_claude(self):
        from operator1.clients.llm_factory import create_llm_client
        client = create_llm_client({"ANTHROPIC_API_KEY": "test_key"}, provider="claude")
        assert client is not None
        assert client.provider_name == "Claude"

    def test_create_llm_client_no_key_returns_none(self):
        from operator1.clients.llm_factory import create_llm_client
        client = create_llm_client({}, provider="gemini")
        assert client is None

    def test_auto_detect_provider(self):
        from operator1.clients.llm_factory import _auto_detect_provider
        assert _auto_detect_provider({"GEMINI_API_KEY": "k"}) == "gemini"
        assert _auto_detect_provider({"ANTHROPIC_API_KEY": "k"}) == "claude"
        assert _auto_detect_provider({}) == "gemini"  # default
