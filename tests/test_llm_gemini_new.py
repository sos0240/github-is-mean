"""Tests for operator1.clients.gemini.GeminiClient."""

from __future__ import annotations

import pytest


class TestGeminiClient:
    def test_import(self):
        from operator1.clients.gemini import GeminiClient
        assert GeminiClient is not None

    def test_instantiation(self):
        from operator1.clients.gemini import GeminiClient
        client = GeminiClient(api_key="test_key_123")
        assert client is not None

    def test_provider_name(self):
        from operator1.clients.gemini import GeminiClient
        client = GeminiClient(api_key="test")
        assert client.provider_name == "Gemini"

    def test_model_name(self):
        from operator1.clients.gemini import GeminiClient
        client = GeminiClient(api_key="test")
        assert "gemini" in client.model_name

    def test_max_output_tokens(self):
        from operator1.clients.gemini import GeminiClient
        client = GeminiClient(api_key="test")
        assert client.max_output_tokens > 0
