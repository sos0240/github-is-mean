"""Tests for operator1.clients.claude.ClaudeClient."""

from __future__ import annotations

import pytest


class TestClaudeClient:
    def test_import(self):
        from operator1.clients.claude import ClaudeClient
        assert ClaudeClient is not None

    def test_instantiation(self):
        from operator1.clients.claude import ClaudeClient
        client = ClaudeClient(api_key="test_key_123")
        assert client is not None

    def test_provider_name(self):
        from operator1.clients.claude import ClaudeClient
        client = ClaudeClient(api_key="test")
        assert client.provider_name == "Claude"

    def test_model_name(self):
        from operator1.clients.claude import ClaudeClient
        client = ClaudeClient(api_key="test")
        assert "claude" in client.model_name

    def test_max_output_tokens(self):
        from operator1.clients.claude import ClaudeClient
        client = ClaudeClient(api_key="test")
        assert client.max_output_tokens > 0
