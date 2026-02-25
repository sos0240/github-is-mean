"""Tests for operator1.clients.llm_base."""

from __future__ import annotations

import pytest


class TestLLMBase:
    def test_llm_client_is_abstract(self):
        from operator1.clients.llm_base import LLMClient
        with pytest.raises(TypeError):
            LLMClient()

    def test_rate_limit_sleep_function(self):
        from operator1.clients.llm_base import _rate_limit_sleep
        assert callable(_rate_limit_sleep)

    def test_get_best_model_gemini(self):
        from operator1.clients.llm_base import get_best_model, GEMINI_MODELS
        best = get_best_model("Gemini", GEMINI_MODELS)
        assert best in GEMINI_MODELS

    def test_get_best_model_claude(self):
        from operator1.clients.llm_base import get_best_model, CLAUDE_MODELS
        best = get_best_model("Claude", CLAUDE_MODELS)
        assert best in CLAUDE_MODELS

    def test_validate_model_valid(self):
        from operator1.clients.llm_base import validate_model, GEMINI_MODELS
        model = validate_model("gemini-2.0-flash", GEMINI_MODELS, "Gemini")
        assert model == "gemini-2.0-flash"

    def test_validate_model_invalid(self):
        from operator1.clients.llm_base import validate_model, GEMINI_MODELS
        model = validate_model("nonexistent-model", GEMINI_MODELS, "Gemini")
        # Should fall back to best model
        assert model in GEMINI_MODELS

    def test_gemini_models_registry(self):
        from operator1.clients.llm_base import GEMINI_MODELS
        assert len(GEMINI_MODELS) > 0
        for name, info in GEMINI_MODELS.items():
            assert "max_output_tokens" in info
            assert "context_window" in info

    def test_claude_models_registry(self):
        from operator1.clients.llm_base import CLAUDE_MODELS
        assert len(CLAUDE_MODELS) > 0
        for name, info in CLAUDE_MODELS.items():
            assert "max_output_tokens" in info
            assert "context_window" in info
