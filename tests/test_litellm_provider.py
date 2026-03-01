"""Tests for the LiteLLM provider backend."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unified_llm import GenerationConfig, Message, UnifiedLLM
from unified_llm.providers.litellm_provider import LiteLLMProvider, _LITELLM_PREFIX_MAP


# ---------------------------------------------------------------------------
# Prefix mapping
# ---------------------------------------------------------------------------


class TestPrefixMapping:
    def test_openai_no_prefix(self):
        p = LiteLLMProvider(provider_name="openai")
        assert p._litellm_model("gpt-4o") == "gpt-4o"

    def test_anthropic_prefix(self):
        p = LiteLLMProvider(provider_name="anthropic")
        assert p._litellm_model("claude-sonnet-4-20250514") == "anthropic/claude-sonnet-4-20250514"

    def test_google_prefix(self):
        p = LiteLLMProvider(provider_name="google")
        assert p._litellm_model("gemini-2.5-flash") == "gemini/gemini-2.5-flash"

    def test_ollama_prefix(self):
        p = LiteLLMProvider(provider_name="ollama")
        assert p._litellm_model("llama3.1") == "ollama/llama3.1"

    def test_no_double_prefix(self):
        p = LiteLLMProvider(provider_name="anthropic")
        assert p._litellm_model("anthropic/claude-sonnet-4-20250514") == "anthropic/claude-sonnet-4-20250514"

    def test_custom_prefix_override(self):
        p = LiteLLMProvider(provider_name="openai", litellm_prefix="custom/")
        assert p._litellm_model("gpt-4o") == "custom/gpt-4o"

    def test_unknown_provider_gets_name_prefix(self):
        p = LiteLLMProvider(provider_name="deepseek")
        assert p._litellm_model("chat") == "deepseek/chat"


# ---------------------------------------------------------------------------
# Mocked litellm.completion
# ---------------------------------------------------------------------------


def _fake_response(content="Hello!", model="gpt-4o"):
    """Return an object shaped like a litellm response."""
    return SimpleNamespace(
        id="chatcmpl-test123",
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


class TestComplete:
    @patch("litellm.completion", return_value=_fake_response())
    def test_complete_calls_litellm(self, mock_completion):
        p = LiteLLMProvider(provider_name="openai", default_model="gpt-4o")
        msgs = [Message(role="user", content="Hi")]
        resp = p.complete(msgs)

        assert resp.content == "Hello!"
        assert resp.provider == "openai"
        assert resp.usage.total_tokens == 15
        assert resp.id == "chatcmpl-test123"

        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    @patch("litellm.completion", return_value=_fake_response(model="anthropic/claude-sonnet-4-20250514"))
    def test_complete_anthropic_prefix(self, mock_completion):
        p = LiteLLMProvider(provider_name="anthropic", default_model="claude-sonnet-4-20250514")
        msgs = [Message(role="user", content="Hi")]
        resp = p.complete(msgs)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4-20250514"
        assert resp.provider == "anthropic"

    @patch("litellm.completion", return_value=_fake_response())
    def test_complete_passes_config(self, mock_completion):
        p = LiteLLMProvider(provider_name="openai", default_model="gpt-4o")
        cfg = GenerationConfig(temperature=0, max_tokens=10, seed=42)
        msgs = [Message(role="user", content="Hi")]
        p.complete(msgs, config=cfg)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 10
        assert call_kwargs["seed"] == 42

    @patch("litellm.completion", return_value=_fake_response())
    def test_complete_json_format(self, mock_completion):
        p = LiteLLMProvider(provider_name="openai", default_model="gpt-4o")
        cfg = GenerationConfig(response_format="json")
        msgs = [Message(role="user", content="Hi")]
        p.complete(msgs, config=cfg)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @patch("litellm.completion", return_value=_fake_response())
    def test_complete_with_api_key_and_base(self, mock_completion):
        p = LiteLLMProvider(
            provider_name="openai",
            default_model="gpt-4o",
            api_key="sk-test",
            api_base="http://localhost:8000/v1",
        )
        msgs = [Message(role="user", content="Hi")]
        p.complete(msgs)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_key"] == "sk-test"
        assert call_kwargs["api_base"] == "http://localhost:8000/v1"


# ---------------------------------------------------------------------------
# Mocked streaming
# ---------------------------------------------------------------------------


class TestStream:
    @patch("litellm.completion")
    def test_stream_yields_chunks(self, mock_completion):
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel"), finish_reason=None)]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="lo!"), finish_reason="stop")]),
        ]
        mock_completion.return_value = iter(chunks)

        p = LiteLLMProvider(provider_name="openai", default_model="gpt-4o")
        msgs = [Message(role="user", content="Hi")]
        result = list(p.stream(msgs))

        assert len(result) == 2
        assert result[0].delta == "Hel"
        assert result[0].finish_reason is None
        assert result[1].delta == "lo!"

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["stream"] is True


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


class TestAsyncComplete:
    @pytest.mark.asyncio
    @patch("litellm.acompletion", new_callable=AsyncMock, return_value=_fake_response())
    async def test_acomplete(self, mock_acompletion):
        p = LiteLLMProvider(provider_name="openai", default_model="gpt-4o")
        msgs = [Message(role="user", content="Hi")]
        resp = await p.acomplete(msgs)

        assert resp.content == "Hello!"
        assert resp.provider == "openai"
        mock_acompletion.assert_called_once()


# ---------------------------------------------------------------------------
# End-to-end through UnifiedLLM
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @patch("litellm.completion", return_value=_fake_response())
    def test_unified_llm_routes_to_litellm(self, mock_completion):
        llm = UnifiedLLM()
        llm.add_provider("openai")
        resp = llm.complete("openai/gpt-4o", "Hello")

        assert resp.content == "Hello!"
        assert resp.provider == "openai"
        mock_completion.assert_called_once()
        llm.shutdown()

    @patch("litellm.completion", return_value=_fake_response(model="anthropic/claude-sonnet-4-20250514"))
    def test_unified_llm_anthropic(self, mock_completion):
        llm = UnifiedLLM()
        llm.add_provider("anthropic")
        resp = llm.complete("anthropic/claude-sonnet-4-20250514", "Hello")

        assert resp.provider == "anthropic"
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4-20250514"
        llm.shutdown()

    @patch("litellm.completion", return_value=_fake_response(model="ollama/llama3.1"))
    def test_unified_llm_ollama_with_api_base(self, mock_completion):
        llm = UnifiedLLM()
        llm.add_provider("ollama", api_base="http://localhost:11434")
        resp = llm.complete("ollama/llama3.1", "Hello")

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "ollama/llama3.1"
        assert call_kwargs["api_base"] == "http://localhost:11434"
        llm.shutdown()


# ---------------------------------------------------------------------------
# Registry defaults
# ---------------------------------------------------------------------------


class TestRegistryDefaults:
    def test_openai_default_model(self):
        llm = UnifiedLLM()
        llm.add_provider("openai")
        provider = llm.get_provider("openai")
        assert provider.default_model == "gpt-4o"
        assert isinstance(provider, LiteLLMProvider)
        llm.shutdown()

    def test_anthropic_default_model(self):
        llm = UnifiedLLM()
        llm.add_provider("anthropic")
        provider = llm.get_provider("anthropic")
        assert provider.default_model == "claude-sonnet-4-20250514"
        llm.shutdown()

    def test_user_kwargs_override_defaults(self):
        llm = UnifiedLLM()
        llm.add_provider("openai", default_model="gpt-3.5-turbo")
        provider = llm.get_provider("openai")
        assert provider.default_model == "gpt-3.5-turbo"
        llm.shutdown()
