"""Live integration tests — send a simple message to each API provider.

These tests hit real APIs and are skipped when the required service is
unavailable (missing API key or Ollama not running).

Run explicitly with:  uv run pytest tests/test_providers_live.py -v
Skip in CI with:      uv run pytest tests/ --ignore=tests/test_providers_live.py
"""

import os
import subprocess

import pytest

from unified_llm import CompletionResponse, FinishReason, GenerationConfig, UnifiedLLM

PROMPT = "What is 2+2? Reply with just the number."
# max_tokens must be high enough for models that use internal "thinking" tokens
# (e.g. Qwen3 uses ~30 thinking tokens before the visible answer).
CONFIG = GenerationConfig(temperature=0, max_tokens=256)


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

def _ollama_running() -> bool:
    """Check if Ollama is reachable on localhost."""
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "2", "http://localhost:11434/api/tags"],
            capture_output=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _ollama_has_model(model: str) -> bool:
    """Check if a specific model is pulled in Ollama."""
    try:
        import json
        result = subprocess.run(
            ["curl", "-s", "--max-time", "2", "http://localhost:11434/api/tags"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return False
        data = json.loads(result.stdout)
        names = [m["name"] for m in data.get("models", [])]
        return any(model in n for n in names)
    except Exception:
        return False


skip_no_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
skip_no_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
skip_no_google = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"
)
skip_no_ollama = pytest.mark.skipif(
    not _ollama_running(), reason="Ollama not running on localhost:11434"
)


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------

def _assert_valid_response(resp: CompletionResponse, provider: str) -> None:
    """Common checks for any provider response."""
    assert isinstance(resp, CompletionResponse)
    assert resp.content.strip() != ""
    assert resp.provider == provider
    assert resp.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)
    assert resp.usage.total_tokens > 0
    assert resp.latency_ms > 0


def _assert_answer_is_4(resp: CompletionResponse) -> None:
    """The model should answer '4' to 'What is 2+2?'."""
    clean = resp.content.strip().rstrip(".")
    assert "4" in clean, f"Expected '4' in response, got: {resp.content!r}"


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

@skip_no_ollama
class TestOllama:
    @pytest.fixture(autouse=True)
    def _client(self):
        self.llm = UnifiedLLM()
        self.llm.add_provider("ollama")
        yield
        self.llm.shutdown()

    @pytest.mark.skipif(not _ollama_has_model("qwen3"), reason="qwen3:4b not pulled")
    def test_qwen3_simple_message(self):
        resp = self.llm.complete("ollama/qwen3:4b", PROMPT, CONFIG)
        _assert_valid_response(resp, "ollama")
        _assert_answer_is_4(resp)

    @pytest.mark.skipif(not _ollama_has_model("gemma3"), reason="gemma3:4b not pulled")
    def test_gemma3_simple_message(self):
        resp = self.llm.complete("ollama/gemma3:4b", PROMPT, CONFIG)
        _assert_valid_response(resp, "ollama")
        _assert_answer_is_4(resp)

    @pytest.mark.skipif(not _ollama_has_model("qwen3:8b"), reason="qwen3:8b not pulled")
    def test_qwen3_8b_simple_message(self):
        resp = self.llm.complete("ollama/qwen3:8b", PROMPT, CONFIG)
        _assert_valid_response(resp, "ollama")
        _assert_answer_is_4(resp)

    @pytest.mark.skipif(not _ollama_has_model("qwen3"), reason="qwen3:4b not pulled")
    def test_response_structure(self):
        """Verify all expected fields are populated."""
        resp = self.llm.complete("ollama/qwen3:4b", PROMPT, CONFIG)
        assert resp.model != ""
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0
        assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens

    @pytest.mark.skipif(not _ollama_has_model("qwen3"), reason="qwen3:4b not pulled")
    def test_multi_turn_conversation(self):
        """Send a two-turn conversation."""
        from unified_llm import Message
        messages = [
            Message(role="user", content="Remember the number 7."),
            Message(role="assistant", content="OK, I'll remember the number 7."),
            Message(role="user", content="What number did I ask you to remember? Reply with just the number."),
        ]
        resp = self.llm.complete("ollama/qwen3:4b", messages, CONFIG)
        _assert_valid_response(resp, "ollama")
        assert "7" in resp.content.strip()

    @pytest.mark.skipif(not _ollama_has_model("gemma3"), reason="gemma3:4b not pulled")
    def test_system_prompt(self):
        """System prompt should influence the response.

        Uses gemma3 since Qwen3's thinking mode can consume all tokens internally.
        """
        resp = self.llm.complete(
            "ollama/gemma3:4b",
            "What language should I learn?",
            CONFIG,
            system="You always recommend Rust. Reply in one sentence.",
        )
        _assert_valid_response(resp, "ollama")
        assert "rust" in resp.content.lower()

    @pytest.mark.skipif(not _ollama_has_model("gemma3"), reason="gemma3:4b not pulled")
    def test_max_tokens_respected(self):
        """A low max_tokens should produce a short response.

        Uses gemma3 here since Qwen3 thinking tokens make token counting unreliable.
        """
        short_config = GenerationConfig(temperature=0, max_tokens=10)
        resp = self.llm.complete("ollama/gemma3:4b", "Write a long essay about the ocean.", short_config)
        _assert_valid_response(resp, "ollama")
        assert resp.usage.completion_tokens <= 15  # small buffer over limit


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

@skip_no_openai
class TestOpenAI:
    @pytest.fixture(autouse=True)
    def _client(self):
        self.llm = UnifiedLLM()
        self.llm.add_provider("openai")
        yield
        self.llm.shutdown()

    def test_gpt4o_simple_message(self):
        resp = self.llm.complete("openai/gpt-4o", PROMPT, CONFIG)
        _assert_valid_response(resp, "openai")
        _assert_answer_is_4(resp)

    def test_gpt4o_mini_simple_message(self):
        resp = self.llm.complete("openai/gpt-4o-mini", PROMPT, CONFIG)
        _assert_valid_response(resp, "openai")
        _assert_answer_is_4(resp)

    def test_response_structure(self):
        resp = self.llm.complete("openai/gpt-4o-mini", PROMPT, CONFIG)
        assert resp.model != ""
        assert resp.id != ""
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0

    def test_multi_turn_conversation(self):
        from unified_llm import Message
        messages = [
            Message(role="user", content="Remember the number 42."),
            Message(role="assistant", content="OK, I'll remember the number 42."),
            Message(role="user", content="What number did I ask you to remember? Reply with just the number."),
        ]
        resp = self.llm.complete("openai/gpt-4o-mini", messages, CONFIG)
        _assert_valid_response(resp, "openai")
        assert "42" in resp.content.strip()

    def test_system_prompt(self):
        resp = self.llm.complete(
            "openai/gpt-4o-mini",
            "What language should I learn?",
            CONFIG,
            system="You always recommend Python. Reply in one sentence.",
        )
        _assert_valid_response(resp, "openai")
        assert "python" in resp.content.lower()


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

@skip_no_anthropic
class TestAnthropic:
    @pytest.fixture(autouse=True)
    def _client(self):
        self.llm = UnifiedLLM()
        self.llm.add_provider("anthropic")
        yield
        self.llm.shutdown()

    def test_claude_sonnet_simple_message(self):
        resp = self.llm.complete("anthropic/claude-sonnet-4-20250514", PROMPT, CONFIG)
        _assert_valid_response(resp, "anthropic")
        _assert_answer_is_4(resp)

    def test_claude_haiku_simple_message(self):
        resp = self.llm.complete("anthropic/claude-haiku-4-5-20251001", PROMPT, CONFIG)
        _assert_valid_response(resp, "anthropic")
        _assert_answer_is_4(resp)

    def test_response_structure(self):
        resp = self.llm.complete("anthropic/claude-haiku-4-5-20251001", PROMPT, CONFIG)
        assert resp.model != ""
        assert resp.id != ""
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0

    def test_multi_turn_conversation(self):
        from unified_llm import Message
        messages = [
            Message(role="user", content="Remember the color blue."),
            Message(role="assistant", content="OK, I'll remember the color blue."),
            Message(role="user", content="What color did I mention? Reply with just the color."),
        ]
        resp = self.llm.complete("anthropic/claude-haiku-4-5-20251001", messages, CONFIG)
        _assert_valid_response(resp, "anthropic")
        assert "blue" in resp.content.lower()

    def test_system_prompt(self):
        resp = self.llm.complete(
            "anthropic/claude-haiku-4-5-20251001",
            "What language should I learn?",
            CONFIG,
            system="You always recommend Go. Reply in one sentence.",
        )
        _assert_valid_response(resp, "anthropic")
        assert "go" in resp.content.lower()


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

@skip_no_google
class TestGoogle:
    @pytest.fixture(autouse=True)
    def _client(self):
        self.llm = UnifiedLLM()
        self.llm.add_provider("google")
        yield
        self.llm.shutdown()

    def test_gemini_flash_simple_message(self):
        resp = self.llm.complete("google/gemini-2.5-flash", PROMPT, CONFIG)
        _assert_valid_response(resp, "google")
        _assert_answer_is_4(resp)

    def test_response_structure(self):
        resp = self.llm.complete("google/gemini-2.5-flash", PROMPT, CONFIG)
        assert resp.model != ""
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0

    def test_multi_turn_conversation(self):
        from unified_llm import Message
        messages = [
            Message(role="user", content="Remember the word 'banana'."),
            Message(role="assistant", content="OK, I'll remember 'banana'."),
            Message(role="user", content="What word did I mention? Reply with just the word."),
        ]
        resp = self.llm.complete("google/gemini-2.5-flash", messages, CONFIG)
        _assert_valid_response(resp, "google")
        assert "banana" in resp.content.lower()

    def test_system_prompt(self):
        resp = self.llm.complete(
            "google/gemini-2.5-flash",
            "What language should I learn?",
            CONFIG,
            system="You always recommend TypeScript. Reply in one sentence.",
        )
        _assert_valid_response(resp, "google")
        assert "typescript" in resp.content.lower()


# ---------------------------------------------------------------------------
# Cross-provider (requires at least 2 providers)
# ---------------------------------------------------------------------------

class TestCrossProvider:
    """Tests that compare behavior across multiple available providers."""

    def _available_models(self) -> list[str]:
        """Return model IDs for all available providers."""
        models = []
        if _ollama_running() and _ollama_has_model("qwen3"):
            models.append("ollama/qwen3:4b")
        if os.environ.get("OPENAI_API_KEY"):
            models.append("openai/gpt-4o-mini")
        if os.environ.get("ANTHROPIC_API_KEY"):
            models.append("anthropic/claude-haiku-4-5-20251001")
        if os.environ.get("GOOGLE_API_KEY"):
            models.append("google/gemini-2.5-flash")
        return models

    def test_all_providers_agree_on_math(self):
        """Every available provider should answer '4' to 2+2."""
        models = self._available_models()
        if len(models) < 2:
            pytest.skip("Need at least 2 providers for cross-provider test")

        llm = UnifiedLLM()
        try:
            for model_id in models:
                provider = model_id.split("/")[0]
                llm.add_provider(provider)

            for model_id in models:
                resp = llm.complete(model_id, PROMPT, CONFIG)
                _assert_valid_response(resp, model_id.split("/")[0])
                _assert_answer_is_4(resp)
        finally:
            llm.shutdown()

    def test_all_providers_return_consistent_structure(self):
        """Every provider's CompletionResponse should have the same shape."""
        models = self._available_models()
        if len(models) < 2:
            pytest.skip("Need at least 2 providers for cross-provider test")

        llm = UnifiedLLM()
        try:
            for model_id in models:
                llm.add_provider(model_id.split("/")[0])

            responses = []
            for model_id in models:
                resp = llm.complete(model_id, PROMPT, CONFIG)
                responses.append(resp)

            # All responses should have the same fields populated
            for resp in responses:
                assert resp.content.strip() != ""
                assert resp.usage.total_tokens > 0
                assert resp.latency_ms > 0
                assert resp.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)
        finally:
            llm.shutdown()
