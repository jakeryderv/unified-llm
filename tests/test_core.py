"""Smoke tests and echo-provider integration tests for unified-llm."""

import pytest

from unified_llm import (
    BaseProvider,
    CompletionResponse,
    FinishReason,
    GenerationConfig,
    Message,
    ModelInfo,
    ModelLocation,
    PerfMetrics,
    ProviderType,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    UnifiedLLM,
    register_provider,
)


# ---------------------------------------------------------------------------
# Smoke: subpackage imports
# ---------------------------------------------------------------------------


class TestImports:
    def test_perf_imports(self):
        from unified_llm.perf import (
            PerfConfig,
            PerfReport,
            PerfRunner,
            CustomPerf,
        )

    def test_optimization_imports(self):
        from unified_llm.optimization import (
            PRESET_PROFILES,
            LoRAConfig,
            OptimizationProfile,
            QuantMethod,
            QuantizationConfig,
            apply_profile_to_provider_kwargs,
            get_profile,
        )

    def test_providers_imports(self):
        from unified_llm.providers import BaseProvider


# ---------------------------------------------------------------------------
# Echo provider (self-contained test fixture)
# ---------------------------------------------------------------------------


class EchoProvider(BaseProvider):
    """Minimal provider that echoes input — no external deps."""

    default_model = "echo"

    def complete(self, messages, config=None, model=None):
        config, model = self._resolve_config(config, model)
        last = messages[-1].content if messages else ""
        word_count = len(last.split())
        return CompletionResponse(
            model=model,
            provider="echo",
            content=f"Echo: {last}",
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(
                prompt_tokens=word_count,
                completion_tokens=word_count,
                total_tokens=word_count * 2,
            ),
            latency_ms=0.1,
        )

    async def acomplete(self, messages, config=None, model=None):
        return self.complete(messages, config, model)

    def list_models(self):
        return [
            ModelInfo(
                name="echo",
                provider=ProviderType.CUSTOM,
                location=ModelLocation.LOCAL,
            )
        ]


@pytest.fixture()
def llm():
    """Return a UnifiedLLM wired up with the echo provider."""
    client = UnifiedLLM()
    client.add_provider("echo", provider=EchoProvider())
    yield client
    client.shutdown()


# ---------------------------------------------------------------------------
# Integration: echo provider through UnifiedLLM
# ---------------------------------------------------------------------------


class TestEchoComplete:
    def test_simple_complete(self, llm: UnifiedLLM):
        resp = llm.complete("echo/echo", "hello world")
        assert resp.content == "Echo: hello world"
        assert resp.finish_reason == FinishReason.STOP
        assert resp.provider == "echo"
        assert resp.usage.prompt_tokens == 2
        assert resp.usage.total_tokens == 4

    def test_complete_with_config(self, llm: UnifiedLLM):
        cfg = GenerationConfig(temperature=0, max_tokens=10)
        resp = llm.complete("echo/echo", "test", cfg)
        assert resp.content == "Echo: test"

    def test_complete_with_system(self, llm: UnifiedLLM):
        resp = llm.complete("echo/echo", "ping", system="be helpful")
        # Echo provider always echoes the *last* message, which is the user msg
        assert resp.content == "Echo: ping"

    def test_complete_with_messages(self, llm: UnifiedLLM):
        msgs = [
            Message(role="user", content="first"),
            Message(role="assistant", content="ok"),
            Message(role="user", content="second"),
        ]
        resp = llm.complete("echo/echo", msgs)
        assert resp.content == "Echo: second"


class TestEchoListModels:
    def test_list_models(self, llm: UnifiedLLM):
        models = llm.list_models()
        assert "echo" in models
        assert len(models["echo"]) == 1
        assert models["echo"][0].name == "echo"


class TestLifecycle:
    def test_add_remove_provider(self):
        client = UnifiedLLM()
        client.add_provider("echo", provider=EchoProvider())
        assert "echo" in client.providers
        client.remove_provider("echo")
        assert "echo" not in client.providers

    def test_context_manager(self):
        with UnifiedLLM() as client:
            client.add_provider("echo", provider=EchoProvider())
            resp = client.complete("echo/echo", "ctx")
            assert resp.content == "Echo: ctx"

    def test_register_custom_provider(self):
        register_provider("echo_test", EchoProvider)
        client = UnifiedLLM()
        client.add_provider("echo_test")
        resp = client.complete("echo_test/echo", "registered")
        assert resp.content == "Echo: registered"
        client.shutdown()


class TestErrors:
    def test_missing_provider(self, llm: UnifiedLLM):
        with pytest.raises(KeyError, match="not found"):
            llm.complete("nonexistent/model", "hello")

    def test_bad_model_id(self, llm: UnifiedLLM):
        with pytest.raises(ValueError, match="must be 'provider/model'"):
            llm.complete("no_slash", "hello")


class TestBaseProviderHelpers:
    def test_resolve_config_defaults(self):
        p = EchoProvider()
        config, model = p._resolve_config(None, None)
        assert isinstance(config, GenerationConfig)
        assert model == "echo"

    def test_resolve_config_passthrough(self):
        p = EchoProvider()
        cfg = GenerationConfig(temperature=0.5)
        config, model = p._resolve_config(cfg, "custom-model")
        assert config is cfg
        assert model == "custom-model"

    def test_timed(self):
        with BaseProvider._timed() as elapsed:
            pass
        ms = elapsed()
        assert isinstance(ms, float)
        assert ms >= 0

    def test_map_finish_reason(self):
        mapping = {"stop": FinishReason.STOP, "length": FinishReason.LENGTH}
        assert BaseProvider._map_finish_reason("stop", mapping) == FinishReason.STOP
        assert BaseProvider._map_finish_reason("length", mapping) == FinishReason.LENGTH
        assert BaseProvider._map_finish_reason("unknown", mapping) == FinishReason.STOP
        assert BaseProvider._map_finish_reason(None, mapping) == FinishReason.STOP


class TestConfigEnvSubstitution:
    def test_plain_var(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "secret")
        from unified_llm.config import _substitute_env_vars
        assert _substitute_env_vars("${TEST_KEY}") == "secret"

    def test_default_used_when_unset(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        from unified_llm.config import _substitute_env_vars
        assert _substitute_env_vars("${MISSING_VAR:-fallback}") == "fallback"

    def test_default_ignored_when_set(self, monkeypatch):
        monkeypatch.setenv("PRESENT", "real")
        from unified_llm.config import _substitute_env_vars
        assert _substitute_env_vars("${PRESENT:-fallback}") == "real"

    def test_empty_default(self, monkeypatch):
        monkeypatch.delenv("NOPE", raising=False)
        from unified_llm.config import _substitute_env_vars
        assert _substitute_env_vars("${NOPE:-}") == ""

    def test_missing_var_no_default_raises(self, monkeypatch):
        monkeypatch.delenv("GONE", raising=False)
        from unified_llm.config import _substitute_env_vars
        with pytest.raises(ValueError, match="GONE"):
            _substitute_env_vars("${GONE}")

    def test_multiple_vars_in_string(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.delenv("PORT", raising=False)
        from unified_llm.config import _substitute_env_vars
        result = _substitute_env_vars("http://${HOST}:${PORT:-8080}/api")
        assert result == "http://localhost:8080/api"



