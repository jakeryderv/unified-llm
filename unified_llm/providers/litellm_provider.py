"""LiteLLM-backed provider — single implementation for all cloud API providers."""

import uuid
from typing import AsyncIterator, Iterator

from unified_llm.providers.base import BaseProvider
from unified_llm.types import (
    CompletionResponse,
    FinishReason,
    GenerationConfig,
    Message,
    ModelInfo,
    ModelLocation,
    ProviderType,
    StreamChunk,
    TokenUsage,
)

# Prefix map: how each logical provider name maps to a LiteLLM model prefix.
# OpenAI models are passed bare (no prefix); others need "provider/" prepended.
_LITELLM_PREFIX_MAP: dict[str, str] = {
    "openai": "",
    "anthropic": "anthropic/",
    "google": "gemini/",
    "ollama": "ollama/",
}


class LiteLLMProvider(BaseProvider):
    """
    Provider that delegates to LiteLLM for all API calls.

    A separate instance is created per logical provider name (openai, anthropic, etc.)
    so that each carries the correct prefix and defaults.

    Args:
        provider_name: Logical name (e.g. "openai", "anthropic").
        litellm_prefix: Override the auto-detected prefix if needed.
        api_key: API key (optional — LiteLLM reads env vars by default).
        api_base: Override the API base URL.
        default_model: Model to use when none is specified.
    """

    _FINISH_REASON_MAP = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALL,
        "function_call": FinishReason.TOOL_CALL,
        "content_filter": FinishReason.STOP,
    }

    def __init__(
        self,
        provider_name: str = "openai",
        litellm_prefix: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "",
        **kwargs,
    ):
        self._provider_name = provider_name
        self._litellm_prefix = (
            litellm_prefix
            if litellm_prefix is not None
            else _LITELLM_PREFIX_MAP.get(provider_name, f"{provider_name}/")
        )
        self._api_key = api_key
        self._api_base = api_base
        self.default_model = default_model
        self._extra_kwargs = kwargs

    def _litellm_model(self, model: str) -> str:
        """Prepend the LiteLLM prefix to a bare model name."""
        if self._litellm_prefix and not model.startswith(self._litellm_prefix):
            return f"{self._litellm_prefix}{model}"
        return model

    def _base_kwargs(self, model: str, config: GenerationConfig) -> dict:
        """Build the kwargs dict for ``litellm.completion()``."""
        kwargs: dict = {
            "model": self._litellm_model(model),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "stop": config.stop_sequences or None,
            "seed": config.seed,
            "drop_params": True,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if config.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        if config.extra:
            kwargs.update(config.extra)
        return kwargs

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def complete(self, messages, config=None, model=None) -> CompletionResponse:
        import litellm

        config, model = self._resolve_config(config, model)
        kwargs = self._base_kwargs(model, config)
        kwargs["messages"] = [{"role": m.role, "content": m.content} for m in messages]

        with self._timed() as elapsed:
            response = litellm.completion(**kwargs)

        choice = response.choices[0]
        usage = response.usage

        return CompletionResponse(
            id=response.id or str(uuid.uuid4()),
            model=response.model or model,
            provider=self._provider_name,
            content=choice.message.content or "",
            finish_reason=self._map_finish_reason(choice.finish_reason, self._FINISH_REASON_MAP),
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            latency_ms=elapsed(),
        )

    def stream(self, messages, config=None, model=None) -> Iterator[StreamChunk]:
        import litellm

        config, model = self._resolve_config(config, model)
        kwargs = self._base_kwargs(model, config)
        kwargs["messages"] = [{"role": m.role, "content": m.content} for m in messages]
        kwargs["stream"] = True

        response = litellm.completion(**kwargs)

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(
                    delta=chunk.choices[0].delta.content,
                    finish_reason=(
                        self._map_finish_reason(chunk.choices[0].finish_reason, self._FINISH_REASON_MAP)
                        if chunk.choices[0].finish_reason
                        else None
                    ),
                )

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

    async def acomplete(self, messages, config=None, model=None) -> CompletionResponse:
        import litellm

        config, model = self._resolve_config(config, model)
        kwargs = self._base_kwargs(model, config)
        kwargs["messages"] = [{"role": m.role, "content": m.content} for m in messages]

        with self._timed() as elapsed:
            response = await litellm.acompletion(**kwargs)

        choice = response.choices[0]
        usage = response.usage

        return CompletionResponse(
            id=response.id or str(uuid.uuid4()),
            model=response.model or model,
            provider=self._provider_name,
            content=choice.message.content or "",
            finish_reason=self._map_finish_reason(choice.finish_reason, self._FINISH_REASON_MAP),
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            latency_ms=elapsed(),
        )

    async def astream(self, messages, config=None, model=None) -> AsyncIterator[StreamChunk]:
        import litellm

        config, model = self._resolve_config(config, model)
        kwargs = self._base_kwargs(model, config)
        kwargs["messages"] = [{"role": m.role, "content": m.content} for m in messages]
        kwargs["stream"] = True

        response = await litellm.acompletion(**kwargs)

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(
                    delta=chunk.choices[0].delta.content,
                    finish_reason=(
                        self._map_finish_reason(chunk.choices[0].finish_reason, self._FINISH_REASON_MAP)
                        if chunk.choices[0].finish_reason
                        else None
                    ),
                )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        try:
            import litellm

            provider_models = litellm.models_by_provider.get(self._provider_name, [])
            return [
                ModelInfo(
                    name=m,
                    provider=ProviderType.LITELLM,
                    location=ModelLocation.LOCAL if self._provider_name == "ollama" else ModelLocation.CLOUD,
                )
                for m in provider_models
            ]
        except Exception:
            return []

    @property
    def provider_name(self) -> str:
        return self._provider_name
