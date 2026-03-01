"""Abstract base class for all LLM providers."""

import abc
import contextlib
import time
from typing import AsyncIterator, Iterator

from unified_llm.types import (
    CompletionResponse,
    FinishReason,
    GenerationConfig,
    Message,
    ModelInfo,
    StreamChunk,
)


class BaseProvider(abc.ABC):
    """
    Every provider (cloud or local) implements this interface.

    To add a new provider:
        1. Subclass BaseProvider
        2. Implement the abstract methods
        3. Register it via the ProviderRegistry or YAML config
    """

    default_model: str = ""

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def complete(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
        model: str | None = None,
    ) -> CompletionResponse:
        """Synchronous completion."""
        ...

    @abc.abstractmethod
    async def acomplete(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
        model: str | None = None,
    ) -> CompletionResponse:
        """Async completion."""
        ...

    @abc.abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """Return models available through this provider."""
        ...

    # ------------------------------------------------------------------
    # Optional streaming (default raises)
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
        model: str | None = None,
    ) -> Iterator[StreamChunk]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming")

    async def astream(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support async streaming")
        yield  # makes this an async generator  # noqa: E501

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, **kwargs) -> None:
        """Called once when the provider is first loaded. Override for setup."""
        pass

    def shutdown(self) -> None:
        """Called when the provider is unloaded. Override for cleanup."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__

    def _resolve_config(
        self,
        config: GenerationConfig | None,
        model: str | None,
    ) -> tuple[GenerationConfig, str]:
        """Return (config, model) with defaults filled in."""
        return config or GenerationConfig(), model or self.default_model

    @staticmethod
    @contextlib.contextmanager
    def _timed():
        """Context manager that yields an ``elapsed`` callable returning ms."""
        t0 = time.perf_counter()
        yield lambda: (time.perf_counter() - t0) * 1000

    @staticmethod
    def _map_finish_reason(
        reason: str | None,
        mapping: dict[str, FinishReason],
    ) -> FinishReason:
        """Look up *reason* in a provider-specific mapping, defaulting to STOP."""
        return mapping.get(reason, FinishReason.STOP)
