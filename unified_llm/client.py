"""
UnifiedLLM — the single interface for all LLM providers.

Used internally by the perf profiling module.
"""

from typing import Any

from unified_llm.providers.base import BaseProvider
from unified_llm.registry import create_provider, register_provider
from unified_llm.types import (
    CompletionResponse,
    GenerationConfig,
    Message,
    ModelInfo,
)


class UnifiedLLM:
    """
    Single interface to call any LLM — local or cloud, stock or custom.

    Models are addressed as "provider/model", e.g.:
        - "openai/gpt-4o"
        - "anthropic/claude-sonnet-4-20250514"
        - "ollama/llama3.1"
        - "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct"
        - "my_custom/my-model"
    """

    def __init__(self):
        self._providers: dict[str, BaseProvider] = {}
        self._default_config = GenerationConfig()

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def add_provider(self, name: str, *, provider: BaseProvider | None = None, **kwargs) -> None:
        """
        Add a provider by name (uses built-in registry) or pass an instance directly.

        Examples:
            llm.add_provider("openai")
            llm.add_provider("openai", api_key="sk-...")  # explicit key
            llm.add_provider("local_openai", provider=LiteLLMProvider(provider_name="openai", api_base="http://localhost:8000/v1"))
            llm.add_provider("my_custom", provider=my_custom_instance)
        """
        if provider is not None:
            provider.initialize(**kwargs)
            self._providers[name.lower()] = provider
        else:
            self._providers[name.lower()] = create_provider(name, **kwargs)

    def remove_provider(self, name: str) -> None:
        key = name.lower()
        if key in self._providers:
            self._providers[key].shutdown()
            del self._providers[key]

    def get_provider(self, name: str) -> BaseProvider:
        key = name.lower()
        if key not in self._providers:
            raise KeyError(f"Provider '{name}' not found. Added: {list(self._providers.keys())}")
        return self._providers[key]

    @staticmethod
    def register_custom_provider(name: str, cls: type) -> None:
        """Register a custom provider class globally so it can be used by name."""
        register_provider(name, cls)

    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------

    def _resolve(self, model_id: str) -> tuple[BaseProvider, str]:
        """Parse 'provider/model' → (provider_instance, model_name)."""
        if "/" not in model_id:
            raise ValueError(
                f"Model ID must be 'provider/model', got '{model_id}'. "
                f"Example: 'openai/gpt-4o' or 'ollama/llama3.1'"
            )
        provider_name, *model_parts = model_id.split("/", 1)
        model_name = model_parts[0] if model_parts else None
        return self.get_provider(provider_name), model_name

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def complete(
        self,
        model_id: str,
        prompt: str | list[Message],
        config: GenerationConfig | None = None,
        system: str | None = None,
    ) -> CompletionResponse:
        """
        Generate a completion.

        Args:
            model_id: "provider/model" string
            prompt: A string (becomes a user message) or list of Message objects
            config: Generation parameters (uses defaults if omitted)
            system: Optional system message prepended to the conversation
        """
        provider, model = self._resolve(model_id)
        messages = self._normalize_messages(prompt, system)
        return provider.complete(messages, config or self._default_config, model)

    async def acomplete(
        self,
        model_id: str,
        prompt: str | list[Message],
        config: GenerationConfig | None = None,
        system: str | None = None,
    ) -> CompletionResponse:
        provider, model = self._resolve(model_id)
        messages = self._normalize_messages(prompt, system)
        return await provider.acomplete(messages, config or self._default_config, model)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_models(self) -> dict[str, list[ModelInfo]]:
        """List all available models grouped by provider."""
        result = {}
        for name, provider in self._providers.items():
            try:
                result[name] = provider.list_models()
            except Exception:
                result[name] = []
        return result

    @property
    def providers(self) -> list[str]:
        return list(self._providers.keys())

    @property
    def default_config(self) -> GenerationConfig:
        return self._default_config

    @default_config.setter
    def default_config(self, config: GenerationConfig):
        self._default_config = config

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_messages(prompt: str | list[Message], system: str | None = None) -> list[Message]:
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        if isinstance(prompt, str):
            messages.append(Message(role="user", content=prompt))
        else:
            messages.extend(prompt)
        return messages

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()

    def shutdown(self):
        for provider in self._providers.values():
            provider.shutdown()
        self._providers.clear()
