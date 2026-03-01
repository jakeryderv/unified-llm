"""Provider registry — maps provider names to their implementation classes."""

from typing import Any

from unified_llm.providers.base import BaseProvider

# Global registry: name → provider class
_REGISTRY: dict[str, type[BaseProvider]] = {}


def register_provider(name: str, cls: type[BaseProvider]) -> None:
    """Register a provider class under *name* so it can be created by name."""
    _REGISTRY[name.lower()] = cls


def create_provider(name: str, **kwargs: Any) -> BaseProvider:
    """
    Create and initialize a provider instance by name.

    Falls back to lazy-importing built-in providers if not explicitly registered.
    """
    key = name.lower()

    # Try explicit registry first
    if key in _REGISTRY:
        provider = _REGISTRY[key](**kwargs)
        provider.initialize(**kwargs)
        return provider

    # Lazy-import built-in providers
    result = _resolve_builtin(key)
    if result is not None:
        cls, default_kwargs = result
        merged = {**default_kwargs, **kwargs}
        provider = cls(**merged)
        provider.initialize(**merged)
        return provider

    available = list(_REGISTRY.keys()) + list(_BUILTIN_MAP.keys())
    raise KeyError(
        f"Unknown provider '{name}'. "
        f"Available: {sorted(set(available))}. "
        f"Use register_provider() to add custom providers."
    )


# ---------------------------------------------------------------------------
# Built-in provider map (lazy imports to avoid heavy deps at startup)
# ---------------------------------------------------------------------------

_LITELLM = ("unified_llm.providers.litellm_provider", "LiteLLMProvider")

_BUILTIN_MAP: dict[str, tuple[tuple[str, str], dict[str, Any]]] = {
    "openai": (_LITELLM, {"provider_name": "openai", "default_model": "gpt-4o"}),
    "anthropic": (_LITELLM, {"provider_name": "anthropic", "default_model": "claude-sonnet-4-20250514"}),
    "google": (_LITELLM, {"provider_name": "google", "default_model": "gemini-2.5-flash"}),
    "ollama": (_LITELLM, {"provider_name": "ollama", "default_model": "llama3.1"}),
    "huggingface": (("unified_llm.providers.huggingface_provider", "HuggingFaceProvider"), {}),
    "hf": (("unified_llm.providers.huggingface_provider", "HuggingFaceProvider"), {}),
}


def _resolve_builtin(name: str) -> tuple[type[BaseProvider], dict[str, Any]] | None:
    if name not in _BUILTIN_MAP:
        return None
    (module_path, class_name), default_kwargs = _BUILTIN_MAP[name]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls, default_kwargs


def list_registered() -> list[str]:
    """Return all registered + built-in provider names."""
    return sorted(set(list(_REGISTRY.keys()) + list(_BUILTIN_MAP.keys())))
