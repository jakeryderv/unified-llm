"""
Provider implementations.

Providers are lazily imported to avoid pulling in heavy dependencies
(torch, transformers, etc.) unless actually needed.
"""

from unified_llm.providers.base import BaseProvider


def __getattr__(name: str):
    """Lazy-import providers on first access."""
    _providers = {
        "LiteLLMProvider": "unified_llm.providers.litellm_provider",
        "HuggingFaceProvider": "unified_llm.providers.huggingface_provider",
    }
    if name in _providers:
        import importlib
        mod = importlib.import_module(_providers[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseProvider",
    "LiteLLMProvider",
    "HuggingFaceProvider",
]
