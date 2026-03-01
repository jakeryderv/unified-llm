"""YAML configuration loader with environment variable substitution."""

import os
import re
from pathlib import Path
from typing import Any

from unified_llm.types import GenerationConfig


def _substitute_env_vars(value: str) -> str:
    """Replace ``${VAR}`` and ``${VAR:-default}`` patterns with environment values.

    If a default is provided via ``:-``, it is used when the variable is unset.
    Without a default, a missing variable raises ``ValueError``.
    """
    def _replace(match):
        var_name = match.group(1)
        default = match.group(2)          # None when no ``:-`` was given
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        if default is not None:
            return default
        raise ValueError(
            f"Environment variable '{var_name}' not set. "
            f"Set it, provide a default (${{{{VAR:-default}}}}), "
            f"or remove the ${{{{...}}}} reference from your config."
        )
    return re.sub(r"\$\{(\w+)(?::-(.*?))?\}", _replace, value)


def _process_values(obj: Any) -> Any:
    """Recursively substitute env vars in strings throughout a config dict."""
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _process_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_process_values(v) for v in obj]
    return obj


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and parse a YAML config file with env var substitution."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config loading: pip install pyyaml")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return _process_values(raw or {})


def create_client_from_config(path: str | Path) -> "UnifiedLLM":
    """
    Create a fully-configured UnifiedLLM client from a YAML config file.

    Config format:
        defaults:
          temperature: 0.7
          max_tokens: 1024
        providers:
          openai:
            default_model: gpt-4o
          ollama:
            api_base: ${OLLAMA_HOST:-http://localhost:11434}
    """
    from unified_llm.client import UnifiedLLM

    config = load_config(path)
    client = UnifiedLLM()

    # Apply defaults
    defaults = config.get("defaults", {})
    if defaults:
        client.default_config = GenerationConfig(**defaults)

    # Add providers
    providers = config.get("providers", {})
    for name, provider_cfg in providers.items():
        provider_cfg = provider_cfg or {}
        # Backward compat: Ollama configs may use "host" instead of "api_base"
        if "host" in provider_cfg and "api_base" not in provider_cfg:
            provider_cfg["api_base"] = provider_cfg.pop("host")
        provider_type = provider_cfg.pop("type", name)

        # If 'type' was specified, use that as the registry key
        # but register under the config key name
        if provider_type != name:
            from unified_llm.registry import create_provider
            provider = create_provider(provider_type, **provider_cfg)
            client.add_provider(name, provider=provider)
        else:
            client.add_provider(name, **provider_cfg)

    return client
