"""
unified-llm — one interface to rule them all.

Profile and benchmark any LLM — local or cloud, stock or custom — through
a single, consistent API.
"""

from unified_llm.client import UnifiedLLM
from unified_llm.providers.base import BaseProvider
from unified_llm.registry import register_provider
from unified_llm.types import (
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
)

__all__ = [
    # Core
    "UnifiedLLM",
    "BaseProvider",
    "register_provider",
    # Types
    "GenerationConfig",
    "Message",
    "CompletionResponse",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "ModelInfo",
    "ProviderType",
    "ModelLocation",
    "FinishReason",
    # Perf
    "PerfMetrics",
]
