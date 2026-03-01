"""Core types and data models for the unified LLM interface."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    LITELLM = "litellm"
    VLLM = "vllm"
    HUGGINGFACE = "huggingface"
    LLAMACPP = "llamacpp"
    CUSTOM = "custom"


class ModelLocation(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    """A tool/function the model can call."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------

class GenerationConfig(BaseModel):
    """Provider-agnostic generation parameters."""
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] = Field(default_factory=list)
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None
    tools: list[ToolDefinition] = Field(default_factory=list)
    response_format: Literal["text", "json"] = "text"
    stream: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # JSON string


class CompletionResponse(BaseModel):
    """Unified response from any provider."""
    id: str = ""
    model: str = ""
    provider: str = ""
    content: str = ""
    finish_reason: FinishReason = FinishReason.STOP
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamChunk(BaseModel):
    """A single chunk from a streaming response."""
    delta: str = ""
    finish_reason: FinishReason | None = None
    usage: TokenUsage | None = None


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    """Metadata about a registered model."""
    name: str
    provider: ProviderType
    location: ModelLocation
    context_window: int = 0
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    cost_per_input_token: float | None = None   # USD
    cost_per_output_token: float | None = None
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Performance types
# ---------------------------------------------------------------------------

@dataclass
class PerfMetrics:
    """Performance metrics from a perf run."""
    requests_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    total_tokens: int = 0
    total_requests: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
