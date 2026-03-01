"""
Optimization utilities for local models.

Provides helpers for:
    - Quantization (GGUF, GPTQ, AWQ, bitsandbytes)
    - LoRA / QLoRA fine-tuning preparation
    - Model export and conversion
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "QuantMethod",
    "QuantizationConfig",
    "LoRAConfig",
    "OptimizationProfile",
    "PRESET_PROFILES",
    "get_profile",
    "apply_profile_to_provider_kwargs",
]


class QuantMethod(str, Enum):
    NONE = "none"
    BNB_4BIT = "bnb_4bit"
    BNB_8BIT = "bnb_8bit"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF = "gguf"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    method: QuantMethod = QuantMethod.NONE
    bits: int = 4
    group_size: int = 128
    double_quant: bool = True
    quant_type: str = "nf4"  # nf4 or fp4 for bitsandbytes
    compute_dtype: str = "bfloat16"
    # GGUF-specific
    gguf_type: str = "Q4_K_M"


@dataclass
class LoRAConfig:
    """Configuration for LoRA / QLoRA fine-tuning."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    # QLoRA
    use_qlora: bool = False
    qlora_bits: int = 4


@dataclass
class OptimizationProfile:
    """
    A named optimization profile combining quantization + other settings.

    Profiles can be saved/loaded from YAML configs.
    """
    name: str = "default"
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig | None = None
    use_flash_attention: bool = True
    use_bettertransformer: bool = False
    torch_compile: bool = False
    device_map: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Preset profiles
# ---------------------------------------------------------------------------

PRESET_PROFILES: dict[str, OptimizationProfile] = {
    "full_precision": OptimizationProfile(
        name="full_precision",
        quantization=QuantizationConfig(method=QuantMethod.NONE),
    ),
    "fast_4bit": OptimizationProfile(
        name="fast_4bit",
        quantization=QuantizationConfig(
            method=QuantMethod.BNB_4BIT,
            bits=4,
            double_quant=True,
            quant_type="nf4",
        ),
        use_flash_attention=True,
    ),
    "balanced_8bit": OptimizationProfile(
        name="balanced_8bit",
        quantization=QuantizationConfig(method=QuantMethod.BNB_8BIT, bits=8),
        use_flash_attention=True,
    ),
    "qlora_finetune": OptimizationProfile(
        name="qlora_finetune",
        quantization=QuantizationConfig(method=QuantMethod.BNB_4BIT, bits=4),
        lora=LoRAConfig(use_qlora=True, qlora_bits=4),
        use_flash_attention=True,
    ),
    "lora_finetune": OptimizationProfile(
        name="lora_finetune",
        quantization=QuantizationConfig(method=QuantMethod.NONE),
        lora=LoRAConfig(use_qlora=False),
        use_flash_attention=True,
    ),
}


def get_profile(name: str) -> OptimizationProfile:
    """Get a preset optimization profile by name."""
    if name not in PRESET_PROFILES:
        available = list(PRESET_PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return PRESET_PROFILES[name]


def apply_profile_to_provider_kwargs(profile: OptimizationProfile) -> dict[str, Any]:
    """
    Convert an OptimizationProfile into kwargs for HuggingFaceProvider.

    Usage:
        profile = get_profile("fast_4bit")
        kwargs = apply_profile_to_provider_kwargs(profile)
        llm.add_provider("hf", provider=HuggingFaceProvider(model_name_or_path="...", **kwargs))
    """
    kwargs: dict[str, Any] = {
        "device_map": profile.device_map,
        "use_flash_attention": profile.use_flash_attention,
    }

    q = profile.quantization
    if q.method == QuantMethod.BNB_4BIT:
        kwargs["load_in_4bit"] = True
    elif q.method == QuantMethod.BNB_8BIT:
        kwargs["load_in_8bit"] = True

    if profile.lora and profile.lora.use_qlora:
        kwargs["load_in_4bit"] = True

    return kwargs
