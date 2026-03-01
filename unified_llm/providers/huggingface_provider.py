"""HuggingFace Transformers provider for local model inference."""

import uuid
from typing import Iterator

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


class HuggingFaceProvider(BaseProvider):
    """
    Provider for local HuggingFace Transformers models.

    Supports quantization (4-bit/8-bit via bitsandbytes), Flash Attention,
    and LoRA adapter loading.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        device_map: Device mapping strategy (default: "auto").
        load_in_4bit: Enable 4-bit quantization via bitsandbytes.
        load_in_8bit: Enable 8-bit quantization via bitsandbytes.
        use_flash_attention: Enable Flash Attention 2.
        lora_adapter_path: Path to a LoRA adapter to merge.
        torch_dtype: Torch dtype string (default: "auto").
    """

    def __init__(
        self,
        model_name_or_path: str = "",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_flash_attention: bool = False,
        lora_adapter_path: str | None = None,
        torch_dtype: str = "auto",
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.use_flash_attention = use_flash_attention
        self.lora_adapter_path = lora_adapter_path
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None

    def initialize(self, **kwargs) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "HuggingFace provider requires transformers and torch: "
                "pip install transformers torch"
            )

        if not self.model_name_or_path:
            return  # Will be loaded on first use

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "device_map": self.device_map,
        }

        # Torch dtype
        if self.torch_dtype == "auto":
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = getattr(torch, self.torch_dtype, torch.float16)

        # Quantization
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                raise ImportError("4-bit quantization requires bitsandbytes: pip install bitsandbytes")
        elif self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        # Flash Attention
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **model_kwargs)

        # LoRA adapter
        if self.lora_adapter_path:
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, self.lora_adapter_path)
                model = model.merge_and_unload()
            except ImportError:
                raise ImportError("LoRA support requires peft: pip install peft")

        self._model = model
        self._tokenizer = tokenizer

    def complete(self, messages, config=None, model=None) -> CompletionResponse:
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call initialize() or provide model_name_or_path.")

        config = config or GenerationConfig()

        prompt = self._format_chat(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        with self._timed() as elapsed:
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=max(config.temperature, 0.01),
                    top_p=config.top_p,
                    top_k=config.top_k or 50,
                    do_sample=config.temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

        # Decode only the new tokens
        new_tokens = outputs[0][prompt_tokens:]
        content = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        completion_tokens = len(new_tokens)

        return CompletionResponse(
            id=str(uuid.uuid4()),
            model=model or self.model_name_or_path,
            provider="huggingface",
            content=content,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_ms=elapsed(),
        )

    async def acomplete(self, messages, config=None, model=None) -> CompletionResponse:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self.complete, messages, config, model
        )

    def stream(self, messages, config=None, model=None) -> Iterator[StreamChunk]:
        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded.")

        config = config or GenerationConfig()
        prompt = self._format_chat(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(self._tokenizer, skip_special_tokens=True, skip_prompt=True)

        generation_kwargs = {
            **inputs,
            "max_new_tokens": config.max_tokens,
            "temperature": max(config.temperature, 0.01),
            "do_sample": config.temperature > 0,
            "streamer": streamer,
            "pad_token_id": self._tokenizer.pad_token_id,
        }

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            if text:
                yield StreamChunk(delta=text)

        thread.join()

    def list_models(self) -> list[ModelInfo]:
        if self.model_name_or_path:
            return [
                ModelInfo(
                    name=self.model_name_or_path,
                    provider=ProviderType.HUGGINGFACE,
                    location=ModelLocation.LOCAL,
                    supports_streaming=True,
                )
            ]
        return []

    def _format_chat(self, messages: list[Message]) -> str:
        """Format messages using the tokenizer's chat template if available."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            chat = [{"role": m.role, "content": m.content} for m in messages]
            return self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # Fallback: simple concatenation
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(f"<<SYS>>{m.content}<</SYS>>")
            elif m.role == "user":
                parts.append(f"[INST] {m.content} [/INST]")
            else:
                parts.append(m.content)
        return "\n".join(parts)

    def shutdown(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # Free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
