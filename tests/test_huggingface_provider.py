"""Tests for HuggingFaceProvider with mocked transformers."""

from unittest.mock import MagicMock, patch

import pytest
import torch as real_torch

from unified_llm.types import (
    CompletionResponse,
    FinishReason,
    GenerationConfig,
    Message,
    ModelInfo,
    ModelLocation,
    ProviderType,
)


@pytest.fixture()
def mock_tokenizer():
    """Provide a mocked AutoTokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 0

    # apply_chat_template returns a formatted string
    tokenizer.apply_chat_template.return_value = "[INST] hello [/INST]"

    # tokenizer() returns input_ids tensor
    input_ids = MagicMock()
    input_ids.shape = [1, 5]  # batch=1, seq_len=5
    inputs = {"input_ids": input_ids}
    inputs_on_device = MagicMock()
    inputs_on_device.__getitem__ = lambda self, k: inputs[k]
    inputs_on_device.keys = lambda: inputs.keys()
    inputs_on_device.__iter__ = lambda self: iter(inputs)

    tokenizer.return_value.to.return_value = inputs_on_device

    # decode returns generated text
    tokenizer.decode.return_value = "Hello world response"

    return tokenizer


@pytest.fixture()
def mock_model():
    """Provide a mocked AutoModelForCausalLM."""
    model = MagicMock()
    model.device = "cpu"

    # generate() returns tensor of shape [1, prompt_len + gen_len]
    output_tensor = real_torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    model.generate.return_value = output_tensor

    return model


class TestHuggingFaceProvider:
    def test_init_defaults(self):
        from unified_llm.providers.huggingface_provider import HuggingFaceProvider

        p = HuggingFaceProvider(model_name_or_path="test/model")
        assert p.model_name_or_path == "test/model"
        assert p.device_map == "auto"
        assert p.load_in_4bit is False
        assert p.load_in_8bit is False
        assert p.use_flash_attention is False
        assert p.lora_adapter_path is None
        assert p.torch_dtype == "auto"
        assert p._model is None
        assert p._tokenizer is None

    def test_init_quantization_flags(self):
        from unified_llm.providers.huggingface_provider import HuggingFaceProvider

        p = HuggingFaceProvider(
            model_name_or_path="test/model",
            load_in_4bit=True,
            use_flash_attention=True,
            lora_adapter_path="/path/to/adapter",
        )
        assert p.load_in_4bit is True
        assert p.use_flash_attention is True
        assert p.lora_adapter_path == "/path/to/adapter"

    def test_list_models(self):
        from unified_llm.providers.huggingface_provider import HuggingFaceProvider

        p = HuggingFaceProvider(model_name_or_path="meta-llama/Llama-3.1-8B")
        models = p.list_models()
        assert len(models) == 1
        assert models[0].name == "meta-llama/Llama-3.1-8B"
        assert models[0].provider == ProviderType.HUGGINGFACE
        assert models[0].location == ModelLocation.LOCAL

    def test_list_models_empty(self):
        from unified_llm.providers.huggingface_provider import HuggingFaceProvider

        p = HuggingFaceProvider()
        assert p.list_models() == []

    def test_complete_requires_loaded_model(self):
        from unified_llm.providers.huggingface_provider import HuggingFaceProvider

        p = HuggingFaceProvider(model_name_or_path="test/model")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            p.complete([Message(role="user", content="hi")])

    @patch("unified_llm.providers.huggingface_provider.uuid.uuid4", return_value="test-uuid")
    def test_complete_with_mocked_model(self, _uuid, mock_tokenizer, mock_model):
        from unified_llm.providers.huggingface_provider import HuggingFaceProvider

        p = HuggingFaceProvider(model_name_or_path="test/model")
        p._model = mock_model
        p._tokenizer = mock_tokenizer

        messages = [Message(role="user", content="hello")]
        config = GenerationConfig(temperature=0.5, max_tokens=100)

        # Real torch is used — no need to mock it since it's imported inside complete()
        resp = p.complete(messages, config)

        assert isinstance(resp, CompletionResponse)
        assert resp.provider == "huggingface"
        assert resp.content == "Hello world response"
        assert resp.finish_reason == FinishReason.STOP
        assert resp.id == "test-uuid"

        # Verify generate was called with expected kwargs
        mock_model.generate.assert_called_once()
        call_kwargs = mock_model.generate.call_args
        assert call_kwargs.kwargs["max_new_tokens"] == 100
        assert call_kwargs.kwargs["do_sample"] is True  # temperature > 0

    @patch("unified_llm.providers.huggingface_provider.uuid.uuid4", return_value="test-uuid")
    def test_complete_no_sample_at_zero_temp(self, _uuid, mock_tokenizer, mock_model):
        from unified_llm.providers.huggingface_provider import HuggingFaceProvider

        p = HuggingFaceProvider(model_name_or_path="test/model")
        p._model = mock_model
        p._tokenizer = mock_tokenizer

        config = GenerationConfig(temperature=0.0, max_tokens=50)

        resp = p.complete([Message(role="user", content="test")], config)

        call_kwargs = mock_model.generate.call_args.kwargs
        assert call_kwargs["do_sample"] is False

    def test_format_chat_with_template(self, mock_tokenizer):
        from unified_llm.providers.huggingface_provider import HuggingFaceProvider

        p = HuggingFaceProvider(model_name_or_path="test/model")
        p._tokenizer = mock_tokenizer

        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="hello"),
        ]
        result = p._format_chat(messages)

        mock_tokenizer.apply_chat_template.assert_called_once()
        assert result == "[INST] hello [/INST]"

    def test_format_chat_fallback(self):
        from unified_llm.providers.huggingface_provider import HuggingFaceProvider

        p = HuggingFaceProvider(model_name_or_path="test/model")
        # Tokenizer without apply_chat_template
        tokenizer = MagicMock(spec=[])  # empty spec = no attributes
        p._tokenizer = tokenizer

        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="hello"),
        ]
        result = p._format_chat(messages)
        assert "<<SYS>>Be helpful<</SYS>>" in result
        assert "[INST] hello [/INST]" in result

    def test_shutdown_clears_model(self, mock_tokenizer, mock_model):
        from unified_llm.providers.huggingface_provider import HuggingFaceProvider

        p = HuggingFaceProvider(model_name_or_path="test/model")
        p._model = mock_model
        p._tokenizer = mock_tokenizer

        p.shutdown()

        assert p._model is None
        assert p._tokenizer is None

    def test_initialize_basic(self, mock_tokenizer, mock_model):
        from unified_llm.providers.huggingface_provider import HuggingFaceProvider

        with patch("transformers.AutoTokenizer") as MockTokenizer, \
             patch("transformers.AutoModelForCausalLM") as MockModel:
            MockTokenizer.from_pretrained.return_value = mock_tokenizer
            MockModel.from_pretrained.return_value = mock_model

            p = HuggingFaceProvider(model_name_or_path="test/model")
            p.initialize()

            MockTokenizer.from_pretrained.assert_called_once_with("test/model")
            MockModel.from_pretrained.assert_called_once()
            assert p._model is mock_model
            assert p._tokenizer is mock_tokenizer

    def test_initialize_with_4bit(self, mock_tokenizer, mock_model):
        from unified_llm.providers.huggingface_provider import HuggingFaceProvider

        with patch("transformers.AutoTokenizer") as MockTokenizer, \
             patch("transformers.AutoModelForCausalLM") as MockModel, \
             patch("transformers.BitsAndBytesConfig") as MockBnB:
            MockTokenizer.from_pretrained.return_value = mock_tokenizer
            MockModel.from_pretrained.return_value = mock_model
            MockBnB.return_value = "bnb_config"

            p = HuggingFaceProvider(model_name_or_path="test/model", load_in_4bit=True)
            p.initialize()

            MockBnB.assert_called_once()
            call_kwargs = MockModel.from_pretrained.call_args.kwargs
            assert call_kwargs["quantization_config"] == "bnb_config"

    def test_initialize_with_flash_attention(self, mock_tokenizer, mock_model):
        from unified_llm.providers.huggingface_provider import HuggingFaceProvider

        with patch("transformers.AutoTokenizer") as MockTokenizer, \
             patch("transformers.AutoModelForCausalLM") as MockModel:
            MockTokenizer.from_pretrained.return_value = mock_tokenizer
            MockModel.from_pretrained.return_value = mock_model

            p = HuggingFaceProvider(model_name_or_path="test/model", use_flash_attention=True)
            p.initialize()

            call_kwargs = MockModel.from_pretrained.call_args.kwargs
            assert call_kwargs["attn_implementation"] == "flash_attention_2"

    def test_initialize_empty_model_path(self):
        from unified_llm.providers.huggingface_provider import HuggingFaceProvider

        p = HuggingFaceProvider()
        p.initialize()  # Should not raise
        assert p._model is None
