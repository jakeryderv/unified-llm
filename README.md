# unified-llm

**One interface to rule them all.** Profile and benchmark any LLM — local or cloud — through a single, consistent API.

## Features

- **Unified interface** — `provider/model` addressing for any backend
- **Cloud providers** — OpenAI, Anthropic, Google Gemini (via LiteLLM)
- **Local providers** — Ollama, HuggingFace Transformers, any OpenAI-compatible server (vLLM, LM Studio, llama.cpp)
- **Custom providers** — Subclass `BaseProvider` to add anything
- **Optimization** — 4-bit/8-bit quantization, LoRA/QLoRA adapters, Flash Attention, preset profiles
- **Benchmarking** — 200+ standard benchmarks (MMLU, HumanEval, GSM8K, HellaSwag) via lm-evaluation-harness
- **Performance profiling** — Latency profiling (p50/p95/p99), throughput, concurrent load testing, custom perf suites
- **YAML config** — Configure everything declaratively with env var substitution
- **CLI** — Profile from the command line

## Installation

```bash
# Core (includes LiteLLM for all cloud/API providers)
pip install -e .

# With local model support
pip install -e ".[local]"       # HuggingFace + torch + bitsandbytes
pip install -e ".[cli]"         # CLI (click)
pip install -e ".[benchmark]"   # Benchmarking (lm-eval)
pip install -e ".[all]"         # Everything
```

## Quick Start

```python
from unified_llm import UnifiedLLM
from unified_llm.perf import PerfRunner, PerfConfig

llm = UnifiedLLM()
llm.add_provider("ollama")  # assumes Ollama is running locally

# Performance profiling
runner = PerfRunner(llm)
report = runner.run(["ollama/llama3.1"], PerfConfig(num_requests=10))
print(report.summary_table())
```

## Model Addressing

Models are addressed as `provider/model`:

| Address | Provider | Model |
|---|---|---|
| `openai/gpt-4o` | OpenAI API | GPT-4o |
| `anthropic/claude-sonnet-4-20250514` | Anthropic API | Claude Sonnet |
| `ollama/llama3.1` | Ollama (local) | Llama 3.1 |
| `ollama/deepseek-r1:8b` | Ollama (local) | DeepSeek R1 8B |
| `hf/meta-llama/Llama-3.1-8B-Instruct` | HuggingFace (local) | Llama 3.1 8B |
| `local_vllm/my-model` | OpenAI-compatible server | Any model |
| `my_custom/anything` | Your custom provider | Your model |

## Local Models with Optimization

```python
from unified_llm import UnifiedLLM
from unified_llm.providers import HuggingFaceProvider
from unified_llm.optimization import get_profile, apply_profile_to_provider_kwargs

llm = UnifiedLLM()

# Quick: use a preset optimization profile
profile = get_profile("fast_4bit")  # or: balanced_8bit, qlora_finetune, lora_finetune
kwargs = apply_profile_to_provider_kwargs(profile)

llm.add_provider("hf", provider=HuggingFaceProvider(
    model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    **kwargs,
))

# Or configure manually
llm.add_provider("hf_custom", provider=HuggingFaceProvider(
    model_name_or_path="mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit=True,
    use_flash_attention=True,
    lora_adapter_path="./my-adapter",  # Optional LoRA adapter
))

# Use an OpenAI-compatible local server (vLLM, LM Studio, etc.)
from unified_llm.providers import LiteLLMProvider
llm.add_provider("vllm", provider=LiteLLMProvider(
    provider_name="openai",
    api_base="http://localhost:8000/v1",
    api_key="not-needed",
))
```

## Benchmarking

Run standard LLM benchmarks through the unified interface using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```python
from unified_llm import UnifiedLLM
from unified_llm.benchmark import BenchmarkRunner, BenchmarkConfig

llm = UnifiedLLM()
llm.add_provider("ollama")

runner = BenchmarkRunner(llm)
report = runner.run("ollama/qwen3:4b", BenchmarkConfig(
    tasks=["gsm8k"],
    limit=10,
    generate_only=True,  # for providers without logprobs support
))
print(report.summary_table())
```

Providers that support logprobs (OpenAI, Ollama) can run loglikelihood-based benchmarks like MMLU and HellaSwag. For providers without logprobs (Anthropic, Google), use `generate_only=True` to filter to generation-based tasks (GSM8K, HumanEval, MBPP, etc.).

## Performance Profiling

```python
from unified_llm import UnifiedLLM
from unified_llm.perf import PerfRunner, PerfConfig, CustomPerf

llm = UnifiedLLM()
llm.add_provider("openai")
llm.add_provider("ollama")

# Latency & throughput profiling
runner = PerfRunner(llm)
report = runner.run(
    ["openai/gpt-4o", "ollama/llama3.1"],
    PerfConfig(num_requests=50, concurrency=5, prompt="Explain REST APIs."),
)
print(report.summary_table())

# Custom perf suite with scoring
import json
def json_scorer(resp):
    try:
        json.loads(resp.content)
        return {"valid_json": 1.0}
    except:
        return {"valid_json": 0.0}

bench = CustomPerf("json_gen", ["Output JSON for a user profile."], json_scorer)
results = bench.run(llm, "openai/gpt-4o")
```

## YAML Configuration

```yaml
# config.yaml — API keys are read from env vars automatically by LiteLLM
defaults:
  temperature: 0.7
  max_tokens: 1024

providers:
  openai:
    default_model: gpt-4o
  anthropic:
    default_model: claude-sonnet-4-20250514
  ollama:
    api_base: http://localhost:11434
    default_model: llama3.1
  local_vllm:               # OpenAI-compatible server
    type: openai
    api_base: http://localhost:8000/v1
    api_key: not-needed
```

```python
from unified_llm.config import create_client_from_config
llm = create_client_from_config("config.yaml")
```

## CLI

```bash
# Performance profiling
ullm perf openai/gpt-4o ollama/llama3.1 --requests 50

# With config file
ullm -c config.yaml perf ollama/llama3.1 --requests 20

# Benchmarking
ullm benchmark ollama/qwen3:4b -t gsm8k --limit 5 --generate-only
ullm benchmark openai/gpt-4o -t mmlu,gsm8k --num-fewshot 5 --limit 50

# List providers
ullm list
```

## Architecture

```
unified-llm/
├── unified_llm/
│   ├── __init__.py          # Public API
│   ├── client.py            # UnifiedLLM — the main entry point
│   ├── types.py             # All data models (Pydantic)
│   ├── registry.py          # Provider registry
│   ├── config.py            # YAML config loader
│   ├── cli.py               # CLI (click)
│   ├── providers/
│   │   ├── base.py          # BaseProvider ABC
│   │   ├── litellm_provider.py   # LiteLLM backend (OpenAI, Anthropic, Google, Ollama)
│   │   └── huggingface_provider.py
│   ├── benchmark/
│   │   ├── __init__.py      # Public API
│   │   ├── types.py         # BenchmarkConfig, TaskResult, BenchmarkReport
│   │   ├── adapter.py       # UnifiedLMAdapter (LM base class bridge)
│   │   └── runner.py        # BenchmarkRunner (wraps lm-eval)
│   ├── optimization/
│   │   └── __init__.py      # Quantization, LoRA, profiles
│   └── perf/
│       └── __init__.py      # PerfRunner + CustomPerf
├── configs/
│   └── default.yaml
├── examples/
│   └── advanced_usage.py
└── pyproject.toml
```

## Extending

The project is designed around two extension points:

1. **Custom Providers** — Subclass `BaseProvider`, implement `complete()` and `acomplete()`, register with `register_provider()`
2. **Custom Perf Suites** — Use `CustomPerf` with any scoring function
3. **Custom Benchmarks** — Use `BenchmarkConfig(custom_task_paths=[...])` to point to custom lm-eval task YAML files

## License

MIT
