"""
Example: Performance profiling and optimization profiles.

Shows how to:
    - Run perf profiling across models
    - Use optimization profiles for local models
"""

from unified_llm import (
    UnifiedLLM,
    GenerationConfig,
)
from unified_llm.types import CompletionResponse

llm = UnifiedLLM()
# llm.add_provider("openai")   # uses OPENAI_API_KEY env var
# llm.add_provider("ollama")   # assumes Ollama is running locally


# =====================================================================
# 1. PERFORMANCE PROFILING
# =====================================================================

from unified_llm.perf import PerfRunner, PerfConfig, CustomPerf

runner = PerfRunner(llm)

# -- Basic latency/throughput profiling --
# report = runner.run(
#     ["openai/gpt-4o", "ollama/llama3.1"],
#     PerfConfig(num_requests=20, prompt="Explain REST APIs in one sentence."),
# )
# print(report.summary_table())

# -- Custom perf suite with scoring function --
def json_validity_scorer(response: CompletionResponse) -> dict[str, float]:
    """Score whether the model outputs valid JSON."""
    import json
    try:
        json.loads(response.content)
        return {"valid_json": 1.0}
    except (json.JSONDecodeError, ValueError):
        return {"valid_json": 0.0}

json_perf = CustomPerf(
    name="json_output",
    prompts=[
        "Output a JSON object with keys 'name' and 'age' for a 30-year-old named Alice.",
        "Output a JSON array of 3 colors.",
        "Output a JSON object representing a book with title, author, and year.",
    ],
    scorer=json_validity_scorer,
    config=GenerationConfig(temperature=0, max_tokens=200, response_format="json"),
)

# results = json_perf.run(llm, "openai/gpt-4o")
# print(f"JSON perf: {results}")


# =====================================================================
# 2. OPTIMIZATION PROFILES (for local models)
# =====================================================================

from unified_llm.optimization import get_profile, apply_profile_to_provider_kwargs
from unified_llm.providers import HuggingFaceProvider

# Get a preset profile
profile = get_profile("fast_4bit")
print(f"\nOptimization profile '{profile.name}':")
print(f"  Quantization: {profile.quantization.method.value}")
print(f"  Flash Attention: {profile.use_flash_attention}")

# Convert to provider kwargs
# kwargs = apply_profile_to_provider_kwargs(profile)
# llm.add_provider("hf_optimized", provider=HuggingFaceProvider(
#     model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
#     **kwargs,
# ))

# Or load a LoRA-adapted model
# llm.add_provider("hf_finetuned", provider=HuggingFaceProvider(
#     model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
#     lora_adapter_path="./my-lora-adapter",
#     load_in_4bit=True,
# ))

print("\nDone! See the README for more examples.")
llm.shutdown()
