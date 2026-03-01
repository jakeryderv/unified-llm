"""
Example: Running LLM benchmarks with lm-evaluation-harness.

Shows how to:
    - Run standard benchmarks (GSM8K, MMLU, etc.) through BenchmarkRunner
    - Use generate_only mode for providers without logprobs
    - Use custom task directories

Requires: pip install -e ".[benchmark]"
"""

from unified_llm import UnifiedLLM
from unified_llm.benchmark import BenchmarkConfig, BenchmarkRunner

llm = UnifiedLLM()
# llm.add_provider("ollama")   # assumes Ollama is running locally
# llm.add_provider("openai")   # uses OPENAI_API_KEY env var


# =====================================================================
# 1. BASIC BENCHMARK RUN
# =====================================================================

runner = BenchmarkRunner(llm)

# Run GSM8K (math) with a small sample limit for quick testing
# report = runner.run("ollama/qwen3:4b", BenchmarkConfig(
#     tasks=["gsm8k"],
#     limit=5,
#     generate_only=True,  # GSM8K is generation-based
# ))
# print(report.summary_table())


# =====================================================================
# 2. MULTIPLE TASKS
# =====================================================================

# Run multiple benchmarks at once
# report = runner.run("openai/gpt-4o", BenchmarkConfig(
#     tasks=["mmlu", "gsm8k", "hellaswag"],
#     num_fewshot=5,
#     limit=50,
# ))
# print(report.summary_table())


# =====================================================================
# 3. GENERATE-ONLY MODE (for providers without logprobs)
# =====================================================================

# Anthropic and Google don't support logprobs, so use generate_only
# to filter to generation-based tasks (GSM8K, HumanEval, MBPP, etc.)
# report = runner.run("anthropic/claude-sonnet-4-20250514", BenchmarkConfig(
#     tasks=["gsm8k", "humaneval"],
#     generate_only=True,
#     limit=10,
# ))
# print(report.summary_table())


# =====================================================================
# 4. CUSTOM TASK DIRECTORY
# =====================================================================

# Point to a directory containing custom lm-eval task YAML files
# report = runner.run("ollama/llama3.1", BenchmarkConfig(
#     tasks=["my_custom_task"],
#     custom_task_paths=["./my_tasks/"],
#     limit=20,
# ))
# print(report.summary_table())


# =====================================================================
# 5. ACCESS RAW RESULTS
# =====================================================================

# The raw lm-eval output is available for detailed analysis
# report = runner.run("ollama/qwen3:4b", BenchmarkConfig(
#     tasks=["gsm8k"],
#     limit=5,
#     generate_only=True,
# ))
# print("Raw results keys:", list(report.raw_results.keys()))
# for task, result in report.task_results.items():
#     print(f"\n{task}:")
#     for metric, value in result.metrics.items():
#         print(f"  {metric}: {value:.4f}")


print("Benchmark examples ready! Uncomment sections above to run.")
llm.shutdown()
