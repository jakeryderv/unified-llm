"""
Benchmarking — run standard LLM benchmarks via lm-evaluation-harness.

Supports 200+ built-in benchmarks (MMLU, HumanEval, GSM8K, HellaSwag, etc.)
through the unified provider interface.
"""

from unified_llm.benchmark.types import (
    BENCHMARK_SUITES,
    BenchmarkConfig,
    BenchmarkReport,
    TaskResult,
    comparison_table,
)

__all__ = [
    "BENCHMARK_SUITES",
    "BenchmarkConfig",
    "BenchmarkReport",
    "BenchmarkRunner",
    "TaskResult",
    "comparison_table",
]


def __getattr__(name: str):
    """Lazy-import BenchmarkRunner to avoid requiring lm-eval at import time."""
    if name == "BenchmarkRunner":
        from unified_llm.benchmark.runner import BenchmarkRunner
        return BenchmarkRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
