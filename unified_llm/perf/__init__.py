"""
Performance profiling — measure latency, throughput, and custom metrics.

Supports:
    - Latency profiling (avg, p50, p95, p99)
    - Throughput (requests/sec, tokens/sec)
    - Concurrent load testing
    - Custom perf suites
"""

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from unified_llm.types import (
    PerfMetrics,
    GenerationConfig,
    Message,
)

if TYPE_CHECKING:
    from unified_llm.client import UnifiedLLM

__all__ = [
    "PerfConfig",
    "PerfReport",
    "PerfRunner",
    "CustomPerf",
]


@dataclass
class PerfConfig:
    """Configuration for a perf run."""
    num_requests: int = 10
    concurrency: int = 1
    warmup_requests: int = 1
    prompt: str | list[Message] = "Write a haiku about programming."
    system: str | None = None
    generation_config: GenerationConfig | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class PerfReport:
    """Full report from a perf run, potentially across multiple models."""
    results: dict[str, PerfMetrics]  # model_id → metrics
    config: PerfConfig
    timestamp: str = ""

    def summary_table(self) -> str:
        """Return a formatted summary table."""
        lines = []
        header = f"{'Model':<40} {'Avg(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'P99(ms)':>10} {'Tok/s':>10} {'Req/s':>10} {'Errors':>7}"
        lines.append(header)
        lines.append("-" * len(header))
        for model, m in self.results.items():
            lines.append(
                f"{model:<40} {m.avg_latency_ms:>10.1f} {m.p50_latency_ms:>10.1f} "
                f"{m.p95_latency_ms:>10.1f} {m.p99_latency_ms:>10.1f} "
                f"{m.avg_tokens_per_second:>10.1f} {m.requests_per_second:>10.2f} {m.errors:>7}"
            )
        return "\n".join(lines)


class PerfRunner:
    """
    Run performance profiling against any model.

    Example:
        runner = PerfRunner(llm)
        report = runner.run(
            ["openai/gpt-4o", "ollama/llama3.1"],
            PerfConfig(num_requests=50, concurrency=5),
        )
        print(report.summary_table())
    """

    def __init__(self, client: "UnifiedLLM"):
        self.client = client

    def run(
        self,
        model_ids: list[str] | str,
        config: PerfConfig | None = None,
    ) -> PerfReport:
        """Run perf profiling synchronously (no concurrency)."""
        config = config or PerfConfig()
        if isinstance(model_ids, str):
            model_ids = [model_ids]

        results = {}
        for mid in model_ids:
            results[mid] = self._bench_single(mid, config)

        from datetime import datetime
        return PerfReport(
            results=results,
            config=config,
            timestamp=datetime.now().isoformat(),
        )

    async def arun(
        self,
        model_ids: list[str] | str,
        config: PerfConfig | None = None,
    ) -> PerfReport:
        """Run perf profiling with async concurrency support."""
        config = config or PerfConfig()
        if isinstance(model_ids, str):
            model_ids = [model_ids]

        results = {}
        for mid in model_ids:
            results[mid] = await self._abench_single(mid, config)

        from datetime import datetime
        return PerfReport(
            results=results,
            config=config,
            timestamp=datetime.now().isoformat(),
        )

    def _bench_single(self, model_id: str, config: PerfConfig) -> PerfMetrics:
        gen_config = config.generation_config or GenerationConfig(temperature=0, max_tokens=256)

        # Warmup
        for _ in range(config.warmup_requests):
            try:
                self.client.complete(model_id, config.prompt, gen_config, config.system)
            except Exception:
                pass

        # Actual profiling
        latencies: list[float] = []
        total_tokens = 0
        errors = 0

        t_start = time.perf_counter()
        for _ in range(config.num_requests):
            try:
                resp = self.client.complete(model_id, config.prompt, gen_config, config.system)
                latencies.append(resp.latency_ms)
                total_tokens += resp.usage.total_tokens
            except Exception:
                errors += 1
        t_end = time.perf_counter()
        duration = t_end - t_start

        return self._compute_metrics(latencies, total_tokens, errors, duration, config.num_requests)

    async def _abench_single(self, model_id: str, config: PerfConfig) -> PerfMetrics:
        gen_config = config.generation_config or GenerationConfig(temperature=0, max_tokens=256)

        # Warmup
        for _ in range(config.warmup_requests):
            try:
                await self.client.acomplete(model_id, config.prompt, gen_config, config.system)
            except Exception:
                pass

        # Concurrent profiling
        latencies: list[float] = []
        total_tokens = 0
        errors = 0
        sem = asyncio.Semaphore(config.concurrency)

        async def _single_request():
            nonlocal total_tokens, errors
            async with sem:
                try:
                    resp = await self.client.acomplete(model_id, config.prompt, gen_config, config.system)
                    latencies.append(resp.latency_ms)
                    total_tokens += resp.usage.total_tokens
                except Exception:
                    errors += 1

        t_start = time.perf_counter()
        await asyncio.gather(*[_single_request() for _ in range(config.num_requests)])
        duration = time.perf_counter() - t_start

        return self._compute_metrics(latencies, total_tokens, errors, duration, config.num_requests)

    @staticmethod
    def _compute_metrics(
        latencies: list[float],
        total_tokens: int,
        errors: int,
        duration: float,
        num_requests: int,
    ) -> PerfMetrics:
        if not latencies:
            return PerfMetrics(
                total_requests=num_requests,
                errors=errors,
                duration_seconds=duration,
            )

        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        return PerfMetrics(
            requests_per_second=num_requests / max(duration, 0.001),
            avg_latency_ms=statistics.mean(sorted_lat),
            p50_latency_ms=sorted_lat[int(n * 0.5)],
            p95_latency_ms=sorted_lat[min(int(n * 0.95), n - 1)],
            p99_latency_ms=sorted_lat[min(int(n * 0.99), n - 1)],
            avg_tokens_per_second=total_tokens / max(duration, 0.001),
            total_tokens=total_tokens,
            total_requests=num_requests,
            errors=errors,
            duration_seconds=duration,
        )


# ---------------------------------------------------------------------------
# Custom perf support
# ---------------------------------------------------------------------------

class CustomPerf:
    """
    Define a custom perf suite with a scoring function.

    Example:
        def code_quality(response):
            # Your custom scoring logic
            return {"parseable": 1.0 if is_valid_python(response.content) else 0.0}

        bench = CustomPerf("code_gen", prompts, code_quality)
        results = bench.run(llm, "openai/gpt-4o")
    """

    def __init__(
        self,
        name: str,
        prompts: list[str | list[Message]],
        scorer: Callable,
        config: GenerationConfig | None = None,
    ):
        self.name = name
        self.prompts = prompts
        self.scorer = scorer
        self.config = config or GenerationConfig(temperature=0, max_tokens=1024)

    def run(self, client: "UnifiedLLM", model_id: str) -> dict[str, Any]:
        """Run the custom perf suite and return aggregated scores."""
        all_scores: dict[str, list[float]] = {}
        latencies: list[float] = []

        for prompt in self.prompts:
            resp = client.complete(model_id, prompt, self.config)
            latencies.append(resp.latency_ms)
            scores = self.scorer(resp)
            for k, v in scores.items():
                all_scores.setdefault(k, []).append(v)

        aggregated = {k: statistics.mean(v) for k, v in all_scores.items()}
        aggregated["avg_latency_ms"] = statistics.mean(latencies) if latencies else 0
        aggregated["num_prompts"] = len(self.prompts)
        return aggregated
