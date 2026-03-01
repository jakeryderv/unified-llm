"""High-level benchmark runner wrapping lm-evaluation-harness."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Callable

from unified_llm.benchmark.adapter import UnifiedLMAdapter
from unified_llm.benchmark.types import BenchmarkConfig, BenchmarkReport, TaskResult

if TYPE_CHECKING:
    from unified_llm.client import UnifiedLLM

logger = logging.getLogger(__name__)

# Tasks that rely only on generation (no loglikelihood needed).
_GENERATE_ONLY_TASKS = frozenset({
    "gsm8k",
    "humaneval",
    "mbpp",
    "triviaqa",
    "drop",
    "math",
    "minerva_math",
    "bbh",
    "gpqa",
    "ifeval",
})


class BenchmarkRunner:
    """
    Run standard LLM benchmarks through the unified provider interface.

    Wraps lm-evaluation-harness ``simple_evaluate()`` and converts results
    into a clean ``BenchmarkReport``.

    Example::

        runner = BenchmarkRunner(llm)
        report = runner.run("ollama/qwen3:4b", BenchmarkConfig(
            tasks=["gsm8k"],
            limit=10,
            generate_only=True,
        ))
        print(report.summary_table())
    """

    def __init__(self, client: UnifiedLLM, progress: Callable[[str], None] | None = None):
        self.client = client
        self._progress = progress or self._default_progress

    @staticmethod
    def _default_progress(msg: str) -> None:
        """Print progress messages to stderr so they don't mix with results."""
        print(msg, file=sys.stderr, flush=True)

    def run(
        self,
        model_id: str,
        config: BenchmarkConfig | None = None,
    ) -> BenchmarkReport:
        """
        Run benchmarks for *model_id* with the given config.

        Args:
            model_id: "provider/model" string (e.g. "ollama/qwen3:4b")
            config: Benchmark configuration. Defaults to empty (must specify tasks).

        Returns:
            BenchmarkReport with per-task results and a summary_table() method.
        """
        try:
            import lm_eval
        except ImportError as e:
            raise ImportError(
                "lm-eval is required for benchmarking. Install with: pip install -e '.[benchmark]'"
            ) from e

        config = config or BenchmarkConfig()
        config = config.resolve_tasks()
        if not config.tasks:
            raise ValueError("No tasks specified. Pass tasks=['gsm8k', ...] or suite='standard' in BenchmarkConfig.")

        # Auto-register provider if needed
        provider_name = model_id.split("/")[0]
        if provider_name not in self.client.providers:
            self.client.add_provider(provider_name)

        # Filter to generate-only tasks if requested
        tasks = list(config.tasks)
        if config.generate_only:
            tasks = self._filter_generate_only_tasks(tasks)
            if not tasks:
                raise ValueError(
                    "No generation-only tasks remain after filtering. "
                    f"Generation-only tasks include: {', '.join(sorted(_GENERATE_ONLY_TASKS))}"
                )

        # Create the adapter
        adapter = UnifiedLMAdapter(
            client=self.client,
            model_id=model_id,
            batch_size=config.batch_size,
            max_gen_toks=1024,
        )

        # Build evaluate kwargs
        eval_kwargs: dict = {
            "model": adapter,
            "tasks": tasks,
            "batch_size": config.batch_size,
            "log_samples": config.log_samples,
        }
        if config.num_fewshot is not None:
            eval_kwargs["num_fewshot"] = config.num_fewshot
        if config.limit is not None:
            eval_kwargs["limit"] = config.limit
        if config.custom_task_paths:
            eval_kwargs["task_manager"] = lm_eval.tasks.TaskManager(
                include_path=config.custom_task_paths
            )
        if config.seed is not None:
            eval_kwargs["random_seed"] = config.seed
            eval_kwargs["numpy_random_seed"] = config.seed
            eval_kwargs["torch_random_seed"] = config.seed
        if config.confirm_run_unsafe_code:
            eval_kwargs["confirm_run_unsafe_code"] = True

        # Run evaluation with progress
        self._progress(f"Running {len(tasks)} task(s): {', '.join(tasks)}")
        logger.info("Starting benchmark: model=%s tasks=%s", model_id, tasks)
        raw = lm_eval.simple_evaluate(**eval_kwargs)
        self._progress("Evaluation complete.")

        return self._parse_results(model_id, raw)

    @staticmethod
    def _filter_generate_only_tasks(tasks: list[str]) -> list[str]:
        """Keep only tasks that use generation (not loglikelihood)."""
        filtered = []
        for t in tasks:
            # Match against known generate-only tasks (case-insensitive, prefix match)
            task_lower = t.lower()
            if any(task_lower.startswith(gen_task) for gen_task in _GENERATE_ONLY_TASKS):
                filtered.append(t)
            else:
                logger.info("Skipping task '%s' (not generation-only)", t)
        return filtered

    @staticmethod
    def _parse_results(model_id: str, raw: dict) -> BenchmarkReport:
        """Convert lm-eval raw output into a clean BenchmarkReport."""
        results_dict = raw.get("results", {})
        n_samples = raw.get("n-samples", {})
        n_shot = raw.get("n-shot", {})
        versions = raw.get("versions", {})
        hib_top = raw.get("higher_is_better", {})
        task_results: dict[str, TaskResult] = {}

        for task_name, task_data in results_dict.items():
            # lm-eval uses keys like "acc,none", "acc_norm,none", "exact_match,none"
            metrics: dict[str, float] = {}
            higher_is_better: dict[str, bool] = {}

            # Get higher_is_better from task-level or top-level
            hib_data = task_data.get("higher_is_better", hib_top.get(task_name, {}))

            for key, value in task_data.items():
                if key in ("alias", "higher_is_better"):
                    continue
                if not isinstance(value, (int, float)):
                    continue

                # Clean up metric names: "acc,none" -> "acc", "exact_match,strict-match" -> "exact_match"
                clean_name = key.split(",")[0] if "," in key else key

                # Skip stderr metrics
                if clean_name.endswith("_stderr"):
                    continue

                metrics[clean_name] = float(value)
                if clean_name in hib_data:
                    higher_is_better[clean_name] = hib_data[clean_name]

            # Get effective sample count from n-samples
            samples_info = n_samples.get(task_name, {})
            effective_samples = samples_info.get("effective", 0) if isinstance(samples_info, dict) else 0

            task_results[task_name] = TaskResult(
                task_name=task_name,
                metrics=metrics,
                num_samples=effective_samples,
                num_fewshot=n_shot.get(task_name, 0),
                version=versions.get(task_name, ""),
                higher_is_better=higher_is_better,
            )

        return BenchmarkReport(
            model_id=model_id,
            task_results=task_results,
            raw_results=raw,
        )
