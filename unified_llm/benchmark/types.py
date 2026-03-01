"""Data models for the benchmark module."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Preset suites
# ---------------------------------------------------------------------------

BENCHMARK_SUITES: dict[str, dict] = {
    "standard": {
        "description": "Comprehensive eval across all major categories (requires confirm_run_unsafe_code)",
        "tasks": ["gsm8k", "minerva_math", "drop", "triviaqa", "bbh", "ifeval", "humaneval"],
        "generate_only": True,
        "confirm_run_unsafe_code": True,
    },
    "safe": {
        "description": "Standard suite without code execution — no special flags needed",
        "tasks": ["gsm8k", "minerva_math", "drop", "triviaqa", "bbh", "ifeval"],
        "generate_only": True,
    },
    "math": {
        "description": "Math reasoning benchmarks",
        "tasks": ["gsm8k", "minerva_math"],
        "generate_only": True,
    },
    "reasoning": {
        "description": "Hard reasoning and reading comprehension",
        "tasks": ["bbh", "drop"],
        "generate_only": True,
    },
    "knowledge": {
        "description": "Factual knowledge and instruction following",
        "tasks": ["triviaqa", "ifeval"],
        "generate_only": True,
    },
    "code": {
        "description": "Code generation benchmarks (requires confirm_run_unsafe_code)",
        "tasks": ["humaneval", "mbpp"],
        "generate_only": True,
        "confirm_run_unsafe_code": True,
    },
}


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    tasks: list[str] = Field(default_factory=list)
    suite: str | None = None
    num_fewshot: int | None = None
    batch_size: int | str = 1
    limit: int | float | None = None
    custom_task_paths: list[str] = Field(default_factory=list)
    generate_only: bool = False
    log_samples: bool = False
    seed: int | None = None
    confirm_run_unsafe_code: bool = False

    def resolve_tasks(self) -> "BenchmarkConfig":
        """Return a new config with suite expanded into tasks and flags."""
        if not self.suite:
            return self
        if self.suite not in BENCHMARK_SUITES:
            available = ", ".join(sorted(BENCHMARK_SUITES))
            raise ValueError(f"Unknown suite '{self.suite}'. Available: {available}")

        preset = BENCHMARK_SUITES[self.suite]
        # Merge: explicit tasks extend suite tasks, explicit flags override
        merged_tasks = list(preset["tasks"])
        for t in self.tasks:
            if t not in merged_tasks:
                merged_tasks.append(t)

        overrides: dict = {
            "tasks": merged_tasks,
            "suite": None,
        }
        if preset.get("generate_only") and not self.tasks:
            overrides["generate_only"] = True
        if preset.get("confirm_run_unsafe_code"):
            overrides["confirm_run_unsafe_code"] = True

        return self.model_copy(update=overrides)


@dataclass
class TaskResult:
    """Results for a single benchmark task."""

    task_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    num_fewshot: int = 0
    version: str | int = ""
    higher_is_better: dict[str, bool] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Full report from a benchmark run."""

    model_id: str
    task_results: dict[str, TaskResult] = field(default_factory=dict)
    raw_results: dict | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary_table(self) -> str:
        """Return a formatted summary table of benchmark results.

        Groups subtasks under their parent task and shows only relevant
        metrics for each task (no empty columns from unrelated tasks).
        """
        if not self.task_results:
            return "No benchmark results."

        lines = [
            f"Benchmark Report: {self.model_id}",
            f"Timestamp: {self.timestamp}",
            "",
        ]

        # Separate parent/top-level tasks from subtasks.
        # A subtask's name starts with a parent name + "_" (e.g. bbh_cot_fewshot_...).
        # A parent has num_samples == 0 when it's just an aggregate group.
        parents, top_level, subtasks = self._group_tasks()

        # Render top-level tasks (those with no subtasks) grouped together
        if top_level:
            lines.extend(self._render_task_group("Results", top_level))
            lines.append("")

        # Render each parent group with its subtasks
        for parent_name in parents:
            children = subtasks.get(parent_name, [])
            parent_tr = self.task_results.get(parent_name)

            group_tasks = list(children)
            group_label = parent_name
            if parent_tr and parent_tr.metrics:
                group_label = f"{parent_name} (aggregate)"

            lines.extend(self._render_task_group(group_label, group_tasks, parent_tr))
            lines.append("")

        # Remove trailing blank line
        while lines and lines[-1] == "":
            lines.pop()

        return "\n".join(lines)

    def _group_tasks(self) -> tuple[list[str], list[TaskResult], dict[str, list[TaskResult]]]:
        """Split task_results into parents, top-level tasks, and subtask groups."""
        all_names = list(self.task_results.keys())

        # Find parent tasks: tasks that have subtasks (other tasks starting with parent_)
        parent_names: list[str] = []
        subtask_map: dict[str, list[TaskResult]] = {}
        claimed: set[str] = set()

        # Sort by name length descending so longer parent prefixes match first
        candidates = sorted(all_names, key=len)
        for name in candidates:
            children = [
                self.task_results[other]
                for other in all_names
                if other != name and other.startswith(name + "_") and other not in claimed
            ]
            if children:
                parent_names.append(name)
                subtask_map[name] = children
                claimed.add(name)
                for c in children:
                    claimed.add(c.task_name)

        # Top-level: everything not claimed as parent or subtask
        top_level = [
            self.task_results[n] for n in all_names
            if n not in claimed
        ]

        return parent_names, top_level, subtask_map

    @staticmethod
    def _render_task_group(
        label: str,
        tasks: list[TaskResult],
        aggregate: TaskResult | None = None,
    ) -> list[str]:
        """Render a group of tasks sharing the same metrics into a mini-table."""
        if not tasks and not (aggregate and aggregate.metrics):
            return []

        # Collect metrics from this group only
        group_metrics: list[str] = []
        all_trs = list(tasks)
        if aggregate and aggregate.metrics:
            all_trs = [aggregate] + all_trs
        for tr in all_trs:
            for m in tr.metrics:
                if m not in group_metrics:
                    group_metrics.append(m)

        if not group_metrics:
            return []

        # Truncate metric names for display
        display_metrics = group_metrics[:4]
        metric_headers = "".join(f"{m[:16]:>18}" for m in display_metrics)
        header = f"  {'Task':<40} {'N':>6}{metric_headers}"
        lines = [label, header, "  " + "-" * (len(header) - 2)]

        # Aggregate row first (if it has metrics)
        if aggregate and aggregate.metrics:
            vals = "".join(
                f"{aggregate.metrics.get(m, 0.0):>18.4f}" for m in display_metrics
            )
            lines.append(f"  {'(average)':<40} {'':>6}{vals}")

        # Subtask/individual rows
        for tr in tasks:
            # Shorten subtask names by removing common prefix
            display_name = tr.task_name
            if len(display_name) > 40:
                display_name = "..." + display_name[-37:]
            vals = "".join(
                f"{tr.metrics.get(m, 0.0):>18.4f}" for m in display_metrics
            )
            lines.append(f"  {display_name:<40} {tr.num_samples:>6}{vals}")

        return lines

    def save_json(self, path: str | Path) -> Path:
        """Save report to a JSON file. Returns the path written."""
        path = Path(path)
        data = {
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "task_results": {
                name: asdict(tr) for name, tr in self.task_results.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        return path


def comparison_table(reports: list[BenchmarkReport]) -> str:
    """Render a side-by-side comparison of multiple benchmark reports.

    Shows only top-level / aggregate tasks (not individual subtasks)
    with one column per model.
    """
    if not reports:
        return "No reports to compare."
    if len(reports) == 1:
        return reports[0].summary_table()

    # Collect top-level tasks across all reports (skip subtask rows).
    # A task is "top-level" if it either has no subtasks, or is the aggregate parent.
    all_task_names: list[str] = []
    for r in reports:
        for name in r.task_results:
            if name not in all_task_names:
                all_task_names.append(name)

    # Filter to top-level only: remove names that are subtasks of another
    top_level: list[str] = []
    for name in all_task_names:
        is_subtask = any(
            name != other and name.startswith(other + "_")
            for other in all_task_names
        )
        if not is_subtask:
            top_level.append(name)

    # For each top-level task, find the primary metric (first metric from first report that has it)
    task_metrics: dict[str, str] = {}
    for task in top_level:
        for r in reports:
            tr = r.task_results.get(task)
            if tr and tr.metrics:
                task_metrics[task] = next(iter(tr.metrics))
                break

    # Build table
    model_names = [r.model_id for r in reports]
    # Truncate model names for column headers
    col_width = max(18, max(len(m) for m in model_names) + 2)
    model_headers = "".join(f"{m:>{col_width}}" for m in model_names)
    header = f"  {'Task':<30} {'Metric':<18}{model_headers}"

    lines = [
        "Comparison",
        "",
        header,
        "  " + "-" * (len(header) - 2),
    ]

    for task in top_level:
        metric_name = task_metrics.get(task, "")
        if not metric_name:
            continue

        values: list[str] = []
        for r in reports:
            tr = r.task_results.get(task)
            if tr and metric_name in tr.metrics:
                values.append(f"{tr.metrics[metric_name]:.4f}")
            else:
                values.append("-")
        val_str = "".join(f"{v:>{col_width}}" for v in values)
        lines.append(f"  {task:<30} {metric_name:<18}{val_str}")

    return "\n".join(lines)
