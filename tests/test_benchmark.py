"""Tests for the benchmark module."""

import json
from pathlib import Path

import pytest

from unified_llm import (
    BaseProvider,
    CompletionResponse,
    FinishReason,
    GenerationConfig,
    ModelInfo,
    ModelLocation,
    ProviderType,
    TokenUsage,
    UnifiedLLM,
)
from unified_llm.benchmark.types import BENCHMARK_SUITES, BenchmarkConfig, BenchmarkReport, TaskResult, comparison_table


# ---------------------------------------------------------------------------
# Echo provider for adapter tests
# ---------------------------------------------------------------------------


class EchoProvider(BaseProvider):
    """Minimal provider that echoes input — no external deps."""

    default_model = "echo"

    def complete(self, messages, config=None, model=None):
        config, model = self._resolve_config(config, model)
        last = messages[-1].content if messages else ""
        return CompletionResponse(
            model=model,
            provider="echo",
            content=f"Echo: {last}",
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            latency_ms=0.1,
        )

    async def acomplete(self, messages, config=None, model=None):
        return self.complete(messages, config, model)

    def list_models(self):
        return [ModelInfo(name="echo", provider=ProviderType.CUSTOM, location=ModelLocation.LOCAL)]


@pytest.fixture()
def llm():
    client = UnifiedLLM()
    client.add_provider("echo", provider=EchoProvider())
    yield client
    client.shutdown()


# ---------------------------------------------------------------------------
# BenchmarkConfig
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.tasks == []
        assert cfg.num_fewshot is None
        assert cfg.batch_size == 1
        assert cfg.limit is None
        assert cfg.generate_only is False
        assert cfg.log_samples is False
        assert cfg.seed is None
        assert cfg.confirm_run_unsafe_code is False

    def test_custom_values(self):
        cfg = BenchmarkConfig(
            tasks=["mmlu", "gsm8k"],
            num_fewshot=5,
            batch_size=4,
            limit=100,
            generate_only=True,
            log_samples=True,
            seed=42,
        )
        assert cfg.tasks == ["mmlu", "gsm8k"]
        assert cfg.num_fewshot == 5
        assert cfg.batch_size == 4
        assert cfg.limit == 100
        assert cfg.generate_only is True
        assert cfg.seed == 42

    def test_custom_task_paths(self):
        cfg = BenchmarkConfig(custom_task_paths=["/path/to/tasks"])
        assert cfg.custom_task_paths == ["/path/to/tasks"]


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------


class TestBenchmarkSuites:
    def test_resolve_standard_suite(self):
        cfg = BenchmarkConfig(suite="standard").resolve_tasks()
        assert cfg.tasks == BENCHMARK_SUITES["standard"]["tasks"]
        assert cfg.generate_only is True
        assert cfg.confirm_run_unsafe_code is True
        assert cfg.suite is None

    def test_resolve_safe_suite(self):
        cfg = BenchmarkConfig(suite="safe").resolve_tasks()
        assert cfg.tasks == BENCHMARK_SUITES["safe"]["tasks"]
        assert cfg.generate_only is True
        assert cfg.confirm_run_unsafe_code is False

    def test_resolve_math_suite(self):
        cfg = BenchmarkConfig(suite="math").resolve_tasks()
        assert cfg.tasks == ["gsm8k", "minerva_math"]

    def test_suite_with_extra_tasks(self):
        cfg = BenchmarkConfig(suite="math", tasks=["drop"]).resolve_tasks()
        assert "gsm8k" in cfg.tasks
        assert "minerva_math" in cfg.tasks
        assert "drop" in cfg.tasks

    def test_suite_no_duplicates(self):
        cfg = BenchmarkConfig(suite="math", tasks=["gsm8k"]).resolve_tasks()
        assert cfg.tasks.count("gsm8k") == 1

    def test_unknown_suite_raises(self):
        with pytest.raises(ValueError, match="Unknown suite"):
            BenchmarkConfig(suite="nonexistent").resolve_tasks()

    def test_no_suite_passthrough(self):
        cfg = BenchmarkConfig(tasks=["gsm8k"]).resolve_tasks()
        assert cfg.tasks == ["gsm8k"]

    def test_all_suites_have_tasks(self):
        for name, info in BENCHMARK_SUITES.items():
            assert len(info["tasks"]) > 0, f"Suite '{name}' has no tasks"
            assert "description" in info, f"Suite '{name}' has no description"


# ---------------------------------------------------------------------------
# BenchmarkReport
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    def test_empty_report(self):
        report = BenchmarkReport(model_id="test/model")
        assert report.summary_table() == "No benchmark results."

    def test_summary_table_formatting(self):
        report = BenchmarkReport(
            model_id="ollama/llama3.1",
            task_results={
                "gsm8k": TaskResult(
                    task_name="gsm8k",
                    metrics={"exact_match": 0.45, "acc": 0.50},
                    num_samples=100,
                    num_fewshot=5,
                ),
                "mmlu": TaskResult(
                    task_name="mmlu",
                    metrics={"acc": 0.65},
                    num_samples=200,
                    num_fewshot=5,
                ),
            },
        )
        table = report.summary_table()
        assert "ollama/llama3.1" in table
        assert "gsm8k" in table
        assert "mmlu" in table
        assert "exact_match" in table
        assert "acc" in table

    def test_summary_table_grouped(self):
        """Subtasks should be grouped under their parent."""
        report = BenchmarkReport(
            model_id="test/model",
            task_results={
                "bbh": TaskResult(task_name="bbh", metrics={"exact_match": 0.30}, num_samples=0),
                "bbh_boolean_expressions": TaskResult(
                    task_name="bbh_boolean_expressions", metrics={"exact_match": 0.40}, num_samples=5,
                ),
                "bbh_causal_judgement": TaskResult(
                    task_name="bbh_causal_judgement", metrics={"exact_match": 0.20}, num_samples=5,
                ),
                "gsm8k": TaskResult(
                    task_name="gsm8k", metrics={"exact_match": 0.50}, num_samples=10,
                ),
            },
        )
        table = report.summary_table()
        # gsm8k should appear as a top-level result
        assert "gsm8k" in table
        # bbh should appear as a group label
        assert "bbh" in table
        # subtasks should appear
        assert "bbh_boolean_expressions" in table
        assert "bbh_causal_judgement" in table

    def test_task_result_defaults(self):
        tr = TaskResult(task_name="test")
        assert tr.metrics == {}
        assert tr.num_samples == 0
        assert tr.num_fewshot == 0
        assert tr.higher_is_better == {}


# ---------------------------------------------------------------------------
# UnifiedLMAdapter (requires lm_eval)
# ---------------------------------------------------------------------------


class TestUnifiedLMAdapter:
    @pytest.fixture(autouse=True)
    def _skip_without_lm_eval(self):
        pytest.importorskip("lm_eval")

    def test_generate_until(self, llm):
        from lm_eval.api.instance import Instance
        from unified_llm.benchmark.adapter import UnifiedLMAdapter

        adapter = UnifiedLMAdapter(client=llm, model_id="echo/echo")

        req = Instance(
            request_type="generate_until",
            doc={},
            arguments=("What is 2+2?", {"until": ["\n"], "max_gen_toks": 50}),
            idx=0,
        )
        results = adapter.generate_until([req])
        assert len(results) == 1
        assert isinstance(results[0], str)
        assert results[0] != ""

    def test_loglikelihood_fallback(self, llm):
        from lm_eval.api.instance import Instance
        from unified_llm.benchmark.adapter import UnifiedLMAdapter

        # Force logprobs_supported=False to test fallback
        adapter = UnifiedLMAdapter(
            client=llm,
            model_id="echo/echo",
            logprobs_supported=False,
        )

        req = Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("The capital of France is", " Paris"),
            idx=0,
        )

        with pytest.warns(UserWarning, match="does not support logprobs"):
            results = adapter.loglikelihood([req])

        assert len(results) == 1
        assert results[0] == (0.0, False)

    def test_loglikelihood_rolling_fallback(self, llm):
        from lm_eval.api.instance import Instance
        from unified_llm.benchmark.adapter import UnifiedLMAdapter

        adapter = UnifiedLMAdapter(
            client=llm,
            model_id="echo/echo",
            logprobs_supported=False,
        )

        req = Instance(
            request_type="loglikelihood_rolling",
            doc={},
            arguments=("Some text to score",),
            idx=0,
        )

        with pytest.warns(UserWarning, match="does not support logprobs"):
            results = adapter.loglikelihood_rolling([req])

        assert len(results) == 1
        assert results[0] == (0.0, False)

    def test_adapter_properties(self, llm):
        from unified_llm.benchmark.adapter import UnifiedLMAdapter

        adapter = UnifiedLMAdapter(
            client=llm,
            model_id="echo/echo",
            batch_size=4,
            max_gen_toks=512,
        )
        assert adapter.batch_size == 4
        assert adapter.max_gen_toks == 512
        assert adapter.device == "cpu"
        assert adapter.max_length == 4096

    def test_logprobs_detection(self):
        from unified_llm.benchmark.adapter import _detect_logprobs_support

        assert _detect_logprobs_support("openai/gpt-4o") is True
        assert _detect_logprobs_support("ollama/llama3.1") is True
        assert _detect_logprobs_support("anthropic/claude-3") is False
        assert _detect_logprobs_support("google/gemini-pro") is False


# ---------------------------------------------------------------------------
# BenchmarkRunner (requires lm_eval)
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    @pytest.fixture(autouse=True)
    def _skip_without_lm_eval(self):
        pytest.importorskip("lm_eval")

    def test_filter_generate_only_tasks(self):
        from unified_llm.benchmark.runner import BenchmarkRunner

        tasks = ["mmlu", "gsm8k", "hellaswag", "humaneval", "triviaqa"]
        filtered = BenchmarkRunner._filter_generate_only_tasks(tasks)
        assert "gsm8k" in filtered
        assert "humaneval" in filtered
        assert "triviaqa" in filtered
        assert "mmlu" not in filtered
        assert "hellaswag" not in filtered

    def test_parse_results(self):
        from unified_llm.benchmark.runner import BenchmarkRunner

        raw = {
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": 0.45,
                    "exact_match_stderr,strict-match": 0.02,
                    "alias": "gsm8k",
                    "higher_is_better": {"exact_match": True},
                },
            },
            "n-samples": {"gsm8k": {"original": 1319, "effective": 100}},
            "n-shot": {"gsm8k": 5},
            "versions": {"gsm8k": 3},
            "higher_is_better": {},
        }
        report = BenchmarkRunner._parse_results("test/model", raw)
        assert report.model_id == "test/model"
        assert "gsm8k" in report.task_results
        tr = report.task_results["gsm8k"]
        assert tr.metrics["exact_match"] == 0.45
        assert "exact_match_stderr" not in tr.metrics
        assert tr.num_samples == 100
        assert tr.num_fewshot == 5
        assert tr.version == 3

    def test_run_no_tasks_raises(self, llm):
        from unified_llm.benchmark.runner import BenchmarkRunner

        runner = BenchmarkRunner(llm)
        with pytest.raises(ValueError, match="No tasks specified"):
            runner.run("echo/echo", BenchmarkConfig())

    def test_progress_callback(self, llm):
        from unified_llm.benchmark.runner import BenchmarkRunner

        messages = []
        runner = BenchmarkRunner(llm, progress=lambda msg: messages.append(msg))
        # We can't run a full eval without lm-eval tasks, but we can verify
        # the runner stores the callback
        assert runner._progress is not None
        runner._progress("test message")
        assert "test message" in messages


# ---------------------------------------------------------------------------
# save_json
# ---------------------------------------------------------------------------


class TestSaveJson:
    def test_save_json(self, tmp_path):
        report = BenchmarkReport(
            model_id="test/model",
            task_results={
                "gsm8k": TaskResult(
                    task_name="gsm8k",
                    metrics={"exact_match": 0.45},
                    num_samples=100,
                ),
            },
        )
        out = report.save_json(tmp_path / "results.json")
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["model_id"] == "test/model"
        assert data["task_results"]["gsm8k"]["metrics"]["exact_match"] == 0.45
        assert data["task_results"]["gsm8k"]["num_samples"] == 100

    def test_save_json_creates_dirs(self, tmp_path):
        report = BenchmarkReport(model_id="test/model")
        out = report.save_json(tmp_path / "nested" / "dir" / "results.json")
        assert out.exists()


# ---------------------------------------------------------------------------
# comparison_table
# ---------------------------------------------------------------------------


class TestComparisonTable:
    def test_empty(self):
        assert comparison_table([]) == "No reports to compare."

    def test_single_report_delegates(self):
        report = BenchmarkReport(
            model_id="test/model",
            task_results={
                "gsm8k": TaskResult(task_name="gsm8k", metrics={"exact_match": 0.5}, num_samples=10),
            },
        )
        result = comparison_table([report])
        # Should return the normal summary_table
        assert "test/model" in result

    def test_two_models(self):
        r1 = BenchmarkReport(
            model_id="model_a",
            task_results={
                "gsm8k": TaskResult(task_name="gsm8k", metrics={"exact_match": 0.40}, num_samples=10),
                "drop": TaskResult(task_name="drop", metrics={"f1": 0.55}, num_samples=10),
            },
        )
        r2 = BenchmarkReport(
            model_id="model_b",
            task_results={
                "gsm8k": TaskResult(task_name="gsm8k", metrics={"exact_match": 0.60}, num_samples=10),
                "drop": TaskResult(task_name="drop", metrics={"f1": 0.70}, num_samples=10),
            },
        )
        table = comparison_table([r1, r2])
        assert "Comparison" in table
        assert "model_a" in table
        assert "model_b" in table
        assert "gsm8k" in table
        assert "drop" in table
        assert "0.4000" in table
        assert "0.6000" in table

    def test_filters_subtasks(self):
        r1 = BenchmarkReport(
            model_id="model_a",
            task_results={
                "bbh": TaskResult(task_name="bbh", metrics={"exact_match": 0.30}, num_samples=0),
                "bbh_sub1": TaskResult(task_name="bbh_sub1", metrics={"exact_match": 0.40}, num_samples=5),
                "gsm8k": TaskResult(task_name="gsm8k", metrics={"exact_match": 0.50}, num_samples=10),
            },
        )
        table = comparison_table([r1, r1])
        # Should show bbh and gsm8k, but not bbh_sub1
        assert "bbh" in table
        assert "gsm8k" in table
        assert "bbh_sub1" not in table

    def test_missing_task_shows_dash(self):
        r1 = BenchmarkReport(
            model_id="model_a",
            task_results={
                "gsm8k": TaskResult(task_name="gsm8k", metrics={"exact_match": 0.40}, num_samples=10),
            },
        )
        r2 = BenchmarkReport(
            model_id="model_b",
            task_results={
                "gsm8k": TaskResult(task_name="gsm8k", metrics={"exact_match": 0.60}, num_samples=10),
                "drop": TaskResult(task_name="drop", metrics={"f1": 0.70}, num_samples=10),
            },
        )
        table = comparison_table([r1, r2])
        assert "-" in table  # model_a doesn't have drop
