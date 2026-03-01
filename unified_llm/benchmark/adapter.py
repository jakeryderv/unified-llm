"""Bridge between UnifiedLLM and lm-evaluation-harness's LM base class."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

try:
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
except ImportError as e:
    raise ImportError(
        "lm-eval is required for benchmarking. Install with: pip install -e '.[benchmark]'"
    ) from e

if TYPE_CHECKING:
    from unified_llm.client import UnifiedLLM

logger = logging.getLogger(__name__)

# Providers known to support logprobs in their API responses.
_LOGPROBS_PROVIDERS = {"openai", "ollama"}


def _detect_logprobs_support(model_id: str) -> bool:
    """Heuristic: check if the provider portion of model_id supports logprobs."""
    provider = model_id.split("/")[0].lower()
    return provider in _LOGPROBS_PROVIDERS


class UnifiedLMAdapter(LM):
    """
    Adapter that lets lm-evaluation-harness call any model through UnifiedLLM.

    Subclasses ``lm_eval.api.model.LM`` directly (not ``TemplateAPI``), since we
    don't make raw HTTP calls — we go through the unified ``complete()`` method.
    """

    def __init__(
        self,
        client: UnifiedLLM,
        model_id: str,
        batch_size: int | str = 1,
        max_gen_toks: int = 1024,
        logprobs_supported: bool | None = None,
    ):
        super().__init__()
        self.client = client
        self.model_id = model_id
        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks
        self.logprobs_supported = (
            logprobs_supported
            if logprobs_supported is not None
            else _detect_logprobs_support(model_id)
        )
        self._warned_logprobs = False

    # ------------------------------------------------------------------
    # Properties required by LM
    # ------------------------------------------------------------------

    @property
    def eot_token_id(self):
        return None

    @property
    def max_length(self):
        return 4096

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return int(self._batch_size) if isinstance(self._batch_size, str) else self._batch_size

    @property
    def device(self):
        return "cpu"

    # ------------------------------------------------------------------
    # generate_until — for generation-based benchmarks
    # ------------------------------------------------------------------

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate text for each request, stopping at specified sequences."""
        from unified_llm.types import GenerationConfig

        results: list[str] = []
        for req in requests:
            context = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}

            stop_sequences = gen_kwargs.get("until", [])
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]

            max_tokens = gen_kwargs.get("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)
            do_sample = gen_kwargs.get("do_sample", False)
            if not do_sample:
                temperature = 0.0

            config = GenerationConfig(
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop_sequences,
            )

            try:
                response = self.client.complete(self.model_id, context, config)
                text = response.content

                # Truncate at first stop sequence (in case the model didn't stop)
                for stop in stop_sequences:
                    idx = text.find(stop)
                    if idx != -1:
                        text = text[:idx]

                results.append(text)
            except Exception as e:
                logger.error("generate_until failed for model %s: %s", self.model_id, e)
                results.append("")

        return results

    # ------------------------------------------------------------------
    # loglikelihood — for multiple-choice benchmarks
    # ------------------------------------------------------------------

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        Compute log-likelihood of a continuation given a context.

        When the provider supports logprobs, we request them via the API.
        Otherwise, we return dummy values with a warning.
        """
        if not self.logprobs_supported:
            return self._loglikelihood_fallback(requests)

        return self._loglikelihood_with_logprobs(requests)

    def _loglikelihood_with_logprobs(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Use the provider's logprobs support to compute log-likelihoods."""
        from unified_llm.types import GenerationConfig

        results: list[tuple[float, bool]] = []
        for req in requests:
            context = req.args[0]
            continuation = req.args[1]
            prompt = context + continuation

            config = GenerationConfig(
                temperature=0.0,
                max_tokens=1,
                extra={"logprobs": True, "echo": True},
            )

            try:
                response = self.client.complete(self.model_id, prompt, config)
                logprobs_data = response.metadata.get("logprobs")

                if logprobs_data and isinstance(logprobs_data, dict):
                    token_logprobs = logprobs_data.get("token_logprobs", [])
                    tokens = logprobs_data.get("tokens", [])
                    # Sum logprobs for continuation tokens
                    # Approximate: use the last N tokens where N ≈ len(continuation.split())
                    if token_logprobs:
                        # Take the average of all available logprobs as an approximation
                        valid = [lp for lp in token_logprobs if lp is not None]
                        total = sum(valid) if valid else 0.0
                        is_greedy = all(
                            lp == max(token_logprobs) for lp in valid
                        ) if valid else False
                        results.append((total, is_greedy))
                    else:
                        results.append((0.0, False))
                else:
                    results.append((0.0, False))
            except Exception as e:
                logger.error("loglikelihood failed for model %s: %s", self.model_id, e)
                results.append((0.0, False))

        return results

    def _loglikelihood_fallback(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Return dummy values when logprobs are not supported."""
        if not self._warned_logprobs:
            warnings.warn(
                f"Provider for '{self.model_id}' does not support logprobs. "
                f"Loglikelihood-based benchmarks will return dummy values. "
                f"Use generate_only=True to filter to generation-based tasks only.",
                UserWarning,
                stacklevel=2,
            )
            self._warned_logprobs = True
        return [(0.0, False)] * len(requests)

    # ------------------------------------------------------------------
    # loglikelihood_rolling — perplexity-style tasks
    # ------------------------------------------------------------------

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Rolling log-likelihood (perplexity). Same strategy as loglikelihood."""
        if not self.logprobs_supported:
            if not self._warned_logprobs:
                warnings.warn(
                    f"Provider for '{self.model_id}' does not support logprobs. "
                    f"Rolling loglikelihood will return dummy values.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_logprobs = True
            return [(0.0, False)] * len(requests)

        return self._loglikelihood_with_logprobs(requests)
