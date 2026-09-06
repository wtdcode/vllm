# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Counts responses whose reasoning ran into the server's thinking budget.

Counted here rather than in the parser: the budget a request runs under is
derived from the request, and the parser chain reaches the request and the
reasoning length on different objects -- the reasoning adapter counts a
non-streamed response through a second engine it builds on the spot, which
never saw the request at all. The serving layer holds both.
"""

from typing import Any

from vllm.sampling_params import resolve_thinking_token_budget

_COUNTER: Any = None
_COUNTER_UNAVAILABLE = False


def _counter() -> Any:
    """Build the counter on first use.

    Not at import: PrometheusStatLogger drops every collector named ``vllm:*``
    from the default registry as it sets vLLM's own metrics up, so anything
    registered while modules are still loading is gone before the server takes
    its first request.
    """
    global _COUNTER, _COUNTER_UNAVAILABLE
    if _COUNTER is None and not _COUNTER_UNAVAILABLE:
        try:
            from prometheus_client import Counter

            _COUNTER = Counter(
                "vllm:reasoning_budget_exhausted_total",
                "Responses whose reasoning was stopped by the thinking budget.",
                ["model_name"],
            )
        except Exception:  # prometheus_client absent, or already registered
            _COUNTER_UNAVAILABLE = True
    return _COUNTER


def note_reasoning_length(
    model_name: str,
    reasoning_tokens: int,
    request: Any,
    default_sampling_params: dict[str, Any],
) -> None:
    """Count this response if its reasoning ran into the budget it ran under."""
    budget = resolve_thinking_token_budget(
        getattr(request, "thinking_token_budget", None),
        default_sampling_params.get("thinking_token_budget"),
        request.named_output_cap(),
    )
    # The sampler spends the last token of the budget on the forced think-end
    # marker, which is not counted as reasoning, so a stopped segment can land
    # one short of the budget.
    if not budget or reasoning_tokens < budget - 1:
        return
    counter = _counter()
    if counter is not None:
        counter.labels(model_name=model_name).inc()
