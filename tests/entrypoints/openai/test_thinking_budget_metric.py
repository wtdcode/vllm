# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The overthinking counter, measured the way /metrics sees it.

Both halves of this have been wrong in production before: the counter was
registered at import and swept out of the registry before the server took a
request, and the budget it compared against came from a parser object that had
never seen the request. So assert against the exposed text, after the sweep.
"""

import pytest
from prometheus_client import REGISTRY, generate_latest

from vllm.entrypoints.openai.thinking_budget_metric import note_reasoning_length
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

METRIC = "vllm:reasoning_budget_exhausted_total"
DEFAULTS = {"thinking_token_budget": 64000}


def exposed_count(model_name="m"):
    prefix = f'{METRIC}{{model_name="{model_name}"}}'
    for line in generate_latest(REGISTRY).decode().splitlines():
        if line.startswith(prefix):
            return float(line.rsplit(" ", 1)[1])
    return None


class Request:
    def __init__(self, cap=None, budget=None):
        self._cap = cap
        self.thinking_token_budget = budget

    def named_output_cap(self):
        return self._cap


@pytest.fixture(scope="module", autouse=True)
def _sweep():
    # Once, before any request -- exactly when PrometheusStatLogger does it.
    # Sweeping between requests would detach the counter for good, which is
    # the failure this module exists to catch.
    unregister_vllm_metrics()


def note(model, tokens, request, defaults=DEFAULTS):
    before = exposed_count(model) or 0.0
    note_reasoning_length(model, tokens, request, defaults)
    return (exposed_count(model) or 0.0) - before


def test_survives_the_registry_sweep():
    assert note("sweep", 64000, Request()) == 1.0
    assert exposed_count("sweep") == 1.0


@pytest.mark.parametrize(
    "name,tokens,request_,expected",
    [
        # a 32000-token cap runs under 24000, the shape this traffic has
        ("capped_hit", 24000, Request(cap=32000), 1.0),
        ("capped_one_short", 23999, Request(cap=32000), 1.0),
        ("capped_under", 20000, Request(cap=32000), 0.0),
        # no cap named: the flat server default
        ("flat_hit", 64000, Request(), 1.0),
        ("flat_under", 40000, Request(), 0.0),
        # the ceiling, not 75% of a huge cap
        ("ceiling", 128000, Request(cap=10**6), 1.0),
        # an explicit per-request budget is what it runs under
        ("explicit", 5000, Request(cap=32000, budget=5000), 1.0),
        ("explicit_zero", 0, Request(cap=32000, budget=0), 0.0),
    ],
)
def test_counts_against_the_budget_the_request_ran_under(
    name, tokens, request_, expected
):
    assert note(name, tokens, request_) == expected


def test_no_server_default_never_counts():
    assert note("nodefault", 500000, Request(cap=32000), defaults={}) == 0.0
