# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""`repetition` must not reach clients as a finish reason.

None of the three API schemas has a value for it. The Anthropic layer maps
finish reasons through a dict and emits `"stop_reason": null` for anything
missing, and a Claude SDK given a null stop_reason reports a successful turn
carrying no result -- observed in production with thinking-only responses.
The engine keeps the precise reason for metrics and for `stop_reason`.
"""

import pytest

from vllm.v1.engine import FinishReason
from vllm.v1.engine.output_processor import _client_finish_reason

pytestmark = pytest.mark.cpu_test


def test_repetition_is_reported_as_stop():
    assert _client_finish_reason(FinishReason.REPETITION) == "stop"


@pytest.mark.parametrize(
    "reason,expected",
    [
        (FinishReason.STOP, "stop"),
        (FinishReason.LENGTH, "length"),
        (FinishReason.ABORT, "abort"),
        (FinishReason.ERROR, "error"),
    ],
)
def test_every_other_reason_is_passed_through(reason, expected):
    assert _client_finish_reason(reason) == expected


def test_engine_value_is_left_alone_for_metrics():
    # The metrics label reads the enum, not this string; keep them distinct so
    # `vllm:request_success_total{finished_reason="repetition"}` stays usable.
    assert str(FinishReason.REPETITION) == "repetition"


def test_every_engine_reason_maps_to_something():
    for reason in FinishReason:
        assert _client_finish_reason(reason)
