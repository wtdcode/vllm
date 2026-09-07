# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Repetition inside reasoning spends the budget instead of ending the turn.

The scheduler stops ending these requests (see
tests/v1/core/test_repetition_detection.py), so something has to close the
block -- otherwise the loop just runs to the budget. That is this holder's job.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

pytestmark = pytest.mark.cpu_test

START, END = [900], [901]
DETECT = RepetitionDetectionParams(max_pattern_size=5, min_pattern_size=2, min_count=3)


def holder():
    return ThinkingBudgetStateHolder(
        # The holder reads only these two attributes; ReasoningConfig fills
        # them from the tokenizer, which this unit test has no use for.
        reasoning_config=SimpleNamespace(
            reasoning_start_token_ids=START, reasoning_end_token_ids=END
        ),
        max_num_seqs=4,
        num_spec_tokens=0,
        device=torch.device("cpu"),
        is_pin_memory=False,
    )


def state(output, *, detect=DETECT, in_end=False, budget=1000):
    return {
        "repetition_detection": detect,
        "output_tok_ids": output,
        "in_think": True,
        "in_end": in_end,
        "start_thinking": 0,
        "check_count_down": budget,
    }


class TestReasoningRepetitionDetection:
    def test_looping_reasoning_is_reported(self):
        assert holder()._reasoning_is_repeating(state([10, 20] * 4))

    def test_varied_reasoning_is_not(self):
        assert not holder()._reasoning_is_repeating(state([1, 2, 3, 4, 5, 6, 7, 8]))

    def test_no_detection_params_is_not(self):
        assert not holder()._reasoning_is_repeating(state([10, 20] * 4, detect=None))

    def test_block_already_closing_is_not(self):
        # in_end means the forcing path is already running; re-arming it would
        # reset the countdown mid-close.
        assert not holder()._reasoning_is_repeating(state([10, 20] * 4, in_end=True))

    def test_empty_output_is_not(self):
        assert not holder()._reasoning_is_repeating(state([]))
