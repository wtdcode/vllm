# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for thinking_token_budget propagation from default_sampling_params
to SamplingParams in ChatCompletionRequest and CompletionRequest.

Same bug class as https://github.com/vllm-project/vllm/issues/22519 (fixed for
stop_token_ids in tests/entrypoints/openai/test_stop_token_ids.py): a value set
at server startup via --override-generation-config landed in
default_sampling_params but was silently discarded on every request, because
to_sampling_params() passed self.thinking_token_budget straight through instead
of falling back to the defaults.
"""

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
)


class TestChatCompletionThinkingTokenBudget:
    """thinking_token_budget defaulting in ChatCompletionRequest."""

    @pytest.fixture
    def minimal_chat_request(self):
        return ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

    def test_default_thinking_token_budget_applied(self, minimal_chat_request):
        """Server-default thinking_token_budget is applied when client sends none."""
        sampling_params = minimal_chat_request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"thinking_token_budget": 128},
        )

        assert sampling_params.thinking_token_budget == 128

    def test_client_thinking_token_budget_overrides_default(self):
        """An explicit per-request value wins over the server default."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            thinking_token_budget=1024,
        )

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"thinking_token_budget": 128},
        )

        assert sampling_params.thinking_token_budget == 1024

    def test_no_default_and_no_request_value_stays_none(self, minimal_chat_request):
        """Absent both, the field stays None (no budget enforced)."""
        sampling_params = minimal_chat_request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={},
        )

        assert sampling_params.thinking_token_budget is None

    def test_client_zero_budget_not_swallowed_by_default(self):
        """0 is a meaningful budget and must not fall through to the server default."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            thinking_token_budget=0,
        )

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"thinking_token_budget": 128},
        )

        assert sampling_params.thinking_token_budget == 0


class TestCompletionThinkingTokenBudget:
    """thinking_token_budget defaulting in CompletionRequest."""

    @pytest.fixture
    def minimal_completion_request(self):
        return CompletionRequest(
            model="test-model",
            prompt="hello",
        )

    def test_default_thinking_token_budget_applied(self, minimal_completion_request):
        """Server-default thinking_token_budget is applied when client sends none."""
        sampling_params = minimal_completion_request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"thinking_token_budget": 128},
        )

        assert sampling_params.thinking_token_budget == 128

    def test_client_thinking_token_budget_overrides_default(self):
        """An explicit per-request value wins over the server default."""
        request = CompletionRequest(
            model="test-model",
            prompt="hello",
            thinking_token_budget=1024,
        )

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"thinking_token_budget": 128},
        )

        assert sampling_params.thinking_token_budget == 1024

    def test_no_default_and_no_request_value_stays_none(
        self, minimal_completion_request
    ):
        """Absent both, the field stays None (no budget enforced)."""
        sampling_params = minimal_completion_request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={},
        )

        assert sampling_params.thinking_token_budget is None

    def test_client_zero_budget_not_swallowed_by_default(self):
        """0 is a meaningful budget and must not fall through to the server default."""
        request = CompletionRequest(
            model="test-model",
            prompt="hello",
            thinking_token_budget=0,
        )

        sampling_params = request.to_sampling_params(
            max_tokens=100,
            default_sampling_params={"thinking_token_budget": 128},
        )

        assert sampling_params.thinking_token_budget == 0


class TestResponsesApiThinkingBudget:
    """The Responses API needs the same server-default merge.

    Upstream #54469 only touched chat_completion and completion, but the
    traffic that actually overthinks here is Responses -- unbounded
    max_output_tokens, reasoning running to the cap. Without this the
    server default is silently ignored on exactly the path it was set for.
    """

    DEFAULTS = {"thinking_token_budget": 30000, "top_p": 0.95}

    def _params(self, **kw):
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

        req = ResponsesRequest(input="hi", **kw)
        return req.to_sampling_params(
            default_max_tokens=1000, default_sampling_params=self.DEFAULTS
        )

    def test_server_default_applies_when_request_omits_it(self):
        assert self._params().thinking_token_budget == 30000

    def test_request_value_wins(self):
        assert self._params(thinking_token_budget=5000).thinking_token_budget == 5000

    def test_request_zero_wins_over_server_default(self):
        # 0 means "no reasoning at all" and must not be treated as unset
        assert self._params(thinking_token_budget=0).thinking_token_budget == 0

    def test_no_server_default_leaves_it_unset(self):
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

        req = ResponsesRequest(input="hi")
        sp = req.to_sampling_params(default_max_tokens=1000, default_sampling_params={})
        assert sp.thinking_token_budget is None


class TestThinkingBudgetScalesToOutputCap:
    """The server default is scaled to whatever output cap the request named.

    A flat server budget is either dead weight or actively harmful: with a
    client that caps output at 32000, a 30000 budget guarantees the answer is
    truncated, and with a client that names no cap at all the reasoning can run
    to the whole context window. Replaying one capped request against the
    upstream API 20 times put its output at a 29049 median with 40% over 32000,
    so the budget has to leave room for the answer rather than sit just under
    the client's cap.
    """

    DEFAULTS = {"thinking_token_budget": 64000}

    def _responses(self, **kw):
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

        return ResponsesRequest(input="hi", **kw).to_sampling_params(
            default_max_tokens=1000, default_sampling_params=self.DEFAULTS
        )

    def _chat(self, **kw):
        return ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            **kw,
        ).to_sampling_params(max_tokens=100, default_sampling_params=self.DEFAULTS)

    def _completion(self, **kw):
        return CompletionRequest(
            model="test-model", prompt="hello", **kw
        ).to_sampling_params(max_tokens=100, default_sampling_params=self.DEFAULTS)

    def test_no_output_cap_uses_the_flat_default(self):
        assert self._responses().thinking_token_budget == 64000

    def test_output_cap_scales_the_budget(self):
        # the shape claude-cli sends: 32000 out, so 24000 to think and 8000 to answer
        assert self._responses(max_output_tokens=32000).thinking_token_budget == 24000

    def test_huge_output_cap_hits_the_ceiling(self):
        got = self._responses(max_output_tokens=1_000_000).thinking_token_budget
        assert got == 128_000

    def test_request_budget_still_wins_over_scaling(self):
        got = self._responses(max_output_tokens=32000, thinking_token_budget=5000)
        assert got.thinking_token_budget == 5000

    def test_request_zero_still_disables_reasoning(self):
        got = self._responses(max_output_tokens=32000, thinking_token_budget=0)
        assert got.thinking_token_budget == 0

    def test_chat_completion_scales_on_max_tokens(self):
        assert self._chat(max_tokens=32000).thinking_token_budget == 24000

    def test_chat_completion_prefers_max_completion_tokens(self):
        got = self._chat(max_tokens=32000, max_completion_tokens=8000)
        assert got.thinking_token_budget == 6000

    def test_completion_scales_on_max_tokens(self):
        assert self._completion(max_tokens=32000).thinking_token_budget == 24000

    def test_completion_legacy_default_cap_does_not_scale(self):
        # max_tokens carries OpenAI's legacy default of 16, which must not be
        # read as the caller naming a 16-token cap
        assert self._completion().thinking_token_budget == 64000

    def test_all_three_apis_agree_on_the_same_cap(self):
        # an inconsistent budget across endpoints is worse than none at all
        assert (
            self._responses(max_output_tokens=32000).thinking_token_budget
            == self._chat(max_tokens=32000).thinking_token_budget
            == self._completion(max_tokens=32000).thinking_token_budget
            == 24000
        )

    @pytest.mark.parametrize(
        "cap,expected", [(None, 64000), (32000, 24000), (8000, 6000), (10**6, 128000)]
    )
    def test_every_endpoint_resolves_a_cap_the_same_way(self, cap, expected):
        """One policy, one function, no endpoint drifting from the others."""
        chat = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            **({"max_completion_tokens": cap} if cap else {}),
        )
        completion = CompletionRequest(
            model="test-model", prompt="hello", **({"max_tokens": cap} if cap else {})
        )
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

        responses = ResponsesRequest(
            input="hi", **({"max_output_tokens": cap} if cap else {})
        )

        assert chat.named_output_cap() == cap
        assert completion.named_output_cap() == cap
        assert responses.named_output_cap() == cap

        got = {
            chat.to_sampling_params(
                max_tokens=100, default_sampling_params=self.DEFAULTS
            ).thinking_token_budget,
            completion.to_sampling_params(
                max_tokens=100, default_sampling_params=self.DEFAULTS
            ).thinking_token_budget,
            responses.to_sampling_params(
                default_max_tokens=100, default_sampling_params=self.DEFAULTS
            ).thinking_token_budget,
        }
        assert got == {expected}

    def test_legacy_max_tokens_default_is_preserved(self):
        # narrowing normalize_null_max_tokens must not change the field itself
        assert CompletionRequest(model="test-model", prompt="hello").max_tokens == 16
        explicit_null = CompletionRequest(
            model="test-model", prompt="hello", max_tokens=None
        )
        assert explicit_null.max_tokens == 16

    def test_no_server_default_leaves_reasoning_unbounded(self):
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

        sp = ResponsesRequest(input="hi", max_output_tokens=32000).to_sampling_params(
            default_max_tokens=1000, default_sampling_params={}
        )
        assert sp.thinking_token_budget is None

    def test_server_default_zero_disables_reasoning_regardless_of_cap(self):
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

        sp = ResponsesRequest(input="hi", max_output_tokens=32000).to_sampling_params(
            default_max_tokens=1000,
            default_sampling_params={"thinking_token_budget": 0},
        )
        assert sp.thinking_token_budget == 0
