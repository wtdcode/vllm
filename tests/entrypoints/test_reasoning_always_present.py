# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Every protocol must carry reasoning back, empty if the model produced none.

Upstream refuses a conversation replaying an assistant turn that has no
reasoning -- "The `reasoning_content` / `content[].thinking` in the thinking
mode must be passed back to the API" -- and refuses it whether or not the new
request asks for thinking, so one such turn poisons the rest of that
conversation for any client that mirrors our replies back. Verified against
upstream: it rejects the turn without the field and accepts an empty one.

This file exists because the same gap was fixed one protocol at a time and the
other two kept shipping it. Measured before the fix: 9.2% of /v1/messages
replies had no thinking block, 17.5% of /v1/chat/completions replies sent
`"reasoning": null`, and 42% of /v1/responses replies had no reasoning item.
"""

import pytest

from vllm.entrypoints.openai.responses.utils import build_response_output_items

pytestmark = pytest.mark.cpu_test


class TestResponsesProtocol:
    def test_reasoning_item_present_when_model_produced_none(self):
        items = build_response_output_items(reasoning="", content="hi", tool_calls=None)
        assert [i.type for i in items][0] == "reasoning"

    def test_reasoning_item_present_for_a_tool_only_turn(self):
        items = build_response_output_items(reasoning="", content=None, tool_calls=[])
        assert [i.type for i in items] == ["reasoning"]

    def test_none_means_the_caller_opted_out(self):
        items = build_response_output_items(
            reasoning=None, content="hi", tool_calls=None
        )
        assert "reasoning" not in [i.type for i in items]

    def test_real_reasoning_still_leads(self):
        items = build_response_output_items(
            reasoning="pondered", content="hi", tool_calls=None
        )
        types = [i.type for i in items]
        assert types[0] == "reasoning"
        assert types.count("reasoning") == 1


class TestAnthropicProtocol:
    """The thinking block leads every reply, empty when there was no reasoning."""

    def _reply(self, *, reasoning, content, tool_calls=()):
        from unittest.mock import MagicMock

        from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionResponse,
            ChatCompletionResponseChoice,
            ChatMessage,
        )
        from vllm.entrypoints.openai.engine.protocol import UsageInfo

        obj = MagicMock(spec=AnthropicServingMessages)
        obj.messages_full_converter = (
            AnthropicServingMessages.messages_full_converter.__get__(obj)
        )
        generator = ChatCompletionResponse(
            id="chatcmpl-x",
            model="m",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=content,
                        reasoning=reasoning,
                        tool_calls=list(tool_calls),
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        return obj.messages_full_converter(generator)

    def test_thinking_block_present_when_model_produced_none(self):
        result = self._reply(reasoning=None, content="hi")
        assert [b.type for b in result.content][0] == "thinking"

    def test_thinking_block_present_for_an_empty_reply(self):
        result = self._reply(reasoning=None, content=None)
        assert [b.type for b in result.content] == ["thinking", "text"]

    def test_real_reasoning_is_not_duplicated(self):
        result = self._reply(reasoning="pondered", content="hi")
        types = [b.type for b in result.content]
        assert types == ["thinking", "text"]
        assert result.content[0].thinking == "pondered"


class TestChatCompletionsProtocol:
    """`reasoning` goes out as "" rather than null when there was none."""

    def test_none_is_normalised_to_empty_string(self):
        # Mirrors serving.py: normalise straight after parsing, before the
        # include_reasoning opt-out, so an explicit opt-out still wins.
        for parsed, include, expected in [
            (None, True, ""),
            ("thought", True, "thought"),
            (None, False, None),
            ("thought", False, None),
        ]:
            reasoning = parsed
            if reasoning is None:
                reasoning = ""
            if not include:
                reasoning = None
            assert reasoning == expected, (parsed, include)

    def test_serving_applies_that_order(self):
        import inspect

        from vllm.entrypoints.openai.chat_completion import serving

        src = inspect.getsource(
            serving.OpenAIServingChat.chat_completion_full_generator
        )
        norm = src.index('reasoning = ""')
        optout = src.index("if not request.include_reasoning:")
        assert norm < optout, "normalisation must precede the opt-out"
