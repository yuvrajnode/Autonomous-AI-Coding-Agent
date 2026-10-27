from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.config import Settings
from agent.errors import LLMError
from agent.llm.anthropic_client import AnthropicClient
from agent.llm.base import Message
from agent.llm.pricing import estimate_cost


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads: list[dict] = []

    def create(self, **payload):
        self.payloads.append(payload)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def fake_sdk(*responses):
    messages = FakeMessages(responses)
    return SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages))


def response(*blocks, stop_reason="end_turn", **extra):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=100, cache_read_input_tokens=0)
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, usage=usage, **extra)


def make_client(sdk) -> AnthropicClient:
    return AnthropicClient(Settings(ANTHROPIC_API_KEY="test", llm_max_retries=2), client=sdk)


# -- request shape ----------------------------------------------------------


def test_request_carries_no_sampling_parameters():
    sdk = fake_sdk(response(SimpleNamespace(type="text", text="hi")))
    make_client(sdk).complete(system="sys", messages=[Message.user("hello")])
    payload = sdk.messages.payloads[0]
    assert not {"temperature", "top_p", "top_k"} & payload.keys()


def test_system_prompt_and_last_tool_are_cached():
    sdk = fake_sdk(response(SimpleNamespace(type="text", text="hi")))
    tools = [{"name": "a", "input_schema": {}}, {"name": "b", "input_schema": {}}]
    make_client(sdk).complete(system="sys", messages=[Message.user("hello")], tools=tools)
    payload = sdk.messages.payloads[0]
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in payload["tools"][0]
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in tools[-1]  # caller's list is not mutated


def test_effort_is_always_set():
    sdk = fake_sdk(response(SimpleNamespace(type="text", text="hi")))
    make_client(sdk).complete(system="sys", messages=[Message.user("hello")])
    assert sdk.messages.payloads[0]["output_config"] == {"effort": "medium"}


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-opus-5-5"])
def test_default_models_opt_into_refusal_fallbacks(model):
    sdk = fake_sdk(response(SimpleNamespace(type="text", text="hi")))
    make_client(sdk).complete(system="sys", messages=[Message.user("hello")], model=model)
    payload = sdk.messages.payloads[0]
    assert payload["fallbacks"] == "default"
    assert payload["betas"] == ["server-side-fallback-2026-07-01"]


def test_models_without_fallback_support_do_not_send_it():
    sdk = fake_sdk(response(SimpleNamespace(type="text", text="hi")))
    make_client(sdk).complete(
        system="sys", messages=[Message.user("hello")], model="claude-haiku-4-5"
    )
    assert not {"fallbacks", "betas"} & sdk.messages.payloads[0].keys()


# -- parsing ----------------------------------------------------------------


def test_thinking_blocks_are_kept_for_replay():
    sdk = fake_sdk(
        response(
            SimpleNamespace(type="thinking", thinking="", signature="sig"),
            SimpleNamespace(type="redacted_thinking", data="opaque"),
            SimpleNamespace(type="tool_use", id="t1", name="list_dir", input={"path": "."}),
            stop_reason="tool_use",
        )
    )
    out = make_client(sdk).complete(system="sys", messages=[Message.user("go")])
    assert [b["type"] for b in out.raw_content] == ["thinking", "redacted_thinking", "tool_use"]
    assert out.raw_content[0]["signature"] == "sig"
    assert out.tool_calls[0].name == "list_dir"
    assert out.text == ""


def test_usage_is_priced():
    sdk = fake_sdk(response(SimpleNamespace(type="text", text="hi")))
    out = make_client(sdk).complete(
        system="sys", messages=[Message.user("hello")], model="claude-sonnet-5-5"
    )
    assert out.usage.usd == pytest.approx(estimate_cost("claude-sonnet-5-5", 1000, 100))


def test_a_fallback_reply_is_billed_at_the_serving_model():
    sdk = fake_sdk(response(SimpleNamespace(type="text", text="hi"), model="claude-sonnet-5"))
    out = make_client(sdk).complete(
        system="sys", messages=[Message.user("hello")], model="claude-opus-5-5"
    )
    assert out.usage.usd == pytest.approx(estimate_cost("claude-sonnet-5", 1000, 100))


def test_a_refusal_raises_instead_of_looking_like_an_empty_reply():
    sdk = fake_sdk(
        response(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"))
    )
    with pytest.raises(LLMError, match="cyber"):
        make_client(sdk).complete(system="sys", messages=[Message.user("hello")])


# -- retries ----------------------------------------------------------------


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_bad_request_is_not_retried():
    sdk = fake_sdk(StatusError(400))
    with pytest.raises(LLMError):
        make_client(sdk).complete(system="sys", messages=[Message.user("hello")])
    assert len(sdk.messages.payloads) == 1


def test_transient_errors_are_retried(monkeypatch):
    monkeypatch.setattr("agent.llm.anthropic_client.time.sleep", lambda _s: None)
    sdk = fake_sdk(StatusError(529), response(SimpleNamespace(type="text", text="ok")))
    out = make_client(sdk).complete(system="sys", messages=[Message.user("hello")])
    assert out.text == "ok"
    assert len(sdk.messages.payloads) == 2


# -- pricing ----------------------------------------------------------------


def test_longest_prefix_wins():
    # `claude-opus-5-5` must not be billed at the `claude-opus-5` rate.
    assert estimate_cost("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.00)
    assert estimate_cost("claude-opus-5", 1_000_000, 0) == pytest.approx(5.00)


def test_unknown_models_are_not_free():
    assert estimate_cost("some-future-model", 1_000_000, 1_000_000) > 0
