import pytest

from recipes.meta_harness.examples.terminal_bench.history import (
    HistoryBinding,
    ProposerBudgetReached,
    ProposerUsageUnknown,
)
from recipes.meta_harness.examples.terminal_bench.proposer_usage import response_cost

PRICING = {
    "rates": {
        "input_cost_per_token": 4e-6,
        "output_cost_per_token": 20e-6,
        "cache_read_input_token_cost": 0.4e-6,
        "input_cost_per_token_above_272k_tokens": 8e-6,
        "output_cost_per_token_above_272k_tokens": 30e-6,
    }
}


def test_usage_prices_cached_input_without_double_counting_reasoning():
    response = {
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 800},
            "completion_tokens_details": {"reasoning_tokens": 90},
        }
    }
    assert response_cost(response, PRICING) == pytest.approx(0.00312)
    response["usage"]["prompt_tokens"] = 300000
    assert response_cost(response, PRICING) == pytest.approx(2.403)


def test_responses_usage_prices_cached_input_and_reasoning_once():
    response = {
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 800},
            "output_tokens_details": {"reasoning_tokens": 90},
        }
    }
    assert response_cost(response, PRICING) == pytest.approx(0.00312)
    del response["usage"]["output_tokens"]
    assert response_cost(response, PRICING) is None


def test_responses_tool_round_trip_preserves_reasoning_and_charges_before_next_call():
    import copy
    import json

    reasoning = {"type": "reasoning", "id": "reasoning-1", "summary": [], "encrypted_content": "opaque"}
    calls = []

    class ResponsesBinding:
        api = "responses"

        def complete(self, body):
            calls.append(copy.deepcopy(body))
            assert body["reasoning"] == {"effort": "xhigh"} and body["store"] is False
            assert body["max_output_tokens"] == 16384 and "max_completion_tokens" not in body
            assert all(tool["type"] == "function" and tool["strict"] is False for tool in body["tools"])
            if len(calls) == 1:
                output = [
                    reasoning,
                    {
                        "type": "function_call",
                        "name": "read_evaluation",
                        "call_id": "call-1",
                        "arguments": '{"record_id":"trial"}',
                    },
                ]
            else:
                assert history.cost_usd == pytest.approx(0.006)
                assert reasoning in body["input"]
                result = body["input"][-1]
                assert result["type"] == "function_call_output" and result["call_id"] == "call-1"
                assert json.loads(result["output"])["steps"][0]["message"] == "retained evidence"
                output = [{"type": "message", "content": [{"type": "output_text", "text": '{"proposal":"complete"}'}]}]
            return {"status": "completed", "output": output, "usage": {"input_tokens": 1000, "output_tokens": 100}}

    history = HistoryBinding(
        ResponsesBinding(),
        {"trial": {"trajectory": [{"message": "retained evidence"}]}},
        pricing=PRICING,
        remaining_cost_usd=1,
    )
    assert history.chat([]) == '{"proposal":"complete"}'
    assert len(calls) == 2 and history.cost_usd == pytest.approx(0.012) and not history.unknown_usage


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": True, "completion_tokens": 10},
        {"prompt_tokens": 10, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 11}},
    ],
)
def test_incomplete_usage_is_not_free(usage):
    assert response_cost({"usage": usage}, PRICING) is None


class Binding:
    api = "openai"

    def __init__(self, usage):
        self.usage = usage
        self.calls = 0

    def complete(self, body):
        self.calls += 1
        return {
            "usage": self.usage,
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call",
                                "function": {"name": "read_evaluation", "arguments": '{"record_id":"missing"}'},
                            }
                        ]
                    }
                }
            ],
        }


def test_observed_tool_turn_cost_stops_next_call():
    binding = Binding({"prompt_tokens": 1000, "completion_tokens": 100})
    history = HistoryBinding(binding, {}, pricing=PRICING, remaining_cost_usd=0.001)
    with pytest.raises(ProposerBudgetReached):
        history.chat([])
    assert binding.calls == 1
    assert history.cost_usd == pytest.approx(0.006)
    assert not history.unknown_usage


def test_missing_usage_stops_even_if_response_looks_successful():
    history = HistoryBinding(Binding(None), {}, pricing=PRICING, remaining_cost_usd=10)
    with pytest.raises(ProposerUsageUnknown):
        history.chat([])
    assert history.unknown_usage
    assert history.cost_usd == 0
    assert history.audit[0]["cost_usd"] is None


@pytest.mark.parametrize(
    "status,code,known", [(400, "unsupported_value", True), (500, "unsupported_value", False), (400, "other", False)]
)
def test_only_explicit_effort_validation_rejection_is_known_zero(status, code, known):
    import json

    from reef.harness.model_binding import ModelBindingError

    class RejectedBinding:
        api = "openai"

        def complete(self, body):
            raise ModelBindingError(
                "rejected",
                status=status,
                detail=json.dumps(
                    {"error": {"type": "invalid_request_error", "code": code, "param": "reasoning_effort"}}
                ),
            )

    history = HistoryBinding(RejectedBinding(), {}, pricing=PRICING, remaining_cost_usd=1)
    with pytest.raises(ModelBindingError):
        history.chat([])
    assert history.unknown_usage is not known
    assert history.audit[0]["rejected_before_generation"] is known
    assert history.audit[0]["cost_usd"] == (0.0 if known else None)


def test_tool_turn_deadline_stops_before_another_call_and_keeps_known_usage(monkeypatch):
    from recipes.meta_harness.examples.terminal_bench import history as module

    clock = iter([0, 0, 11])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))

    class TimedBinding(Binding):
        def complete(self, body, *, timeout_s):
            assert timeout_s == 10
            return super().complete(body)

    binding = TimedBinding({"prompt_tokens": 1000, "completion_tokens": 100})
    history = HistoryBinding(binding, {}, pricing=PRICING, remaining_cost_usd=1, timeout_s=10)
    with pytest.raises(TimeoutError, match="time allowance"):
        history.chat([])
    assert binding.calls == 1 and not history.unknown_usage
    assert history.cost_usd == pytest.approx(0.006)
