"""SDK call-order tests without credentials, model downloads, or GPU dependencies."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from reef.runtime.interfaces import UpstreamStatusError
from reef.tinker.checkpoint import TinkerCheckpoint
from reef.tinker.sampling import TinkerSDKSampler
from reef.train.tinker_backend.client import TinkerSDKClient
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.losses import ImportanceSamplingLoss, TinkerCustomLoss, TokenRow


class Future:
    def __init__(self, value):
        self.value = value

    def result(self, timeout=None):
        return self.value


class Trainer:
    def __init__(self, events, *, fail=False):
        self.events = events
        self.fail = fail

    def forward_backward(self, data, loss_fn, loss_fn_config=None):
        self.events.append(("forward_backward", data, loss_fn))
        return Future(SimpleNamespace(metrics={"loss:sum": -2.0}))

    def forward_backward_custom(self, data, loss_fn):
        value, metrics = loss_fn(data, [[-0.1] * len(datum.model_input) for datum in data])
        self.events.append(("forward_backward_custom", data, value))
        return Future(SimpleNamespace(metrics=metrics))

    def optim_step(self, adam_params):
        self.events.append(("optim_step", adam_params))
        if self.fail:
            raise TimeoutError("remote optimizer may have completed")
        return Future(None)

    def save_state(self, name, ttl_seconds):
        self.events.append(("save_state", name, ttl_seconds))
        return Future(SimpleNamespace(path="tinker://next/state"))

    def save_weights_for_sampler(self, name, ttl_seconds):
        self.events.append(("save_weights_for_sampler", name, ttl_seconds))
        return Future(SimpleNamespace(path="tinker://next/sampler"))


class Sampler:
    def __init__(self, events):
        self.events = events
        self.logprobs = [-0.25, -0.5]
        self.topk_logprobs: list | None = None
        self.refusal: Exception | None = None

    def get_base_model(self):
        return "Qwen/Qwen3-8B"

    def compute_logprobs(self, prompt):
        return Future([None, -0.1, -0.5, -0.7])

    def sample(self, *, prompt, num_samples, sampling_params, topk_sample_logprobs):
        self.events.append(("sample", prompt, num_samples, sampling_params, topk_sample_logprobs))
        if self.refusal is not None:
            raise self.refusal
        sequence = SimpleNamespace(
            tokens=[20, 21], logprobs=self.logprobs, topk_logprobs=self.topk_logprobs, stop_reason="length"
        )
        return Future(SimpleNamespace(sequences=[sequence]))


class Service:
    close_refusal: Exception | None = None

    def __init__(self, events, *, fail=False):
        self.events = events
        self.fail = fail
        self.sampler = Sampler(events)
        self.weights_info_paths: list[str] = []

    def create_rest_client(self):
        return self

    def get_weights_info_by_tinker_path(self, path):
        self.weights_info_paths.append(path)
        # Spindle refuses a sampler path at weights_info with a 400.
        if "/sampler_weights/" in path:
            raise BadRequestError(f"invalid checkpoint path: {path}")
        if path.endswith("/missing"):
            raise NotFoundError("checkpoint metadata not found")
        rank = 8 if path.endswith("/rank-8") else 32
        return Future(SimpleNamespace(base_model="Qwen/Qwen3-8B", is_lora=True, lora_rank=rank))

    def create_training_client_from_state_with_optimizer(self, path):
        self.events.append(("restore_with_optimizer", path))
        return Trainer(self.events, fail=self.fail)

    def create_lora_training_client(self, *, base_model, rank, seed):
        self.events.append(("initialize", base_model, rank, seed))
        return Trainer(self.events)

    def create_sampling_client(self, *, model_path=None, base_model=None):
        self.events.append(("sampler", model_path) if base_model is None else ("base_sampler", base_model))
        return self.sampler

    def close(self, status):
        self.events.append(("close", status))
        if self.close_refusal is not None:
            raise self.close_refusal
        return Future(None)


class ModelInput:
    @staticmethod
    def from_ints(tokens):
        return tuple(tokens)


class BadRequestError(Exception):
    pass


class NotFoundError(Exception):
    pass


class SDK:
    BadRequestError = BadRequestError
    NotFoundError = NotFoundError
    ModelInput = ModelInput
    Datum = SimpleNamespace
    AdamParams = SimpleNamespace
    SamplingParams = SimpleNamespace

    def __init__(self):
        self.events = []
        self.fail = False

    def ServiceClient(self, **kwargs):  # noqa: N802
        self.events.append(("session",))
        return Service(self.events, fail=self.fail)


@pytest.fixture
def client(tmp_path):
    # Bypass only SDK discovery, so these boundary tests also run on Python 3.10.
    value = object.__new__(TinkerSDKClient)
    value._sdk = SDK()
    value._model = "Qwen/Qwen3-8B"
    value._config = TinkerConfig(state_dir=str(tmp_path), train_timeout_s=10)
    value._lora_rank = 32
    value._seed = 0
    value._api_key = "test-only"
    value._service = Service(value._sdk.events)
    value.base_client = Sampler(value._sdk.events)
    return value


@pytest.fixture
def sampler(client):
    value = object.__new__(TinkerSDKSampler)
    value._sdk = client._sdk
    value._model = client._model
    value._timeout_s = 10
    value._service = client._service
    value.base_client = client.base_client
    value.tokenizer = None
    value.clients = {}
    return value


def test_training_restores_optimizer_then_saves_both_durable_snapshots(client):
    base = TinkerCheckpoint(client._model, 32, "tinker://base/state", "tinker://base/sampler")
    row = TokenRow((10, 11, 20, 21), (1, 1), (-0.25, -0.5), 2)
    checkpoint, _ = client.train(base, [[row], [row]], ImportanceSamplingLoss(), [1e-4, 5e-5], kl_coef=0.0)
    events = client._sdk.events
    assert [event[0] for event in events] == [
        "session",
        "restore_with_optimizer",
        "forward_backward",
        "optim_step",
        "forward_backward",
        "optim_step",
        "save_state",
        "save_weights_for_sampler",
        "close",
    ]
    assert events[1][1] == base.state_path
    datum = events[2][1][0]
    assert datum.model_input == (10, 11, 20)
    assert datum.loss_fn_inputs["target_tokens"] == [11, 20, 21]
    assert datum.loss_fn_inputs["advantages"] == [0, 2, 2]
    assert events[6][1] == events[7][1]
    assert events[6][2] is None and events[7][2] is None
    assert checkpoint.state_path == "tinker://next/state"
    # Every optimizer step takes its own rate: a schedule advances inside one job.
    assert [event[1].learning_rate for event in events if event[0] == "optim_step"] == [1e-4, 5e-5]
    assert events[-1] == ("close", "success")


def test_uncertain_optimizer_closes_attempt_and_retry_restores_incumbent(client):
    base = TinkerCheckpoint(client._model, 32, "tinker://base/state", "tinker://base/sampler")
    row = TokenRow((10, 20), (1,), (-0.25,), 1)
    client._sdk.fail = True
    with pytest.raises(TimeoutError):
        client.train(base, [[row]], ImportanceSamplingLoss(), [1e-4], kl_coef=0.0)
    assert client._sdk.events[-1] == ("close", "errored")
    assert not any(event[0] == "save_state" for event in client._sdk.events)
    client._sdk.fail = False
    client.train(base, [[row]], ImportanceSamplingLoss(), [1e-4], kl_coef=0.0)
    assert client._sdk.events[-1] == ("close", "success")
    assert [event[1] for event in client._sdk.events if event[0] == "restore_with_optimizer"] == [base.state_path] * 2


def test_custom_loss_routes_to_forward_backward_custom(client):
    class Custom(TinkerCustomLoss):
        def inputs(self, rows, base_logprobs, *, kl_coef):
            return [row.inputs([row.advantage] * len(row.mask)) for row in rows]

        def loss(self, data, logprobs):
            return sum(sum(values) for values in logprobs), {"custom:samples": float(len(data))}

    base = TinkerCheckpoint(client._model, 32, "tinker://base/state", "tinker://base/sampler")
    row = TokenRow((10, 11, 20, 21), (1, 1), (-0.25, -0.5), 2)
    _, metrics = client.train(base, [[row]], Custom(), [1e-4], kl_coef=0.0)
    names = [event[0] for event in client._sdk.events]
    assert "forward_backward_custom" in names and "forward_backward" not in names
    assert next(event for event in client._sdk.events if event[0] == "forward_backward_custom")[2] == pytest.approx(
        -0.3
    )
    assert metrics["custom:samples"] == 1.0


def test_initial_checkpoint_failure_closes_session_as_errored(client, monkeypatch):
    def fail_save(self, name, ttl_seconds):
        raise TimeoutError("checkpoint export timed out")

    monkeypatch.setattr(Trainer, "save_state", fail_save)
    with pytest.raises(TimeoutError, match="checkpoint export timed out"):
        client.initialize()
    assert client._sdk.events[-1] == ("close", "errored")


def test_initial_snapshot_and_sampling_use_explicit_immutable_paths(client, sampler):
    checkpoint = client.initialize()
    assert client._sdk.events[1] == ("initialize", client._model, 32, 0)
    assert client._sdk.events[-1] == ("close", "success")
    result = sampler.sample(checkpoint.sampler_path, [10, 11], {"max_tokens": 2}, topk_logprobs=0)
    assert result.tokens == (20, 21)
    assert result.logprobs == (-0.25, -0.5)
    assert client._sdk.events[-2] == ("sampler", checkpoint.sampler_path)
    sampler.sample(None, [10, 11], {"max_tokens": 2}, topk_logprobs=0)
    assert client._sdk.events[-1][0] == "sample"  # the base sampler needs no new client
    client._service.sampler.topk_logprobs = [[(20, -0.25), (3, -2.0)], [(21, -0.5), (4, -1.5)]]
    assert sampler.sample(checkpoint.sampler_path, [10, 11], {"max_tokens": 2}, topk_logprobs=2).topk == (
        ((20, -0.25), (3, -2.0)),
        ((21, -0.5), (4, -1.5)),
    )
    assert client._sdk.events[-1][-1] == 2
    client._service.sampler.topk_logprobs = [[(20, -0.25)], None]
    with pytest.raises(ValueError, match="no top-k log probabilities for every sampled token"):
        sampler.sample(checkpoint.sampler_path, [10, 11], {"max_tokens": 2}, topk_logprobs=2)
    client._service.sampler.logprobs = None
    with pytest.raises(ValueError, match="exact log probabilities"):
        sampler.sample(checkpoint.sampler_path, [10, 11], {"max_tokens": 2}, topk_logprobs=0)
    client.close()
    assert client._sdk.events[-1] == ("close", "success")


def test_a_null_seed_lets_the_service_choose(client):
    # Spindle's multi-LoRA trainer refuses a per-model seed.
    client._seed = None
    client.initialize()
    assert ("initialize", client._model, 32, None) in client._sdk.events


def test_a_service_without_the_session_finish_call_still_trains(client):
    # Spindle answers 404 to POST /api/v1/sessions/{id}/finish and ends the session when its heartbeats stop.
    Service.close_refusal = NotFoundError("Error code: 404 - Not Found")
    try:
        base = TinkerCheckpoint(client._model, 32, "tinker://b/state", "tinker://b/sampler")
        row = TokenRow((10, 11, 20, 21), (1, 1), (-0.25, -0.5), 2)
        checkpoint, _ = client.train(base, [[row]], ImportanceSamplingLoss(), [1e-4], kl_coef=0.0)
        client.close()
    finally:
        Service.close_refusal = None
    assert checkpoint.state_path == "tinker://next/state"
    assert [event for event in client._sdk.events if event[0] == "close"] == [("close", "success")] * 2


def test_base_sampling_clients_are_made_at_their_first_use(client, sampler):
    # On Spindle a sampling client starts a sampler pool, which a component that only trains never needs.
    client.base_client = None
    sampler.base_client = None
    row = TokenRow((10, 11, 20, 21), (1, 1), (-0.25, -0.5), 2)
    client.train(
        TinkerCheckpoint(client._model, 32, "tinker://b/state", "tinker://b/sampler"),
        [[row]],
        ImportanceSamplingLoss(),
        [1e-4],
        kl_coef=0.0,
    )
    assert not any(event[0] == "base_sampler" for event in client._sdk.events)

    sampler.sample(None, [10, 11], {"max_tokens": 2}, topk_logprobs=0)
    sampler.sample(None, [10, 11], {"max_tokens": 2}, topk_logprobs=0)
    assert [event for event in client._sdk.events if event[0] == "base_sampler"] == [("base_sampler", "Qwen/Qwen3-8B")]


def test_a_refused_sample_is_the_callers_400_and_an_overflow_names_the_context_length(sampler):
    # Tinker's message for a Qwen3.5-4B rollout whose history outgrew the 64K window.
    sampler.base_client.refusal = BadRequestError(
        "Error code: 400 - Prompt length plus max_tokens exceeds the model's context window: "
        "65614 prompt tokens + 4096 max_tokens > 65536"
    )
    with pytest.raises(UpstreamStatusError, match=r"maximum context length.*65614 prompt tokens") as overflow:
        sampler.sample(None, [10, 11], {"max_tokens": 2}, topk_logprobs=0)
    assert overflow.value.status == 400 and str(overflow.value).startswith("context_length_exceeded")
    sampler.base_client.refusal = BadRequestError("Error code: 400 - invalid sampling params")
    with pytest.raises(UpstreamStatusError, match=r"^Error code: 400 - invalid sampling params$"):
        sampler.sample(None, [10, 11], {"max_tokens": 2}, topk_logprobs=0)


def test_frozen_base_logprobs_align_to_response_and_reject_missing_values(client):
    rows = [TokenRow((10, 11, 20, 21), (1, 1), (-0.25, -0.5), 2)]
    assert client._base_logprobs(rows) == [[-0.5, -0.7]]
    with pytest.raises(ValueError, match="align"):
        client._base_logprobs([TokenRow((10, 20), (1,), (-0.1,), 1)])


def test_remote_model_mismatch_fails_before_training_or_sampling(client, sampler, monkeypatch):
    base = TinkerCheckpoint(client._model, 32, "tinker://base/state", "tinker://base/sampler")
    row = TokenRow((10, 20), (1,), (-0.25,), 1)
    monkeypatch.setattr(
        Service,
        "get_weights_info_by_tinker_path",
        lambda _, path: Future(SimpleNamespace(base_model="other/model", is_lora=True, lora_rank=32)),
    )
    with pytest.raises(ValueError, match="tinker://base/state is no rank 32 LoRA of Qwen/Qwen3-8B"):
        client.train(base, [[row]], ImportanceSamplingLoss(), [1e-4], kl_coef=0.0)
    assert not any(event[0] == "forward_backward" for event in client._sdk.events)
    monkeypatch.setattr(Sampler, "get_base_model", lambda _: "other/model")
    with pytest.raises(ValueError, match="remote Tinker sampler checkpoint"):
        sampler.sample(base.sampler_path, [10], {"max_tokens": 1}, topk_logprobs=0)
    assert not any(event[0] == "sample" for event in client._sdk.events)


def test_real_chat_template_returns_token_ids_without_model_download(sampler):
    client = sampler
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "assistant": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    client.tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        chat_template=(
            "{% for message in messages %}{{ message['content'] }} {% endfor %}"
            "{% if add_generation_prompt %}assistant{% endif %}"
        ),
    )
    assert client.render([{"role": "user", "content": "hello"}], template_kwargs={"enable_thinking": False}) == [1, 3]
    assert client.render(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}], template_kwargs={}
    ) == [1, 2]


def test_installed_sdk_call_signatures_and_datum_without_network():
    import inspect

    tinker = pytest.importorskip("tinker")
    from tinker.lib.public_interfaces.rest_client import RestClient

    calls = [
        (tinker.ServiceClient.close, ("success",), {}),
        (tinker.ServiceClient.close, ("errored",), {}),
        (tinker.ServiceClient.create_rest_client, (), {}),
        (tinker.ServiceClient.create_lora_training_client, (), {"base_model": "Qwen/Qwen3-8B", "rank": 32, "seed": 0}),
        (tinker.ServiceClient.create_training_client_from_state_with_optimizer, ("tinker://base/state",), {}),
        (tinker.ServiceClient.create_sampling_client, (), {"model_path": "tinker://base/sampler"}),
        (RestClient.get_weights_info_by_tinker_path, ("tinker://base/state",), {}),
        (RestClient.get_external_weights_urls, ("tinker://next/external_weights/checkpoint",), {}),
        (tinker.TrainingClient.forward_backward, ([],), {"loss_fn": "importance_sampling"}),
        (tinker.TrainingClient.forward_backward_custom, ([],), {"loss_fn": lambda data, logprobs: (None, {})}),
        (tinker.TrainingClient.optim_step, (tinker.AdamParams(learning_rate=1e-4),), {}),
        (tinker.TrainingClient.save_state, ("checkpoint",), {"ttl_seconds": None}),
        (tinker.TrainingClient.save_weights_for_sampler, ("checkpoint",), {"ttl_seconds": None}),
        (tinker.TrainingClient.save_weights_external, ("checkpoint",), {"ttl_seconds": None}),
        (tinker.SamplingClient.get_base_model, (), {}),
        (tinker.SamplingClient.get_tokenizer, (), {}),
        (tinker.SamplingClient.compute_logprobs, (tinker.ModelInput.from_ints([10, 11]),), {}),
        (
            tinker.SamplingClient.sample,
            (),
            {
                "prompt": tinker.ModelInput.from_ints([10]),
                "num_samples": 1,
                "sampling_params": tinker.SamplingParams(max_tokens=1),
            },
        ),
    ]
    for function, args, kwargs in calls:
        inspect.signature(function).bind(object(), *args, **kwargs)
    row = TokenRow((10, 11, 20, 21), (1, 0), (-0.2, -0.4), 2)
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(row.tokens[:-1])),
        loss_fn_inputs=ImportanceSamplingLoss().inputs([row], [], kl_coef=0)[0],
    )
    assert datum.model_input.length == 3
    assert datum.loss_fn_inputs["target_tokens"].data == [11, 20, 21]
    assert datum.loss_fn_inputs["advantages"].data == [0, 2, 0]


def test_an_imported_manifest_is_checked_by_its_state_path_alone(client):
    checkpoint = TinkerCheckpoint("Qwen/Qwen3-8B", 32, "tinker://run/weights/step-4", "tinker://run/sampler_weights/4")
    client.check_remote(checkpoint)
    # The sampler path is not looked up: Spindle refuses it at weights_info.
    assert client._service.weights_info_paths == ["tinker://run/weights/step-4"]
    for state_path, message in (
        ("tinker://run/weights/missing", "does not resolve"),
        ("tinker://run/sampler_weights/4", "refuses the Tinker path"),
        ("tinker://run/weights/rank-8", "no rank 32 LoRA"),
    ):
        with pytest.raises(ValueError, match=message):
            client.check_remote(replace(checkpoint, state_path=state_path))


def test_the_metrics_of_several_optimizer_steps_combine_by_their_suffixes():
    from reef.train.tinker_backend.client import combined_step_metrics

    combined = combined_step_metrics(
        [
            {"loss:sum": -2.0, "loss:mean": -0.2, "tokens:sum": 10.0, "packing_token_capacity:max": 4.0, "seed": 1.0},
            {"loss:sum": 0.0, "loss:mean": 0.0, "tokens:sum": 30.0, "packing_token_capacity:max": 8.0, "seed": 2.0},
        ]
    )

    assert combined == {
        "loss:sum": -2.0,
        "loss:mean": pytest.approx(-0.05),
        "tokens:sum": 40.0,
        "packing_token_capacity:max": 8.0,
        "seed": 2.0,
    }
    assert combined_step_metrics([{"loss:mean": 1.0}, {"loss:mean": 3.0}]) == {"loss:mean": 2.0}
