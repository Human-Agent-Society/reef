"""SDK call-order tests without credentials, model downloads, or GPU dependencies."""

from types import SimpleNamespace

import pytest

from reef.train.tinker_backend.checkpoint import TinkerCheckpoint
from reef.train.tinker_backend.client import TinkerSDKClient
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.losses import ImportanceSamplingLoss, TokenRow


class Future:
    def __init__(self, value):
        self.value = value

    def result(self, timeout=None):
        return self.value


class Trainer:
    def __init__(self, events, *, fail=False):
        self.events = events
        self.fail = fail

    def forward_backward(self, data, loss_fn):
        self.events.append(("forward_backward", data, loss_fn))
        return Future(SimpleNamespace(metrics={"loss:sum": -2.0}))

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

    def get_base_model(self):
        return "Qwen/Qwen3-8B"

    def compute_logprobs(self, prompt):
        return Future([None, -0.1, -0.5, -0.7])

    def sample(self, *, prompt, num_samples, sampling_params):
        self.events.append(("sample", prompt, num_samples, sampling_params))
        return Future(
            SimpleNamespace(sequences=[SimpleNamespace(tokens=[20, 21], logprobs=self.logprobs, stop_reason="length")])
        )


class Service:
    def __init__(self, events, *, fail=False):
        self.events = events
        self.fail = fail
        self.sampler = Sampler(events)

    def create_rest_client(self):
        return self

    def get_weights_info_by_tinker_path(self, path):
        return Future(SimpleNamespace(base_model="Qwen/Qwen3-8B", is_lora=True, lora_rank=32))

    def create_training_client_from_state_with_optimizer(self, path):
        self.events.append(("restore_with_optimizer", path))
        return Trainer(self.events, fail=self.fail)

    def create_lora_training_client(self, *, base_model, rank, seed):
        self.events.append(("initialize", base_model, rank, seed))
        return Trainer(self.events)

    def create_sampling_client(self, *, model_path):
        self.events.append(("sampler", model_path))
        return self.sampler

    def close(self, status):
        self.events.append(("close", status))
        return Future(None)


class ModelInput:
    @staticmethod
    def from_ints(tokens):
        return tuple(tokens)


class SDK:
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
    value._api_key = "test-only"
    value._service = Service(value._sdk.events)
    value._base_sampler = Sampler(value._sdk.events)
    return value


def test_training_restores_optimizer_then_saves_both_durable_snapshots(client):
    base = TinkerCheckpoint(client._model, 32, "tinker://base/state", "tinker://base/sampler")
    row = TokenRow((10, 11, 20, 21), (1, 1), (-0.25, -0.5), 2)
    checkpoint, metrics = client.train(base, [[row], [row]], ImportanceSamplingLoss())
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
    assert metrics["optimizer_steps"] == 2
    assert events[-1] == ("close", "success")


def test_uncertain_optimizer_closes_attempt_and_retry_restores_incumbent(client):
    base = TinkerCheckpoint(client._model, 32, "tinker://base/state", "tinker://base/sampler")
    row = TokenRow((10, 20), (1,), (-0.25,), 1)
    client._sdk.fail = True
    with pytest.raises(TimeoutError):
        client.train(base, [[row]], ImportanceSamplingLoss())
    assert client._sdk.events[-1] == ("close", "errored")
    assert not any(event[0] == "save_state" for event in client._sdk.events)
    client._sdk.fail = False
    client.train(base, [[row]], ImportanceSamplingLoss())
    assert client._sdk.events[-1] == ("close", "success")
    assert [event[1] for event in client._sdk.events if event[0] == "restore_with_optimizer"] == [base.state_path] * 2


def test_initial_checkpoint_failure_closes_session_as_errored(client, monkeypatch):
    def fail_save(self, name, ttl_seconds):
        raise TimeoutError("checkpoint export timed out")

    monkeypatch.setattr(Trainer, "save_state", fail_save)
    with pytest.raises(TimeoutError, match="checkpoint export timed out"):
        client.initialize()
    assert client._sdk.events[-1] == ("close", "errored")


def test_initial_snapshot_and_sampling_use_explicit_immutable_paths(client):
    checkpoint = client.initialize()
    assert client._sdk.events[1] == ("initialize", client._model, 32, 0)
    assert client._sdk.events[-1] == ("close", "success")
    result = client.sample(checkpoint, [10, 11], {"max_tokens": 2})
    assert result.tokens == (20, 21)
    assert result.logprobs == (-0.25, -0.5)
    assert client._sdk.events[-2] == ("sampler", checkpoint.sampler_path)
    client._service.sampler.logprobs = None
    with pytest.raises(ValueError, match="exact log probabilities"):
        client.sample(checkpoint, [10, 11], {"max_tokens": 2})
    client.close()
    assert client._sdk.events[-1] == ("close", "success")


def test_frozen_base_logprobs_align_to_response_and_reject_missing_values(client):
    rows = [TokenRow((10, 11, 20, 21), (1, 1), (-0.25, -0.5), 2)]
    assert client._base_logprobs(rows) == [[-0.5, -0.7]]
    with pytest.raises(ValueError, match="align"):
        client._base_logprobs([TokenRow((10, 20), (1,), (-0.1,), 1)])


def test_remote_model_mismatch_fails_before_training_or_sampling(client, monkeypatch):
    base = TinkerCheckpoint(client._model, 32, "tinker://base/state", "tinker://base/sampler")
    row = TokenRow((10, 20), (1,), (-0.25,), 1)
    monkeypatch.setattr(
        Service,
        "get_weights_info_by_tinker_path",
        lambda _, path: Future(SimpleNamespace(base_model="other/model", is_lora=True, lora_rank=32)),
    )
    with pytest.raises(ValueError, match="remote Tinker training checkpoint"):
        client.train(base, [[row]], ImportanceSamplingLoss())
    assert not any(event[0] == "forward_backward" for event in client._sdk.events)
    monkeypatch.setattr(Sampler, "get_base_model", lambda _: "other/model")
    with pytest.raises(ValueError, match="remote Tinker sampler checkpoint"):
        client.sample(base, [10], {"max_tokens": 1})
    assert not any(event[0] == "sample" for event in client._sdk.events)


def test_real_chat_template_returns_token_ids_without_model_download(client):
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "assistant": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    client._tokenizer = transformers.PreTrainedTokenizerFast(
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
        (tinker.TrainingClient.forward_backward, ([],), {"loss_fn": "importance_sampling"}),
        (tinker.TrainingClient.optim_step, (tinker.AdamParams(learning_rate=1e-4),), {}),
        (tinker.TrainingClient.save_state, ("checkpoint",), {"ttl_seconds": None}),
        (tinker.TrainingClient.save_weights_for_sampler, ("checkpoint",), {"ttl_seconds": None}),
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
