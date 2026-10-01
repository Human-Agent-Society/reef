"""Independent teacher request, alignment, failure and launch contracts."""

import copy
import io
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from reef.core.errors import DeployConfigError
from reef.service.deploy.service_config import ServiceConfig
from reef.train.slime_backend.distill.algorithm import DistillAlgorithm, DistillSettings, settings_from_args
from reef.train.slime_backend.distill.engine import EngineTeacher, parse_teacher_scores
from reef.train.slime_backend.launch import driver_arguments
from reef.train.slime_backend.teacher_deployment import teacher_service


def settings(**kwargs):
    return DistillSettings(
        **{
            "teacher": "separate",
            "teacher_url": "http://teacher:30001",
            "teacher_model_path": "/models/teacher",
            "top_k": 2,
            **kwargs,
        }
    )


def scores():
    return {
        "meta_info": {
            "completion_tokens": 0,
            "input_token_logprobs": [[None, 2, None], [-0.3, 3, None], [-0.5, 4, None]],
            "input_top_logprobs": [None, [[-0.3, 3, None], [-2, 5, None]], [[-0.5, 4, None], [-3, 6, None]]],
        }
    }


def test_request_scores_exact_ids_without_generation():
    response = io.BytesIO(json.dumps(scores()).encode())
    with patch("urllib.request.OpenerDirector.open", return_value=response) as opened:
        result = EngineTeacher(settings()).score([1, 2, 3, 4], 2)
    body = json.loads(opened.call_args.args[0].data)
    assert body["input_ids"] == [1, 2, 3, 4]
    assert body["sampling_params"]["max_new_tokens"] == 0
    assert body["logprob_start_len"] == 1
    assert result["distill_teacher_sampled_log_probs"] == [-0.3, -0.5]
    assert result["distill_teacher_topk_ids"] == [[3, 5], [4, 6]]


@pytest.mark.parametrize("change", ["missing", "shift", "nan", "topk", "duplicate", "generated", "leading"])
def test_malformed_scores_fail_instead_of_training_on_misaligned_rows(change):
    result = scores()
    meta = result["meta_info"]
    if change == "missing":
        meta["input_token_logprobs"].pop()
    elif change == "shift":
        meta["input_token_logprobs"][1][1] = 4
    elif change == "nan":
        meta["input_token_logprobs"][1][0] = float("nan")
    elif change == "topk":
        meta["input_top_logprobs"][1].pop()
    elif change == "duplicate":
        meta["input_top_logprobs"][1][1][1] = 3
    elif change == "generated":
        meta["completion_tokens"] = 1
    else:
        meta["input_token_logprobs"][0][0] = -1
    with pytest.raises(ValueError):
        parse_teacher_scores(result, [1, 2, 3, 4], 2, 2)


def test_partial_teacher_failure_leaves_batch_unchanged():
    batch = {"teacher_tokens": [[1, 2, 3, 4], [1, 2, 3, 4]], "response_lengths": [2, 2]}
    original = copy.deepcopy(batch)
    good = parse_teacher_scores(scores(), [1, 2, 3, 4], 2, 2)
    with (
        patch.object(EngineTeacher, "validate_model"),
        patch.object(EngineTeacher, "score", side_effect=[good, TimeoutError("deadline")]),
        pytest.raises(TimeoutError),
    ):
        EngineTeacher(settings()).prepare(batch)
    assert batch == original


@pytest.mark.parametrize(
    "overrides",
    [
        {"teacher": "self"},
        {"teacher_checkpoint": "/checkpoint"},
        {"teacher_model_path": ""},
        {"top_k": 0},
        {"top_k_source": "student"},
        {"teacher_timeout": 0},
        {"teacher_url": "file:///tmp/teacher"},
        {"teacher_url": "http://user:secret@teacher"},
    ],
)
def test_unsupported_teacher_modes_fail_early(overrides):
    with pytest.raises(ValueError):
        settings(**overrides)


def test_family_flags_binding_and_worker_settings_preserve_engine():
    family = DistillAlgorithm()
    family.loss_family = "toy"
    parsed, remainder = family.parse_driver_options(
        [
            "--toy-teacher",
            "separate",
            "--toy-teacher-url",
            "http://teacher:30001",
            "--toy-teacher-model-path",
            "/models/teacher",
            "--toy-top-k",
            "2",
            "--lr",
            "0.01",
        ]
    )
    assert parsed == settings()
    assert remainder == ["--lr", "0.01"]
    args = SimpleNamespace()
    family.apply_driver_options(args, parsed)
    assert settings_from_args(args) == settings()
    bound = family.bind(parsed)
    assert bound is not family
    batch = {"teacher_tokens": [[1, 2, 3, 4]], "response_lengths": [2]}
    with (
        patch.object(EngineTeacher, "validate_model"),
        patch.object(EngineTeacher, "score", return_value=parse_teacher_scores(scores(), [1, 2, 3, 4], 2, 2)),
    ):
        assert bound.prepare_rollout(batch)["perf/distill_teacher_time"] >= 0
    assert batch["distill_teacher_sampled_log_probs"] == [[-0.3, -0.5]]
    assert family.prepare_rollout({}) == {}


def test_managed_teacher_has_separate_gpu_allocation_and_scoring_endpoint():
    config = asdict(ServiceConfig(recipe="recipe", teacher_model_path="/models/teacher", teacher_num_gpus=2))
    service = teacher_service(config)
    assert service["executor"] == "ray"
    assert service["resources"]["num_gpus"] == 2
    assert "CUDA_VISIBLE_DEVICES" not in service.get("env", {})
    assert service["command"][-2:] == ["--tp", "2"]
    arguments = driver_arguments(
        {"reef": config, "endpoints": {"distill-teacher": "http://worker:30001"}}, loss_family="opd"
    )
    assert "--opd-teacher-url=http://worker:30001" in arguments
    assert "--opd-teacher-model-path=/models/teacher" in arguments


@pytest.mark.parametrize("flag", ["model-path", "tp", "base-gpu-id", "tokenizer-path", "enable-lora", "enable-mis"])
def test_managed_teacher_rejects_identity_and_placement_overrides(flag):
    with pytest.raises(DeployConfigError):
        teacher_service(
            asdict(ServiceConfig(recipe="recipe", teacher_model_path="/teacher", teacher_options={flag: "value"}))
        )


@pytest.mark.parametrize("actual", ["/wrong/model", None])
def test_endpoint_identity_must_match_validated_model(actual):
    response = io.BytesIO(json.dumps({"model_path": actual}).encode())
    with (
        patch("urllib.request.OpenerDirector.open", return_value=response),
        pytest.raises(ValueError, match="model_path"),
    ):
        EngineTeacher(settings()).validate_model()


def test_teacher_public_config_resolves_to_owned_dependency(tmp_path):
    from reef.service.deploy.orchestrator import resolve_deployment_config

    config, _ = resolve_deployment_config(
        {
            "schema-version": 2,
            "inference": {"model-path": "/student", "num-gpus": 1},
            "teacher": {"model-path": "/teacher", "num-gpus": 2, "port": 30123},
            "recipe": {"implementation": "recipes.opd.recipe:OPDRecipe", "config": {"tokenizer-path": "/student"}},
            "training": {"backend": "slime", "options": {"opd-teacher": "separate", "opd-top-k": 1}},
        },
        None,
        tmp_path / "serve.yaml",
    )
    services = {service["name"]: service for service in config["services"]}
    assert services["slime-driver"]["depends_on"] == ["distill-teacher"]
    assert services["distill-teacher"]["resources"]["num_gpus"] == 2
    assert services["distill-teacher"]["endpoint"] == "http://{host}:30123"
    assert config["reef"]["teacher_model_path"] == "/teacher"


def test_unsupported_training_backend_rejects_teacher():
    from reef.service.deploy.training import assemble_training_services

    with pytest.raises(DeployConfigError, match="independent teacher"):
        assemble_training_services(
            {
                "reef": {
                    "recipe": "recipe",
                    "training_backend": "tinker",
                    "model_path": "student",
                    "teacher_model_path": "teacher",
                }
            }
        )


def test_provider_recipe_rejects_teacher():
    from reef.service.deploy.inference import assemble_provider_services

    with pytest.raises(DeployConfigError, match="weight-training recipe"):
        assemble_provider_services({"reef": {"recipe": "recipe", "teacher_model_path": "teacher"}})


def test_engine_columns_match_native_actor_columns_on_identical_logits(monkeypatch):
    import sys

    import torch

    from reef.train.slime_backend.distill.teacher import gather_teacher_topk

    mpu = SimpleNamespace(get_tensor_model_parallel_group=lambda: None, get_context_parallel_world_size=lambda: 1)
    monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))
    monkeypatch.setitem(sys.modules, "megatron", SimpleNamespace(core=SimpleNamespace(mpu=mpu)))
    tokens = [1, 2, 3, 4]
    logits = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 0.0, 3.0, 5.0, 2.0, 4.0],
                [2.0, 1.0, 0.0, 3.0, 5.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            ]
        ]
    )
    probabilities = logits[0].log_softmax(-1)
    # SGLang emits one unscored prefix token followed by next-token scores.
    meta = {"completion_tokens": 0, "input_token_logprobs": [[None, 2, None]], "input_top_logprobs": [None]}
    for position, token in ((1, 3), (2, 4)):
        values, ids = probabilities[position].topk(2)
        meta["input_token_logprobs"].append([probabilities[position, token].item(), token, None])
        meta["input_top_logprobs"].append([[v, i, None] for v, i in zip(values.tolist(), ids.tolist(), strict=True)])
    parsed = parse_teacher_scores({"meta_info": meta}, tokens, 2, 2)
    _, native = gather_teacher_topk(
        logits,
        args=SimpleNamespace(rollout_temperature=1.0, distill_top_k=2),
        unconcat_tokens=[torch.tensor(tokens)],
        total_lengths=[4],
        response_lengths=[2],
    )
    for key in parsed:
        torch.testing.assert_close(torch.tensor(parsed[key]), native[key][0])


def test_tokenizer_validation_rejects_equal_size_but_different_mapping(monkeypatch):
    import sys

    from reef.train.slime_backend.distill.engine import validate_teacher_tokenizer

    tokenizers = [
        SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1}),
        SimpleNamespace(get_vocab=lambda: {"a": 1, "b": 0}),
    ]
    factory = SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizers.pop(0))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=factory))
    with pytest.raises(ValueError, match="token-to-ID"):
        validate_teacher_tokenizer("student", "teacher")


def test_engine_temperature_is_rejected_before_loading_models():
    family = DistillAlgorithm()
    family.loss_family = "toy"
    args = SimpleNamespace(num_steps_per_rollout=1, rollout_temperature=0.8)
    family.apply_driver_options(args, settings())
    with pytest.raises(ValueError, match="untempered"):
        family.validate_specific_args(args, "test")


@pytest.mark.parametrize("port", [True, "30001", 0, 65536, 8000])
def test_teacher_rejects_invalid_or_conflicting_port(port):
    config = asdict(ServiceConfig(recipe="recipe", port=8000, teacher_model_path="/teacher"))
    config["teacher_port"] = port
    with pytest.raises(DeployConfigError, match=r"teacher\.port"):
        teacher_service(config)
