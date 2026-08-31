"""Validate every shipped Slime deployment YAML against the real parsers.

The SAO migration (commit 55e551d) removed ``--advantage-estimator=sao`` from
Slime's argparse choices while the SAO config still passed it — the deployment
could not boot, and nothing in CI noticed because no test ever
re-parsed a cookbook YAML's flags. These tests close that hole: for each
``training-*.yaml`` they materialize the slime-driver command exactly as
``reef serve`` would, strip the driver/retention/loss-family options exactly as
``reef_adapters.driver`` does, and then feed the remaining flags to the actual
Slime argparse surface plus the recipe's ``validate_backend_args`` and the
bridge preflight.

Level of validation achieved on CPU (no Megatron, no sglang installed):

* Every Slime-owned flag is parsed by the real parser — a typo, a removed
  flag, an invalid choice, or a bad value fails here.
* Driver-level flags (``--ready-file``, ``--reef-checkpoint-*``, ``--sao-*``)
  are parsed by the real driver/spec parsers.
* The recipe's backend contract (``validate_backend_args``) and the bridge
  preflight (``validate_bridge_args``) run on the parsed namespace, with the
  handful of ``slime_validate_args`` derivations they depend on simulated
  inline (each one cites the arguments.py logic it mirrors).
* Megatron-native flags cannot be parsed without a GPU image; they are checked
  **by name** against an explicit allowlist, so an unknown or misspelled flag
  still fails while values of allowlisted Megatron flags go unvalidated.
* ``--sglang-*`` flags are generated from sglang's ServerArgs at runtime and
  are stripped by prefix, unvalidated — the same convention slime itself uses
  to route them.

Slime's parser transitively imports torch (megatron_utils.lora), so this
  module skips where torch is unavailable; the main CPU CI gate installs CPU
  torch and runs it.

Discovery covers both the generic presets under ``configs`` and every
user-facing recipe under ``examples``.  A new example that launches the Reef
Slime driver is therefore part of this contract automatically.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "recipes"
SLIME_DRIVER_MODULE = "reef.train.slime_backend.reef_adapters.driver"


def _discover_training_configs() -> list[Path]:
    candidates = EXAMPLES_DIR.rglob("*.yaml")
    return sorted(path for path in candidates if SLIME_DRIVER_MODULE in path.read_text())


TRAINING_CONFIGS = _discover_training_configs()


def _discover_example_deployments() -> list[Path]:
    deployments: list[Path] = []
    for path in EXAMPLES_DIR.rglob("*.yaml"):
        text = path.read_text()
        if "\nservices:" in text and (text.startswith("reef:") or "\nreef:" in text):
            deployments.append(path)
    return sorted(deployments)


EXAMPLE_DEPLOYMENTS = _discover_example_deployments()


def _config_id(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _resolved_strings(config: dict, value):
    """Yield every string after applying the orchestrator's config pass."""
    from reef.service.deploy.config import interpolate_config

    if isinstance(value, dict):
        for item in value.values():
            yield from _resolved_strings(config, item)
    elif isinstance(value, list):
        for item in value:
            yield from _resolved_strings(config, item)
    elif isinstance(value, str):
        yield interpolate_config(config, value)


# Megatron-native flags used by the cookbook configs. Slime's parser only knows
# them when Megatron is importable (its extra-args provider extends Megatron's
# parser), so on CPU they surface as parse leftovers and are checked by name.
_MEGATRON_ONLY_FLAGS = frozenset(
    {
        "--accumulate-allreduce-grads-in-fp32",
        "--add-qkv-bias",
        "--adam-beta1",
        "--adam-beta2",
        "--adam-eps",
        "--attention-backend",
        "--attention-dropout",
        "--attention-softmax-in-fp32",
        "--ckpt-fully-parallel-save",
        "--context-parallel-size",
        "--disable-bias-linear",
        "--expert-model-parallel-size",
        "--expert-tensor-parallel-size",
        "--group-query-attention",
        "--hidden-dropout",
        "--lr-decay-style",
        "--normalization",
        "--optimizer",
        "--override-opt-param-scheduler",
        "--padded-vocab-size",
        "--pipeline-model-parallel-size",
        "--position-embedding-type",
        "--qk-layernorm",
        "--recompute-granularity",
        "--recompute-method",
        "--recompute-num-layers",
        "--rotary-base",
        "--seq-length",
        "--sequence-parallel",
        "--swiglu",
        "--tensor-model-parallel-size",
        "--weight-decay",
    }
)

# Representative values for the parameterized training deployments.  Each
# example's ``run.sh`` normally supplies these before invoking ``reef serve``;
# setting them here makes the generated command testable without a GPU stack.
_CONFIG_ENV = {
    "REEF_TOKEN": "config-test-token",
    "TTTD_CHECKPOINT_INTERVAL": "2",
    "TTTD_CUDA_GRAPH_MAX_BS": "8",
    "TTTD_CUDA_VISIBLE_DEVICES": "0,1",
    "TTTD_GLOBAL_BATCH_SIZE": "8",
    "TTTD_GROUPS_PER_STEP": "2",
    "TTTD_INFERENCE_HOST": "127.0.0.1",
    "TTTD_LOAD_PATH": "/tmp/reef-config-test/checkpoints/megatron",
    "TTTD_LORA_ALPHA": "32",
    "TTTD_LORA_RANK": "32",
    "TTTD_LOG_PROBS_CHUNK_SIZE": "512",
    "TTTD_MAX_RESPONSE_LENGTH": "4096",
    "TTTD_MAX_TOKENS_PER_GPU": "8192",
    "TTTD_MODEL_PATH": "/tmp/reef-config-test/model",
    "TTTD_NUM_GPUS": "2",
    "TTTD_NUM_STEPS": "1",
    "TTTD_PHASE1_MAX_TOKENS": "4096",
    "TTTD_RAY_NUM_CPUS": str(min(60, os.cpu_count() or 1)),
    "TTTD_ROLLOUTS_PER_GROUP": "4",
    "TTTD_SEQ_LENGTH": "8192",
    "TTTD_STATE_DIR": "/tmp/reef-config-test/state",
    "TTTD_STEPS": "1",
    "TTTD_TENSOR_PARALLEL_SIZE": "2",
}


def _install_slime_parser_stubs() -> None:
    """Make ``slime.utils.arguments`` importable without a GPU image.

    Only modules that are genuinely absent are stubbed, and only with the
    symbols the import chain touches at module scope; anything present stays
    real. The slime flag definitions themselves are untouched — that is the
    surface under test.
    """
    stubs: dict[str, dict[str, object]] = {
        "sglang": {},
        "sglang.srt": {},
        "sglang.srt.server_args": {"ServerArgs": type("ServerArgs", (), {})},
        "sglang_router": {},
        "sglang_router.launch_router": {"RouterArgs": type("RouterArgs", (), {})},
        "wandb": {},
    }
    for name, attrs in stubs.items():
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            module = types.ModuleType(name)
            for attr, value in attrs.items():
                setattr(module, attr, value)
            sys.modules[name] = module


def _build_slime_parser() -> argparse.ArgumentParser:
    _install_slime_parser_stubs()
    from slime.utils.arguments import get_slime_extra_args_provider

    from reef.train.slime_backend.reef_adapters.slime_arguments import add_reef_slime_arguments

    parser = argparse.ArgumentParser(allow_abbrev=False, exit_on_error=False)
    get_slime_extra_args_provider()(parser)
    add_reef_slime_arguments(parser)
    return parser


def _driver_tokens(config: dict) -> tuple[list[str], str, str]:
    """Materialize the driver command, recipe label, and explicit loss reference."""
    from reef.service.deploy.config import interpolate_config

    services = {service["name"]: service for service in config["services"]}
    command = interpolate_config(config, services["slime-driver"]["command"])
    tokens = shlex.split(command)
    module = SLIME_DRIVER_MODULE
    assert module in tokens, f"slime-driver service must invoke {module}"
    recipe = services["slime-driver"]["env"]["REEF_TRAINING_RECIPE"]
    recipe = interpolate_config(config, recipe)
    loss = services["slime-driver"]["env"]["REEF_TRAINING_LOSS"]
    loss = interpolate_config(config, loss)
    return tokens[tokens.index(module) + 1 :], recipe, loss


def _strip_sglang_flags(tokens: list[str]) -> list[str]:
    kept: list[str] = []
    skip_next = False
    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token.startswith("--sglang-"):
            if "=" not in token and index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                skip_next = True
            continue
        kept.append(token)
    return kept


def _parse_config(config_path: Path):
    """Run one cookbook config through the driver's own argument pipeline.

    Returns ``(args, spec, options, recipe)`` with ``args`` parsed by the real
    Slime parser and Megatron-only leftovers verified against the allowlist.
    """
    from reef.service.deploy.config import load_config
    from reef.train.slime_backend.loss_families import resolve_loss_family
    from reef.train.slime_backend.reef_adapters.driver import _driver_options, _retention_options

    with patch.dict(os.environ, _CONFIG_ENV, clear=False):
        config = load_config(config_path)
    tokens, recipe, loss = _driver_tokens(config)
    assert recipe == config["reef"]["recipe"], "slime-driver env must run the configured recipe"
    assert ":" in loss, "cookbook loss families must be explicit dotted references"

    _, tokens = _driver_options(tokens)
    _, tokens = _retention_options(tokens)
    spec = resolve_loss_family(loss)
    options, tokens = spec.parse_driver_options(tokens)
    tokens = _strip_sglang_flags(tokens)

    parser = _build_slime_parser()
    args, leftover = parser.parse_known_args(tokens)

    index = 0
    while index < len(leftover):
        token = leftover[index]
        flag = token.split("=", 1)[0]
        assert flag.startswith("--") and flag in _MEGATRON_ONLY_FLAGS, (
            f"{config_path.name} passes {token!r}, which neither the Slime parser nor the "
            "Megatron-only allowlist recognizes — a typo or a removed flag"
        )
        if "=" not in token and index + 1 < len(leftover) and not leftover[index + 1].startswith("-"):
            index += 1  # the allowlisted flag's value
        index += 1

    return args, spec, options, recipe


def _apply_validation_derivations(args) -> None:
    """Simulate the ``slime_validate_args`` derivations validation depends on.

    Each line mirrors a specific step in
    public ``slime.utils.arguments.slime_validate_args``; running the
    full function requires Megatron.
    """
    # Fallback recording + symmetric fallback for the upper clip bound.
    args.eps_clip_high_explicit = args.eps_clip_high is not None
    if args.eps_clip_high is None:
        args.eps_clip_high = args.eps_clip
    # ppo implies a critic.
    args.use_critic = args.use_critic or args.advantage_estimator == "ppo"
    # External-engine detection consulted by the bridge preflight.
    args.rollout_external = args.rollout_external_engine_addrs is not None
    # Colocation turns both offload phases on unless explicitly overridden.
    # ``slime_validate_args`` applies these defaults after argparse, before
    # Reef's bridge preflight sees the namespace.
    if args.colocate:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
    else:
        if args.offload_train is None:
            args.offload_train = False
        if args.offload_rollout is None:
            args.offload_rollout = False


@pytest.mark.unit
def test_cookbook_training_configs_are_discovered() -> None:
    paths = {_config_id(path) for path in TRAINING_CONFIGS}
    assert paths >= {
        "recipes/openclawrl/examples/openclawrl/serve.yaml",
        "recipes/sao/examples/sao/serve.yaml",
        "recipes/tttd/examples/tttd/serve.yaml",
        "recipes/tttd/examples/guidance_ttt/serve.yaml",
    }


@pytest.mark.unit
def test_user_facing_example_deployments_are_discovered() -> None:
    paths = {_config_id(path) for path in EXAMPLE_DEPLOYMENTS}
    assert paths == {
        "recipes/basic/external-provider.yaml",
        "recipes/basic/local-sglang.yaml",
        "recipes/openclawrl/examples/openclawrl/serve.yaml",
        "recipes/tttd/examples/guidance_ttt/serve.yaml",
        "recipes/harness_evolve/examples/harness_evolve/serve.yaml",
        "recipes/sao/examples/sao/serve.yaml",
        "recipes/tttd/examples/tttd/serve.yaml",
    }


@pytest.mark.unit
@pytest.mark.parametrize("config_path", EXAMPLE_DEPLOYMENTS, ids=_config_id)
def test_user_facing_example_deployment_resolves(config_path: Path) -> None:
    from reef.recipe import load_recipe_config
    from reef.recipe.registry import recipe_class_for
    from reef.service.assembly import _configured_inference_backend_factory, _recipe_owned_settings
    from reef.service.deploy.config import load_config
    from reef.service.deploy.settings import service_settings_from_config

    with patch.dict(os.environ, _CONFIG_ENV, clear=False):
        config = load_config(config_path)

    settings = service_settings_from_config(config)
    if settings.inference_backend_factory is not None:
        assert callable(_configured_inference_backend_factory(settings.inference_backend_factory))
    services = config.get("services")
    assert isinstance(services, list) and services
    names = [service.get("name") for service in services]
    assert all(isinstance(name, str) and name for name in names)
    assert len(names) == len(set(names))

    started: set[str] = set()
    for service in services:
        assert isinstance(service.get("command"), str) and service["command"].strip()
        dependencies = service.get("depends_on") or []
        assert set(dependencies) <= started, f"{service['name']} must follow its dependencies in service order"
        started.add(service["name"])
        for resolved in _resolved_strings(config, service):
            assert "${" not in resolved, f"{config_path} leaves an unresolved service value: {resolved!r}"

    recipe_type = recipe_class_for(settings.recipe)
    if recipe_type is None:
        inline_implementation = config.get("implementation")
        if isinstance(inline_implementation, str):
            recipe_type = recipe_class_for(inline_implementation)
        else:
            named_recipe = config_path.with_name(f"{settings.recipe}.yaml")
            assert named_recipe.is_file(), f"{config_path} selects unknown recipe {settings.recipe!r}"
            recipe_type = recipe_class_for(load_recipe_config(named_recipe)["implementation"])
    assert recipe_type is not None
    if settings.model_path is not None and hasattr(recipe_type, "service_config"):
        recipe_type.service_config(_recipe_owned_settings(settings), model_path=settings.model_path)


@pytest.mark.unit
def test_tttd_deployment_anchors_git_and_checkpoint_state_to_absolute_root() -> None:
    from reef.service.deploy.config import load_config
    from reef.service.deploy.settings import service_settings_from_config

    with patch.dict(os.environ, _CONFIG_ENV, clear=False):
        config = load_config(REPO_ROOT / "recipes" / "tttd" / "examples" / "tttd" / "serve.yaml")

    settings = service_settings_from_config(config)
    state_dir = Path(_CONFIG_ENV["TTTD_STATE_DIR"])
    assert Path(settings.artifact_repository) == state_dir / "artifacts.git"
    assert Path(settings.artifact_work_dir) == state_dir / "artifact-work"
    assert Path(settings.artifact_cache_dir) == state_dir / "artifact-cache"
    assert Path(settings.agent_record_dir) == state_dir / "agent-record"
    assert Path(config["training"]["checkpoint_dir"]) == state_dir / "checkpoints"


@pytest.mark.unit
@pytest.mark.parametrize("config_path", TRAINING_CONFIGS, ids=_config_id)
def test_cookbook_training_config_parses_and_validates(config_path: Path) -> None:
    from reef.train.slime_backend.reef_adapters.preflight import validate_bridge_args
    from reef.train.slime_backend.reef_adapters.slime_arguments import configure_reef_loss_args

    args, spec, options, recipe = _parse_config(config_path)
    _apply_validation_derivations(args)
    spec.apply_driver_options(args, options)
    # Match driver._serve: family projection can intentionally re-enable the
    # Slime pre-train pass after the YAML's initial --disable flag.
    configure_reef_loss_args(args)

    spec.validate_backend_args(args, recipe=recipe)
    validate_bridge_args(args, spec)


@pytest.mark.unit
def test_sao_config_uses_the_hook_based_contract() -> None:
    config_path = EXAMPLES_DIR / "sao" / "examples" / "sao" / "serve.yaml"
    args, spec, options, _ = _parse_config(config_path)
    _apply_validation_derivations(args)
    spec.apply_driver_options(args, options)

    assert args.use_critic is True
    assert args.loss_family == "sao"
    assert not hasattr(args, "custom_advantage_function_path") or args.custom_advantage_function_path is None
    assert args.eps_clip_high is not None


# The parser marks a handful of flags required; standalone parses (outside a
# full cookbook command) must supply them.
_REQUIRED_BASELINE = ["--rollout-batch-size=1"]


@pytest.mark.unit
def test_removed_advantage_estimator_choice_fails_the_parse() -> None:
    # Proof the suite catches the class of regression that shipped: the value
    # 'sao' was removed from --advantage-estimator's choices, so parsing it
    # must fail loudly rather than fall through as an unknown leftover.
    parser = _build_slime_parser()

    with pytest.raises((argparse.ArgumentError, SystemExit)):
        parser.parse_known_args([*_REQUIRED_BASELINE, "--advantage-estimator=sao"])


@pytest.mark.unit
def test_unknown_flag_fails_the_allowlist() -> None:
    parser = _build_slime_parser()
    _, leftover = parser.parse_known_args([*_REQUIRED_BASELINE, "--seq-lenght=4096"])  # typo, on purpose

    assert leftover, "a misspelled flag must not be silently parsed"
    assert leftover[0].split("=", 1)[0] not in _MEGATRON_ONLY_FLAGS


@pytest.mark.unit
def test_critic_steps_per_actor_defaults_to_the_paper_k() -> None:
    # The flag has no default of its own: unset means "the family's default",
    # and for SAO that is the paper's K=2 (SaoSchedule), so an unset flag
    # never silently halves the critic cadence.
    from reef.train.slime_backend.loss_families import resolve_loss_family

    parser = _build_slime_parser()
    args, _ = parser.parse_known_args(_REQUIRED_BASELINE)

    assert args.critic_steps_per_actor is None
    bound = resolve_loss_family("sao").bind(critic_steps_per_actor=args.critic_steps_per_actor)
    assert bound._schedule.critic_steps_per_actor == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
