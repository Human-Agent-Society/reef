"""Lightweight tests for the Slime algorithm registry and bridge policies."""

from __future__ import annotations

import ast
import importlib
import os
import re
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from recipes.sao.slime import CRITIC_ATTENTION_PARAM_PATTERN
from recipes.sao.slime.utils import schedule as sao_runtime
from reef.train.algos.registry import register_loss_family_ref
from reef.train.slime_backend.algorithm import SlimeAlgorithm, resolve_objective_paths
from reef.train.slime_backend.loss_families import (
    LOSS_FAMILIES,
    UnknownLossFamilyError,
    register_loss_family,
    resolve_loss_family,
    unregister_loss_family,
)


class _Group:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def async_train(self, rollout_id, rollout_data, external_data=None):
        self.calls.append((rollout_id, rollout_data, external_data))
        return self.result


class _Teacher:
    def score(self, tokens, response_lengths):
        del tokens, response_lengths
        return [[-0.5, -0.6]]


def _resolve(value):
    return value


class _TestAlgorithm(SlimeAlgorithm):
    def __init__(self, loss_family="toy", loss_type="sft_loss"):
        self.loss_family = loss_family
        self.loss_type = loss_type

    def validate_specific_args(self, args, source):
        pass


#: The cookbook families plus the two plain ones tests/conftest.py registers.
_ALL_FAMILIES = ("openclawrl", "pg", "sao", "sft", "tttd")


@pytest.mark.unit
def test_registry_is_the_complete_slime_loss_family_surface() -> None:
    assert LOSS_FAMILIES.names == _ALL_FAMILIES
    assert resolve_loss_family("sft").loss_type == "sft_loss"
    assert resolve_loss_family("tttd").loss_type == "custom_loss"
    assert resolve_loss_family("openclawrl").loss_type == "custom_loss"

    with pytest.raises(UnknownLossFamilyError, match="available families"):
        resolve_loss_family("missing")
    # The unknown-family error teaches the two extension paths.
    with pytest.raises(UnknownLossFamilyError, match=r"register_loss_family\(\)"):
        resolve_loss_family("missing")
    with pytest.raises(UnknownLossFamilyError, match=r"package\.module:SPEC"):
        resolve_loss_family("missing")


@pytest.mark.unit
def test_register_loss_family_opens_the_registry() -> None:
    toy = _TestAlgorithm(loss_family="toy")
    try:
        assert register_loss_family(toy) is toy
        assert resolve_loss_family("toy") is toy
        assert "toy" in LOSS_FAMILIES.names
        # Re-registering the same spec is idempotent; a different spec under
        # the same name is a conflict that names the taken families.
        assert register_loss_family(toy) is toy
        with pytest.raises(ValueError, match="'toy' is already registered"):
            register_loss_family(_TestAlgorithm(loss_family="toy", loss_type="policy_loss"))
    finally:
        unregister_loss_family("toy")
    assert LOSS_FAMILIES.names == _ALL_FAMILIES
    with pytest.raises(UnknownLossFamilyError):
        resolve_loss_family("toy")


@pytest.mark.unit
def test_registered_family_references_can_be_unregistered() -> None:
    register_loss_family_ref("temporary", "some_package.loss:TemporaryAlgorithm")
    assert "temporary" in LOSS_FAMILIES.names
    unregister_loss_family("temporary")
    assert "temporary" not in LOSS_FAMILIES.names
    with pytest.raises(KeyError, match="registered families"):
        unregister_loss_family("never_registered")


@pytest.mark.unit
def test_dotted_loss_family_reference_resolves_and_registers_the_spec(monkeypatch) -> None:
    module = ModuleType("toy_loss_pkg")
    module.ALGORITHM = _TestAlgorithm(loss_family="toy_dotted")  # type: ignore[attr-defined]
    module.NOT_A_SPEC = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "toy_loss_pkg", module)
    try:
        spec = resolve_loss_family("toy_loss_pkg:ALGORITHM")
        assert spec is module.ALGORITHM
        # Import-registration: the wire payload names the canonical family.
        assert resolve_loss_family("toy_dotted") is spec
    finally:
        unregister_loss_family("toy_dotted")

    with pytest.raises(UnknownLossFamilyError, match="cannot import loss family"):
        resolve_loss_family("no.such.module:ALGORITHM")
    with pytest.raises(UnknownLossFamilyError, match="not a SlimeAlgorithm"):
        resolve_loss_family("toy_loss_pkg:NOT_A_SPEC")
    with pytest.raises(UnknownLossFamilyError, match=r"must be 'package\.module:SPEC'"):
        resolve_loss_family(":ALGORITHM")


@pytest.mark.unit
def test_dotted_reference_cannot_shadow_a_registered_family(monkeypatch) -> None:
    module = ModuleType("toy_loss_shadow")
    module.ALGORITHM = _TestAlgorithm(loss_family="pg", loss_type="policy_loss")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "toy_loss_shadow", module)

    with pytest.raises(UnknownLossFamilyError, match="already registered to a different spec"):
        resolve_loss_family("toy_loss_shadow:ALGORITHM")
    assert LOSS_FAMILIES.names == _ALL_FAMILIES


@pytest.mark.unit
def test_sft_spec_refuses_advantages_instead_of_silently_dropping_them() -> None:
    from reef.train.slime_backend.data_builder import to_slime_rollout_data

    sample = ["a", [10, 11, 12], [1, 1], [], 0.5]
    payload = {"samples": [sample], "rollout_ids": [0], "loss": "sft", "advantages": [0.5]}

    # A reward-weighted-SFT author must hear that sft trains unweighted and
    # where the weighted objective lives, not train silently unweighted.
    with pytest.raises(ValueError, match="sft ignores advantages"):
        to_slime_rollout_data(payload)
    with pytest.raises(ValueError, match="pg loss family"):
        to_slime_rollout_data(payload)

    del payload["advantages"]
    assert to_slime_rollout_data(payload)["loss"] == "sft"


@pytest.mark.unit
def test_common_spec_validation_owns_loss_and_logprob_requirements() -> None:
    pg = resolve_loss_family("pg")
    pg.validate_backend_args(SimpleNamespace(loss_type="policy_loss", use_rollout_logprobs=True))

    with pytest.raises(RuntimeError, match="loss-type policy_loss"):
        pg.validate_backend_args(SimpleNamespace(loss_type="sft_loss", use_rollout_logprobs=True))
    with pytest.raises(RuntimeError, match="use-rollout-logprobs"):
        pg.validate_backend_args(SimpleNamespace(loss_type="policy_loss", use_rollout_logprobs=False))


@pytest.mark.unit
def test_openclawrl_validation_refuses_a_diverged_old_actor() -> None:
    # The OPD branch's old-policy log-probs come from the current actor
    # forward, so any configuration where old and current diverge is refused
    # at validation time, before a worker boots.
    topk = resolve_loss_family("openclawrl")
    accepted = {"loss_type": "custom_loss", "num_steps_per_rollout": 1, "keep_old_actor": False}
    topk.validate_backend_args(SimpleNamespace(**accepted))

    with pytest.raises(RuntimeError, match="num-steps-per-rollout"):
        topk.validate_backend_args(SimpleNamespace(**{**accepted, "num_steps_per_rollout": 2}))
    with pytest.raises(RuntimeError, match="keep-old-actor"):
        topk.validate_backend_args(SimpleNamespace(**{**accepted, "keep_old_actor": True}))


@pytest.mark.unit
def test_openclawrl_knobs_travel_from_argv_onto_args() -> None:
    # The objective reads its knobs off args by name, so the parse -> stamp
    # path is what keeps a Megatron actor from silently running a default.
    from recipes.openclawrl.slime import OpenclawrlSettings

    topk = resolve_loss_family("openclawrl")
    settings, remaining = topk.parse_driver_options(
        [
            "--openclawrl-w-rl=0.5",
            "--openclawrl-w-opd=2.0",
            "--openclawrl-adv-diff-clip=0.25",
            "--openclawrl-hint-selection=shortest",
            "--openclawrl-subset-mode=teacher",
            "--loss-type",
            "custom_loss",
        ]
    )

    assert settings == OpenclawrlSettings(
        w_rl=0.5,
        w_opd=2.0,
        adv_diff_clip=0.25,
        hint_selection="shortest",
        subset_mode="teacher",
    )
    assert remaining == ["--loss-type", "custom_loss"]

    args = SimpleNamespace()
    topk.apply_driver_options(args, settings)
    assert args.loss_family == "openclawrl"
    assert (args.openclawrl_w_rl, args.openclawrl_w_opd, args.openclawrl_adv_diff_clip) == (0.5, 2.0, 0.25)
    assert (args.openclawrl_hint_selection, args.openclawrl_subset_mode) == ("shortest", "teacher")

    # The deprecated --openclaw-topk-* spellings still parse; an argv naming
    # none of them stamps the paper defaults rather than leaving args bare.
    legacy, _ = topk.parse_driver_options(["--openclaw-topk-subset-mode=overlap"])
    assert legacy.subset_mode == "overlap"
    defaults = SimpleNamespace()
    topk.apply_driver_options(defaults, None)
    assert defaults.openclawrl_w_rl == 1.0
    assert defaults.openclawrl_hint_selection == "sequence_optimal"

    # The driver forwards the parsed Settings into bind on every real boot.
    assert topk.bind(settings) is topk
    with pytest.raises(TypeError, match="OpenclawrlSettings"):
        topk.bind(SimpleNamespace(teacher_url="http://teacher"))


@pytest.mark.unit
@pytest.mark.parametrize("name", ["w_rl", "w_opd", "adv_diff_clip"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "1.0", None])
def test_openclawrl_rejects_non_finite_numeric_settings(name, value) -> None:
    from recipes.openclawrl.slime import OpenclawrlSettings

    with pytest.raises(ValueError, match=name):
        OpenclawrlSettings(**{name: value})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("hint_selection", "greedy"),
        ("hint_selection", ""),
        ("subset_mode", "hybrid"),
        ("subset_mode", ""),
    ],
)
def test_openclawrl_rejects_unknown_enum_settings(name, value) -> None:
    from recipes.openclawrl.slime import OpenclawrlSettings

    with pytest.raises(ValueError, match=name):
        OpenclawrlSettings(**{name: value})


@pytest.mark.unit
def test_openclawrl_settings_validate_on_the_cli_path_too() -> None:
    # argparse converts "nan" to a float without complaint, so the dataclass
    # is the layer that has to refuse it on both construction paths.
    topk = resolve_loss_family("openclawrl")
    with pytest.raises(ValueError, match="w_rl"):
        topk.parse_driver_options(["--openclawrl-w-rl=nan"])
    with pytest.raises(ValueError, match="adv_diff_clip"):
        topk.parse_driver_options(["--openclawrl-adv-diff-clip=inf"])


@pytest.mark.unit
def test_openclawrl_keeps_valid_configurations() -> None:
    from recipes.openclawrl.slime import OpenclawrlSettings

    assert OpenclawrlSettings() == OpenclawrlSettings(
        w_rl=1.0,
        w_opd=1.0,
        adv_diff_clip=1.0,
        hint_selection="sequence_optimal",
        subset_mode="student",
    )
    disabled = OpenclawrlSettings(w_rl=0.0, w_opd=-0.5, adv_diff_clip=-1.0)
    assert disabled.adv_diff_clip == -1.0


@pytest.mark.unit
def test_bound_algo_rejects_payload_for_a_different_loss_family() -> None:
    algo = resolve_loss_family("pg").bind()

    with pytest.raises(ValueError, match="received payload loss"):
        algo.validate_payload({"loss": "sft"})


@pytest.mark.unit
def test_sao_algo_owns_critic_actor_schedule() -> None:
    algo = resolve_loss_family("sao").bind(critic_steps_per_actor=2, critic_only_steps=0)
    actor = _Group([{"actor": True}])
    critic = _Group([{"values": [0.25]}])

    result = algo.train(
        3,
        "packed",
        actor_group=actor,
        critic_group=critic,
        resolve=_resolve,
    )

    assert len(critic.calls) == 2
    assert actor.calls == [(3, "packed", [{"values": [0.25]}])]
    assert result.worker_results == [{"actor": True}]
    assert result.durable_metrics == {"sao/critic_updates": 2, "sao/actor_trained": 1}


@pytest.mark.unit
def test_sao_schedule_supports_warmup_and_validates_configuration() -> None:
    schedule = sao_runtime.SaoSchedule(critic_steps_per_actor=2, critic_only_steps=2)

    assert schedule.plan(0).train_actor is False
    assert schedule.plan(1).train_actor is False
    assert schedule.plan(2).train_actor is True
    with pytest.raises(ValueError, match="critic_steps_per_actor"):
        sao_runtime.SaoSchedule(critic_steps_per_actor=0)
    with pytest.raises(ValueError, match="critic_only_steps"):
        sao_runtime.SaoSchedule(critic_only_steps=-1)
    with pytest.raises(ValueError, match="rollout_index"):
        schedule.plan(-1)


@pytest.mark.unit
def test_sao_provenance_metrics_handles_policy_versions_and_clock_skew() -> None:
    sao = resolve_loss_family("sao")
    rollout_data = {
        "producing_runtime_load_ids": ["inc:3"],
        "rollout_created_ats": [time.time() - 2.5],
        "response_lengths": [4],
        "loss_masks": [[1, 1, 1, 0]],
    }
    metrics = sao.provenance_metrics(rollout_data, serving_version="inc:7")
    assert metrics["sao/policy_lag_max"] == 4
    assert metrics["sao/policy_lag_mean"] == 4.0
    assert metrics["sao/effective_token_rate"] == 3 / 4
    assert metrics["sao/queue_age_s_max"] >= 2.0


@pytest.mark.unit
def test_sao_config_freezes_critic_attention_only_for_moe() -> None:
    sao = resolve_loss_family("sao")

    moe_critic = SimpleNamespace(num_experts=8)
    sao.configure_critic_args(moe_critic)
    patterns = moe_critic.freeze_params_name_list
    assert patterns == [CRITIC_ATTENTION_PARAM_PATTERN]
    assert any(re.search(pattern, "decoder.layers.0.self_attention.linear_qkv.weight") for pattern in patterns)
    assert not any(re.search(pattern, "decoder.layers.0.mlp.router.weight") for pattern in patterns)

    dense_critic = SimpleNamespace(num_experts=0)
    sao.configure_critic_args(dense_critic)
    assert not getattr(dense_critic, "freeze_params_name_list", None)


# ---------------------------------------------------------------------------
# Family conformance: the layout and naming conventions documented in
# reef/train/slime_backend/algorithm.py.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ADAPTERS_ROOT = _REPO_ROOT / "reef" / "train" / "slime_backend" / "reef_adapters"
_BUNDLED_FAMILIES = ("openclawrl", "sao", "tttd")
#: Deprecated flag spellings allowed alongside the canonical --<pkg>-* form.
_DEPRECATED_FLAG_PREFIXES = ("--openclaw-topk-",)
#: @objective channel -> required entry-point name for package <pkg>.
_OBJECTIVE_ENTRY_NAMES = {
    "custom_loss_function_path": "{pkg}_loss",
    "custom_pg_loss_function_path": "{pkg}_loss",
    "custom_advantage_function_path": "{pkg}_advantages",
    "reef_actor_init_hook_path": "{pkg}_actor_init",
    "reef_actor_pre_train_hook_path": "{pkg}_actor_pre_train",
}
_EXPECTED_OBJECTIVE_CHANNELS = {
    "openclawrl": frozenset(
        {
            "custom_advantage_function_path",
            "custom_loss_function_path",
            "reef_actor_init_hook_path",
            "reef_actor_pre_train_hook_path",
        }
    ),
    "sao": frozenset({"custom_advantage_function_path", "custom_pg_loss_function_path"}),
    "tttd": frozenset({"custom_advantage_function_path", "custom_loss_function_path"}),
}


def _family_package_name(loss_family: str) -> str:
    """The name a family's classes, flags and settings are spelled after.

    Every cookbook family is spelled after its canonical loss-family name and
    lives in its own method package.
    """
    return resolve_loss_family(loss_family).loss_family


def _family_package_dir(loss_family: str) -> Path:
    """The source directory holding a family's spec module and its ``objective.py``.

    Derived from the module name, not ``__file__``, so the contracts read the
    repository sources even when the suite runs against an installed wheel.
    """
    module_name = type(resolve_loss_family(loss_family)).__module__
    return _REPO_ROOT.joinpath(*module_name.split("."))


def _family_source_files() -> list[Path]:
    """Every cookbook family's source files."""
    files: set[Path] = set()
    for family in _BUNDLED_FAMILIES:
        files.update(_family_package_dir(family).rglob("*.py"))
    return sorted(files)


@pytest.mark.unit
def test_family_packages_never_import_the_adapter_layer() -> None:
    # The dependency points one way: reef_adapters consumes the registry and
    # the SlimeAlgorithm interface; no family package may reach back into the
    # adapter layer.
    # Every import spelling counts: ``from pkg import reef_adapters`` names the
    # module in the alias, and a relative import is resolved against the
    # package so ``from ....reef_adapters import x`` cannot slip through.
    offenders: list[str] = []
    for path in _family_source_files():
        package = ".".join(path.parent.relative_to(_REPO_ROOT).parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split(".")
                    base = ".".join(parts[: len(parts) - node.level + 1])
                    module = f"{base}.{node.module}" if node.module else base
                else:
                    module = node.module or ""
                targets = [f"{module}.{alias.name}" for alias in node.names]
            offenders.extend(
                f"{path.relative_to(_REPO_ROOT)}: {target}" for target in targets if "reef_adapters" in target
            )
    assert offenders == []


@pytest.mark.unit
def test_adapter_layer_never_compares_against_a_family_name() -> None:
    # The adapter layer stays family-blind: per-family behavior rides the
    # spec's declared attributes, never a family name in code. Scanning every
    # string constant rather than just ``==``/``!=`` operands is what makes
    # this hold for the natural multi-family spellings too — ``in {...}``,
    # dispatch dicts, ``match``/``case``, ``startswith`` — and it costs
    # nothing: the tree has no such literal today. Docstrings and comments are
    # free to name families (a docstring is an Expr, not a bare Constant).
    named = {*_BUNDLED_FAMILIES, *(_family_package_name(family) for family in _BUNDLED_FAMILIES)}
    offenders: list[str] = []
    for path in sorted(_ADAPTERS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or id(node) in docstrings:
                continue
            if isinstance(node.value, str) and node.value in named:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno} names family {node.value!r}")
    assert offenders == []


@pytest.mark.unit
def test_family_flags_carry_the_package_prefix() -> None:
    # Every option string a family registers starts with --<pkg>-; the
    # whitelisted deprecated aliases are tolerated next to a canonical flag.
    offenders: list[str] = []
    for family in _BUNDLED_FAMILIES:
        pkg = _family_package_name(family)
        canonical = f"--{pkg}-"
        for path in sorted(_family_package_dir(family).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr != "add_argument":
                    continue
                for argument in node.args:
                    if not (isinstance(argument, ast.Constant) and isinstance(argument.value, str)):
                        continue
                    flag = argument.value
                    if not flag.startswith("--"):
                        continue
                    if flag.startswith(canonical) or flag.startswith(_DEPRECATED_FLAG_PREFIXES):
                        continue
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno} {flag}")
    assert offenders == []


@pytest.mark.unit
def test_spec_and_settings_class_names_follow_the_package() -> None:
    for family in _BUNDLED_FAMILIES:
        spec = resolve_loss_family(family)
        pkg = _family_package_name(family)
        assert type(spec).__name__ == f"{pkg.capitalize()}Algorithm", family
        package = importlib.import_module(type(spec).__module__)
        settings_names = [name for name in vars(package) if name.endswith("Settings")]
        assert settings_names in ([], [f"{pkg.capitalize()}Settings"]), family


@pytest.mark.unit
@pytest.mark.parametrize("family", _BUNDLED_FAMILIES)
def test_objective_entry_points_are_named_by_channel(family: str) -> None:
    pkg = _family_package_name(family)
    objective_path = _family_package_dir(family) / "objective.py"
    if not objective_path.exists():
        assert not _EXPECTED_OBJECTIVE_CHANNELS[family], family
        return

    tree = ast.parse(objective_path.read_text(encoding="utf-8"))
    registered: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "objective"
                and len(decorator.args) == 1
                and isinstance(decorator.args[0], ast.Constant)
                and isinstance(decorator.args[0].value, str)
            ):
                continue
            channel = decorator.args[0].value
            assert channel not in registered, (family, channel)
            registered[channel] = node.name

    assert registered.keys() == _EXPECTED_OBJECTIVE_CHANNELS[family], family

    spec = resolve_loss_family(family)
    assert frozenset(spec.required_objective_hooks) == _EXPECTED_OBJECTIVE_CHANNELS[family], family
    if spec.loss_type == "custom_loss":
        assert "custom_loss_function_path" in registered, family
    if spec.uses_pg_loss_primitive:
        assert "custom_pg_loss_function_path" in registered, family

    for channel, fn_name in registered.items():
        assert fn_name == _OBJECTIVE_ENTRY_NAMES[channel].format(pkg=pkg), (channel, fn_name)


@pytest.mark.unit
def test_objective_resolution_only_ignores_the_missing_objective_module(monkeypatch) -> None:
    # The plain sft family from tests/conftest.py declares no hooks.
    module_path = f"{type(resolve_loss_family('sft')).__module__}.objective"

    def missing_objective(name: str) -> None:
        assert name == module_path
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(importlib, "import_module", missing_objective)
    args = SimpleNamespace(loss_family="sft")
    resolve_objective_paths(args)
    assert vars(args) == {"loss_family": "sft"}


@pytest.mark.unit
def test_objective_resolution_rejects_a_missing_required_objective_module(monkeypatch) -> None:
    module_path = "recipes.tttd.slime.objective"

    def missing_objective(name: str) -> None:
        assert name == module_path
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(importlib, "import_module", missing_objective)
    with pytest.raises(RuntimeError, match="requires objective hooks"):
        resolve_objective_paths(SimpleNamespace(loss_family="tttd"))


@pytest.mark.unit
def test_objective_resolution_rejects_missing_required_hook_registrations(monkeypatch) -> None:
    from reef.train.slime_backend import algorithm as base_module

    monkeypatch.setattr(importlib, "import_module", lambda _name: ModuleType("empty_objective"))
    monkeypatch.setattr(base_module, "_objective_registry", {})
    with pytest.raises(RuntimeError, match="custom_loss_function_path"):
        resolve_objective_paths(SimpleNamespace(loss_family="tttd"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("loss_type", "uses_pg_loss_primitive", "required_channel"),
    [
        ("custom_loss", False, "custom_loss_function_path"),
        ("policy_loss", True, "custom_pg_loss_function_path"),
    ],
)
def test_objective_resolution_derives_core_hook_requirements(
    monkeypatch, loss_type: str, uses_pg_loss_primitive: bool, required_channel: str
) -> None:
    from reef.train.slime_backend import algorithm as base_module

    toy = _TestAlgorithm(loss_family="toy_objective", loss_type=loss_type)
    toy.uses_pg_loss_primitive = uses_pg_loss_primitive
    try:
        register_loss_family(toy)
        monkeypatch.setattr(importlib, "import_module", lambda _name: ModuleType("empty_objective"))
        monkeypatch.setattr(base_module, "_objective_registry", {})
        with pytest.raises(RuntimeError, match=required_channel):
            resolve_objective_paths(SimpleNamespace(loss_family="toy_objective"))
    finally:
        unregister_loss_family("toy_objective")


@pytest.mark.unit
@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named 'worker_dependency'", name="worker_dependency"),
        ImportError("objective module raised an import error"),
    ],
)
def test_objective_resolution_propagates_nested_import_failures(monkeypatch, error: ImportError) -> None:
    def broken_objective(_name: str) -> None:
        raise error

    monkeypatch.setattr(importlib, "import_module", broken_objective)
    with pytest.raises(type(error), match=str(error)):
        resolve_objective_paths(SimpleNamespace(loss_family="tttd"))


@pytest.mark.unit
def test_family_utils_modules_stay_driver_side() -> None:
    # utils/ is the driver-side half of a family package: importable without
    # the GPU stack, so no torch/megatron/slime imports at any scope.
    offenders: list[str] = []
    for path in [p for p in _family_source_files() if p.parent.name == "utils"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                targets = [node.module or ""]
            offenders.extend(
                f"{path.relative_to(_REPO_ROOT)}: {target}"
                for target in targets
                if target.split(".", 1)[0] in {"torch", "megatron", "slime"}
            )
    assert offenders == []


@pytest.mark.unit
@pytest.mark.parametrize("family", _BUNDLED_FAMILIES)
def test_response_aligned_keys_declare_their_dtype(family: str) -> None:
    # The worker slices these per response token with the declared dtype, so
    # a key without one would fail at the first batch instead of at boot.
    spec = resolve_loss_family(family)
    missing = set(spec.response_aligned_keys) - set(spec.rollout_tensor_dtypes)
    assert not missing, f"{family} declares {sorted(missing)} response-aligned without a dtype"


@pytest.mark.unit
def test_dotted_reference_accepts_the_decorated_class(monkeypatch) -> None:
    # The documented external pattern: decorate the class, name it as
    # package.module:ClassName. Resolving returns the instance the decorator
    # registered, so the canonical name resolves to the same object.
    from reef.train.slime_backend.algorithm import register_loss_family as decorate

    module = ModuleType("toy_loss_decorated")

    @decorate
    class ToyAlgorithm(SlimeAlgorithm):
        loss_family = "toy_decorated"
        loss_type = "sft_loss"

        def validate_specific_args(self, args, source):
            return None

    module.ToyAlgorithm = ToyAlgorithm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "toy_loss_decorated", module)
    try:
        spec = resolve_loss_family("toy_loss_decorated:ToyAlgorithm")
        assert isinstance(spec, ToyAlgorithm)
        assert resolve_loss_family("toy_decorated") is spec
        assert "toy_decorated" in LOSS_FAMILIES.names
    finally:
        unregister_loss_family("toy_decorated")
    assert "toy_decorated" not in LOSS_FAMILIES.names


@pytest.mark.unit
def test_dotted_family_reference_survives_a_fresh_worker_process(tmp_path) -> None:
    # The driver's registry does not travel with args: a worker process that
    # only sees the canonical name of an external family cannot resolve it.
    # loss_family_ref carries the import path across.
    import subprocess

    package = tmp_path / "toy_worker_family"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from reef.train.slime_backend.algorithm import SlimeAlgorithm, register_loss_family\n"
        "\n"
        "@register_loss_family\n"
        "class ToyWorkerAlgorithm(SlimeAlgorithm):\n"
        "    loss_family = 'toy_worker'\n"
        "    loss_type = 'sft_loss'\n"
        "\n"
        "    def validate_specific_args(self, args, source):\n"
        "        return None\n",
        encoding="utf-8",
    )
    script = (
        "from argparse import Namespace\n"
        "from reef.train.slime_backend.algorithm import resolve_args_loss_family, resolve_objective_paths\n"
        "from reef.train.slime_backend.loss_families import UnknownLossFamilyError\n"
        "try:\n"
        "    resolve_args_loss_family(Namespace(loss_family='toy_worker'))\n"
        "    print('canonical:resolved')\n"
        "except UnknownLossFamilyError:\n"
        "    print('canonical:unknown')\n"
        "args = Namespace(loss_family='toy_worker', loss_family_ref='toy_worker_family:ToyWorkerAlgorithm')\n"
        "print('ref:' + resolve_args_loss_family(args).loss_family)\n"
        "resolve_objective_paths(args)\n"
        "print('objective:ok')\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(tmp_path), os.environ.get("PYTHONPATH", "")])),
    }
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, env=env)
    assert result.stdout.split() == ["canonical:unknown", "ref:toy_worker", "objective:ok"]
