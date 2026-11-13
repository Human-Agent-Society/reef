"""Slime's facts: what the backend states from its options without Slime's parser, and that the component agrees."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from tests.slime_backend.test_slime_component import worker_args
from tests.slime_backend.test_slime_launch import MODEL, SAO, deployment_config, settings_of

from reef.artifact.artifact import Artifact
from reef.artifact.base_root import BASE_FILE, BaseRoot
from reef.core.errors import RuntimeConfigError
from reef.core.snapshot import RootSource
from reef.core.weight_layout import ArtifactFormat
from reef.train.component import ScenarioRoot, training_backend_facts
from reef.train.slime_backend import component as slime_component
from reef.train.slime_backend.facts import SlimeFacts
from reef.train.slime_backend.launch import SlimeDeployment, component_factory

FULL: dict = {}
LORA = {"megatron-lora-rank": 8, "megatron-lora-target-modules": ["linear_qkv", "linear_proj"]}
#: What Slime's parser makes of the LoRA options above: the alpha defaults to the rank.
LORA_PARSED = {
    "megatron_lora_rank": 8,
    "megatron_lora_alpha": 8,
    "megatron_lora_target_modules": ["linear_qkv", "linear_proj"],
}


def component_config(tmp_path: Path, options: dict) -> dict:
    """The component config of a Slime deployment whose training options add ``options``."""
    config = deployment_config(tmp_path, training={"options": {"global-batch-size": 4, "lr": 1e-6, **options}})
    return SlimeDeployment().component_config(settings_of(config), default_parameters=SAO)


@pytest.fixture
def parsed(tmp_path, monkeypatch):
    """Slime's parse without Slime: ``parsed(values)`` sets the arguments that the component gets."""
    values: dict = {}
    monkeypatch.setattr(
        slime_component,
        "prepare_slime_arguments",
        lambda arguments, **options: (worker_args(tmp_path, **values), None, None),
    )
    monkeypatch.setattr(slime_component, "prepare_checkpoint_storage", lambda args, retention: None)

    def use(**parsed_values: object) -> None:
        values.update({"megatron_lora_rank": 0, "hf_checkpoint": MODEL, **parsed_values})

    return use


def roots(tmp_path: Path, facts: SlimeFacts) -> list[ScenarioRoot]:
    """A root of the facts' default start, one of other inputs, and one of an imported item."""
    artifact = Artifact.local(tmp_path)
    return [
        ScenarioRoot(artifact, RootSource("default", inputs=facts.default_start_inputs)),
        ScenarioRoot(artifact, RootSource("default", inputs={"model": "/models/other"})),
        ScenarioRoot(artifact, RootSource("item", item="upload-1", uri="file:///upload/", version="")),
    ]


def refusal(check, root: ScenarioRoot) -> str | None:
    """The message with which ``check`` refuses ``root``, or None when it accepts it."""
    try:
        check(root)
    except ValueError as exc:
        return str(exc)
    return None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("options", "parsed_values", "artifact_format"),
    [
        (FULL, {}, ArtifactFormat("full")),
        (LORA, LORA_PARSED, ArtifactFormat("adapter", 8, ("q_proj", "k_proj", "v_proj", "o_proj"))),
    ],
)
def test_the_facts_of_a_config_answer_as_the_component_of_the_same_config(
    tmp_path, parsed, options, parsed_values, artifact_format
) -> None:
    config = component_config(tmp_path, options)
    facts = training_backend_facts(config, model_path=MODEL, environ={})
    assert isinstance(facts, SlimeFacts)
    assert facts.artifact_format == artifact_format
    parsed(**parsed_values)
    component = component_factory(config, MODEL, {})
    try:
        assert component.facts == facts
        assert (facts.name, facts.artifact_format, facts.shares_scenarios, facts.concurrent_jobs) == (
            component.name,
            component.artifact_format,
            component.shares_scenarios,
            component.concurrent_jobs,
        )
        default = facts.default_start()
        assert dict(default.files) == {BASE_FILE: BaseRoot(facts.default_start_inputs).to_text()}
        assert default.inputs == facts.default_start_inputs
        for kind in ("trajectory", "task"):
            assert facts.parameters_schema(kind) == component.parameters_schema(kind)
        answers = [
            (refusal(facts.check_root_start, root), refusal(component.root_start, root))
            for root in roots(tmp_path, facts)
        ]
        assert [mine is None for mine, _ in answers] == [True, False, False]
        assert all(mine == theirs for mine, theirs in answers)
    finally:
        component.close()


@pytest.mark.unit
def test_the_component_refuses_options_whose_written_facts_differ_from_the_parsed_arguments(tmp_path, parsed) -> None:
    """A service that reads the facts without the component would describe other snapshots than the workers save."""
    parsed(**LORA_PARSED)
    with pytest.raises(RuntimeError, match=r"as written state .*lora_rank=0.* but Slime parsed .*lora_rank=8"):
        component_factory(component_config(tmp_path, FULL), MODEL, {})


@pytest.mark.unit
def test_the_facts_read_each_option_as_the_deployment_writes_it() -> None:
    assert SlimeFacts.from_arguments(["--hf-checkpoint=/m", "--lr=1e-06"]) == SlimeFacts("/m")
    # A list option is a flag with its values after it; the last flag of a name counts, and the alpha defaults.
    arguments = [
        "--hf-checkpoint",
        "/m",
        "--megatron-lora-rank=4",
        "--megatron-lora-target-modules",
        "a",
        "b",
        "--lr=1",
    ]
    assert SlimeFacts.from_arguments([*arguments, "--megatron-lora-rank=16"]) == SlimeFacts("/m", 16, 16, ("a", "b"))
    assert SlimeFacts.from_arguments([*arguments, "--megatron-lora-alpha=32"]) == SlimeFacts("/m", 4, 32, ("a", "b"))
    with pytest.raises(ValueError, match="name no --hf-checkpoint"):
        SlimeFacts.from_arguments(["--lr=1e-06"])
    with pytest.raises(ValueError, match="are integers"):
        SlimeFacts.from_arguments(["--hf-checkpoint=/m", "--megatron-lora-rank=eight"])
    with pytest.raises(ValueError, match="need a positive lora_rank"):
        SlimeFacts.from_arguments(["--hf-checkpoint=/m", "--megatron-lora-alpha=8"])
    with pytest.raises(ValueError, match="at least 0"):
        SlimeFacts("/m", -1)


@pytest.mark.unit
def test_options_without_a_model_state_no_facts() -> None:
    config = {"type": SlimeDeployment.component_type, "settings": {"training_backend_options": {"lr": 1e-6}}}
    with pytest.raises(RuntimeConfigError, match="name no --hf-checkpoint"):
        training_backend_facts(config, model_path=MODEL, environ={})


def test_the_facts_need_no_training_framework() -> None:
    code = (
        "import sys; "
        "sys.modules.update(dict.fromkeys(('ray', 'torch', 'slime', 'sglang', 'megatron', 'tinker'))); "
        "from reef.core.weight_layout import ArtifactFormat; "
        "from reef.train.component import training_backend_facts; "
        "options = {'hf-checkpoint': '${reef.model_path}', 'megatron-lora-rank': 8}; "
        "settings = {'model_path': '/models/demo', 'training_backend_options': options}; "
        "config = {'type': 'reef.train.slime_backend.launch:component_factory', 'settings': settings}; "
        "facts = training_backend_facts(config, model_path='/models/demo', environ={}); "
        "assert facts.artifact_format.adapter_rank == 8, facts; "
        "assert facts.default_start_inputs['model'] == '/models/demo', facts; "
        "assert facts.parameters_schema('trajectory')"
    )
    subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)
