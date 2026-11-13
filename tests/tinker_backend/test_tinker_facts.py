"""The facts of the Tinker and the Spindle full-parameter backends: what each states from its settings without a
client, and that its component agrees."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests.support.hugging_face_hub import COMMIT
from tests.tinker_backend.test_spindle_full import BASE
from tests.tinker_backend.test_spindle_full import MODEL as SPINDLE_MODEL
from tests.tinker_backend.test_spindle_full import SPINDLE, Exporter, SpindleClient
from tests.tinker_backend.test_spindle_full import settings as spindle_settings
from tests.tinker_backend.test_tinker_backend import MODEL, RemoteClient

from reef.artifact import REFERENCE_FILE, Artifact, HuggingFaceRevision
from reef.artifact.base_root import BASE_FILE, BaseRoot
from reef.core.snapshot import RootSource
from reef.core.weight_layout import ArtifactFormat
from reef.train.component import ScenarioRoot, TrainingBackendFacts, TrainingComponent, training_backend_facts
from reef.train.tinker_backend.checkpoint import FreshStart, TinkerCheckpoint
from reef.train.tinker_backend.component import TinkerTrainingComponent
from reef.train.tinker_backend.config import TinkerSettings
from reef.train.tinker_backend.facts import SpindleFullFacts, TinkerFacts
from reef.train.tinker_backend.full import SpindleFullTrainingComponent
from reef.train.tinker_backend.launch import SpindleFullDeployment, TinkerDeployment

ITEM = RootSource("item", item="upload-1", uri="file:///upload/", version="")


def manifest_root(directory: Path, checkpoint: TinkerCheckpoint) -> ScenarioRoot:
    """A root whose files are the manifest of ``checkpoint``, in ``directory``."""
    checkpoint.write(directory)
    return ScenarioRoot(Artifact.local(directory), ITEM)


def reference_root(directory: Path, revision: HuggingFaceRevision | None) -> ScenarioRoot:
    """A root whose files refer to ``revision``, or hold nothing of a start for None, in ``directory``."""
    directory.mkdir()
    if revision is None:
        (directory / "notes.txt").write_text("no start\n")
    else:
        (directory / REFERENCE_FILE).write_text(json.dumps(revision.to_dict()) + "\n")
    return ScenarioRoot(Artifact.local(directory), ITEM)


def default_root(directory: Path, facts: TrainingBackendFacts) -> ScenarioRoot:
    """The root of a new scenario: the files that ``facts.default_start()`` writes, in ``directory``."""
    default = facts.default_start()
    assert default.inputs == facts.default_start_inputs
    directory.mkdir()
    default.write(directory)
    return ScenarioRoot(Artifact.local(directory), RootSource("default", inputs=default.inputs))


def refusal(check, root: ScenarioRoot) -> str | None:
    """The message with which ``check`` refuses ``root``, or None when it accepts it."""
    try:
        check(root)
    except ValueError as exc:
        return str(exc)
    return None


def assert_same_answers(
    facts: TrainingBackendFacts, component: TrainingComponent, roots: list[ScenarioRoot], accepted: list[bool]
) -> None:
    """Every member of ``facts`` answers as ``component``; of ``roots``, both accept exactly the ``accepted`` ones."""
    assert (facts.name, facts.artifact_format, facts.shares_scenarios, facts.concurrent_jobs) == (
        component.name,
        component.artifact_format,
        component.shares_scenarios,
        component.concurrent_jobs,
    )
    for kind in ("trajectory", "task"):
        assert facts.parameters_schema(kind) == component.parameters_schema(kind)
    answers = [(refusal(facts.check_root_start, root), refusal(component.root_start, root)) for root in roots]
    assert [mine is None for mine, _ in answers] == accepted
    assert all(mine == theirs for mine, theirs in answers)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("options", "service"), [({"lora_rank": 8}, None), ({"lora_rank": 32, "base_url": SPINDLE}, SPINDLE)]
)
def test_the_tinker_facts_of_a_config_answer_as_the_component_of_the_same_config(tmp_path, options, service) -> None:
    config = {"type": TinkerDeployment.component_type, "state_dir": str(tmp_path / "state"), **options}
    # No API key: the facts need no client.
    facts = training_backend_facts(config, model_path=MODEL, environ={})
    assert facts == TinkerFacts(MODEL, service, options["lora_rank"])
    assert facts.artifact_format is None
    assert facts.default_start_inputs == {
        "model": MODEL,
        "lora_rank": str(options["lora_rank"]),
        **({} if service is None else {"service": service}),
    }
    client = RemoteClient()
    component = TinkerTrainingComponent(MODEL, TinkerSettings(state_dir=str(tmp_path / "state"), **options), client)
    try:
        assert component.facts == facts
        rank = options["lora_rank"]
        roots = [
            manifest_root(
                tmp_path / "same", TinkerCheckpoint(MODEL, rank, "tinker://a/state", "tinker://a/s", service=service)
            ),
            manifest_root(
                tmp_path / "rank", TinkerCheckpoint(MODEL, 4, "tinker://b/state", "tinker://b/s", service=service)
            ),
            manifest_root(
                tmp_path / "model",
                TinkerCheckpoint("other/model", rank, "tinker://c/state", "tinker://c/s", service=service),
            ),
            reference_root(tmp_path / "empty", None),
            default_root(tmp_path / "default", facts),
        ]
        assert_same_answers(facts, component, roots, [True, False, False, False, True])
        assert dict(facts.default_start().files) == {BASE_FILE: BaseRoot(facts.default_start_inputs).to_text()}
        # A job from the default start trains a new LoRA of the model. Nothing called the service.
        assert component.start_at(Path(component.root_start(roots[-1]))) == FreshStart(MODEL)
        assert client.initializations == 0
    finally:
        component.close()


@pytest.mark.unit
@pytest.mark.parametrize("commit", [COMMIT, "1" * 40])
def test_the_spindle_facts_of_a_config_answer_as_the_component_of_the_same_config(tmp_path, commit) -> None:
    base = f"hf://{SPINDLE_MODEL}@{commit}"
    settings = spindle_settings(tmp_path, base=base)
    config = {
        "type": SpindleFullDeployment.component_type,
        "state_dir": settings.state_dir,
        "base_url": SPINDLE,
        "base": base,
        "release_repo": settings.release_repo,
    }
    facts = training_backend_facts(config, model_path=SPINDLE_MODEL, environ={})
    revision = HuggingFaceRevision(SPINDLE_MODEL, commit)
    assert facts == SpindleFullFacts(SPINDLE_MODEL, SPINDLE, revision)
    assert facts.artifact_format == ArtifactFormat("full")
    assert facts.default_start_inputs == {"model": base, "service": SPINDLE}
    component = SpindleFullTrainingComponent(SPINDLE_MODEL, settings, SpindleClient(), Exporter())
    try:
        assert component.facts == facts
        roots = [
            manifest_root(
                tmp_path / "full",
                TinkerCheckpoint(SPINDLE_MODEL, None, "tinker://a/state", "tinker://a/s", service=SPINDLE),
            ),
            manifest_root(
                tmp_path / "lora",
                TinkerCheckpoint(SPINDLE_MODEL, 8, "tinker://b/state", "tinker://b/s", service=SPINDLE),
            ),
            reference_root(tmp_path / "base", revision),
            reference_root(tmp_path / "other", HuggingFaceRevision(SPINDLE_MODEL, "2" * 40)),
            reference_root(tmp_path / "files", None),
            default_root(tmp_path / "default", facts),
        ]
        assert_same_answers(facts, component, roots, [True, False, True, False, True, True])
        # The default start is the reference to the base revision, and a job from it starts the base's own weights.
        assert dict(facts.default_start().files) == {REFERENCE_FILE: json.dumps(revision.to_dict()) + "\n"}
        assert component.start_at(Path(component.root_start(roots[-1]))) == FreshStart(SPINDLE_MODEL)
    finally:
        component.close()


@pytest.mark.unit
def test_the_facts_check_their_values() -> None:
    with pytest.raises(ValueError, match="lora_rank must be a positive int"):
        TinkerFacts(MODEL, None, 0)
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        TinkerFacts("", None, 8)
    with pytest.raises(ValueError, match="is not the repo of the base"):
        SpindleFullFacts("other/model", SPINDLE, HuggingFaceRevision.parse(BASE))


def test_the_facts_need_no_sdk_or_training_framework(tmp_path) -> None:
    code = (
        "import sys; "
        "sys.modules.update(dict.fromkeys(('ray', 'torch', 'slime', 'sglang', 'megatron', 'tinker', 'modal'))); "
        "from reef.train.component import training_backend_facts; "
        f"tinker = {{'type': {TinkerDeployment.component_type!r}, 'state_dir': {str(tmp_path)!r}}}; "
        f"assert training_backend_facts(tinker, model_path={MODEL!r}, environ={{}}).name == 'tinker'; "
        f"spindle = {{**tinker, 'type': {SpindleFullDeployment.component_type!r}, 'base': {BASE!r}, "
        f"'base_url': {SPINDLE!r}, 'release_repo': 'org/releases'}}; "
        f"assert training_backend_facts(spindle, model_path={SPINDLE_MODEL!r}, environ={{}}).name == 'spindle_full'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)
