"""Pinned AgentCL export and original-scoring contracts using synthetic fixtures."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tomllib
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from harbor.models.task.config import TaskConfig
from reef_eval import Lab
from reef_eval.types import EpisodeResult, EpisodeSpec

from recipes.agentcl import report, run, taskexport
from recipes.agentcl.harbor.tests import verify
from recipes.agentcl.report import JsonObject

ROOT = Path(__file__).resolve().parents[1]


def manifest_fixture(root: Path) -> report.JsonObject:
    training: list[report.JsonValue] = []
    independent: list[report.JsonValue] = []
    reference_bytes = b"Verified demonstration"
    test_code = "assert candidate() == 1\n"
    for role, count, rows in (("training", 96, training), ("independent", 120, independent)):
        for position in range(count):
            if role == "training":
                category = "raw" if position < 48 else "new"
                identifier = str(position if position < 48 else (position - 47) % 48)
            else:
                category = "new"
                identifier = f"independent-{position}"
            reference = f"privileged/{role}/{position:03d}.py"
            task_relative = f"tasks/{role}/{position:03d}"
            task_path = root / task_relative
            files = {
                "instruction.md": b"Implement candidate().",
                "environment/Dockerfile": b"FROM python:3.12-slim\n",
                "tests/task.json": json.dumps({"test_code": test_code}).encode(),
            }
            for relative, content in files.items():
                path = task_path / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            rows.append(
                {
                    "role": role,
                    "category": category,
                    "id": identifier,
                    "pair_id": identifier,
                    "position": position,
                    "task_path": task_relative,
                    "reference_path": reference,
                    "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
                    "test_sha256": hashlib.sha256(test_code.encode()).hexdigest(),
                    "files": {relative: hashlib.sha256(content).hexdigest() for relative, content in files.items()},
                }
            )
            path = root / reference
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(reference_bytes)
    manifest: report.JsonObject = {
        "schema_version": 1,
        "dataset": "osunlp/AgentCL",
        "revision": taskexport.REVISION,
        "source_files": dict(taskexport.SOURCE_FILES),
        "overlay_version": "fixture",
        "answer_contract": "fixture",
        "corrections_sha256": hashlib.sha256(b"fixture").hexdigest(),
        "training": training,
        "independent": independent,
    }
    report.write_object(root / "manifest.json", manifest)
    manifest_checksum = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    results = []
    for value in training + independent:
        row = cast(report.JsonObject, value)
        results.append(
            {
                **row,
                "task_files": dict(cast(report.JsonObject, row["files"])),
                "reference_key": taskexport.reference_key(manifest_checksum, row),
                "reward": 1,
                "error": None,
            }
        )
    report.write_object(
        root / "reference-verification.json",
        {
            "schema_version": 2,
            "revision": taskexport.REVISION,
            "manifest_sha256": manifest_checksum,
            "status": "passed",
            "total": 216,
            "passed": 216,
            "results": results,
        },
    )
    return manifest


def fixture_streams():
    raw = [
        {
            "id": f"Fixture/{position}",
            "category": "raw",
            "problem": "def base(value):\n",
            "solution": "    return value + 1\n",
            "test_code": "import unittest\nclass TestCases(unittest.TestCase):\n"
            "    def test_value(self):\n        self.assertEqual(base(1), 2)\n",
            "corresponding_new_id": f"Fixture/{position}",
        }
        for position in range(48)
    ]
    complex_tasks = [
        {
            "id": f"Fixture/{position}",
            "category": "new",
            "problem": "# Sum two incremented inputs.\ndef combined(left, right):\n",
            "solution": "    return base(left) + base(right)\n",
            "test_code": "def base(value):\n    return value + 1\nassert combined(1, 2) == 5\n",
            "corresponding_raw_id": f"Fixture/{position}",
        }
        for position in range(48)
    ]
    independent = [{**complex_tasks[0], "id": f"IndependentFixture/{position}"} for position in range(120)]
    return raw + complex_tasks, complex_tasks, independent


def test_export_preserves_order_and_keeps_hidden_assets_out_of_student_image(tmp_path, monkeypatch):
    module = taskexport
    streams = fixture_streams()
    sources = dict(zip(module.SOURCE_FILES, streams, strict=True))
    monkeypatch.setattr(module, "read_source", lambda filename, _cache: sources[filename])
    cache = tmp_path / "cache"
    cache.mkdir()
    for filename, rows in sources.items():
        (cache / filename).write_text(json.dumps(rows))
    output = tmp_path / "export"
    manifest = module.export_tasks(output, tmp_path / "cache")
    assert len(manifest["training"]) == 96
    assert len(manifest["independent"]) == 120
    assert [row["category"] for row in manifest["training"]] == ["raw"] * 48 + ["new"] * 48
    assert [row["id"] for row in manifest["training"]] == [row["id"] for row in streams[0]]
    assert manifest["reference_verification"] == "not-run"
    for row in [*manifest["training"], *manifest["independent"]]:
        task = output / row["task_path"]
        original = sources["bigcodebench_lite_pro.dependent.json"] if row["role"] == "training" else streams[2]
        source = original[row["position"]]
        instruction = (task / "instruction.md").read_text()
        assert instruction.endswith(source["problem"])
        assert source["test_code"] not in instruction
        assert source["id"] not in instruction
        assert source["solution"] not in instruction
        hidden = json.loads((task / "tests" / "task.json").read_text())
        assert hidden["test_code"] == source["test_code"]
        assert "solution" not in hidden
        assert (output / row["reference_path"]).read_text().endswith(source["problem"] + "\n" + source["solution"])
        assert row["test_sha256"] == row["original_test_sha256"]
        assert manifest["task_counts"] == {"training": 96, "independent": 120}
        assert (output / "original" / "bigcodebench_lite_pro.dependent.json").read_text() == (
            cache / "bigcodebench_lite_pro.dependent.json"
        ).read_text()
        assert not (task / "environment" / "task.json").exists()
        configuration = TaskConfig.model_validate(tomllib.loads((task / "task.toml").read_text()))
        assert configuration.verifier.environment_mode.value == "separate"
        assert configuration.environment.network_mode.value == "no-network"
        assert configuration.verifier.environment.network_mode.value == "no-network"
    with pytest.raises(ValueError, match="empty"):
        module.export_tasks(output, tmp_path / "cache")


def test_stream_validation_rejects_reordering_bad_pairs_and_independent_roles():
    module = taskexport
    dependent, conventional, independent = fixture_streams()
    module.validate_streams(dependent, conventional, independent)
    with pytest.raises(ValueError, match="order"):
        module.validate_streams(dependent, conventional[::-1], independent)
    dependent[0]["corresponding_new_id"] = "wrong"
    with pytest.raises(ValueError, match="pair link"):
        module.validate_streams(dependent, conventional, independent)
    dependent, conventional, independent = fixture_streams()
    independent[0]["category"] = "raw"
    with pytest.raises(ValueError, match="120 complex"):
        module.validate_streams(dependent, conventional, independent)


def test_pinned_source_rejects_tampered_cache_without_network(tmp_path):
    module = taskexport
    filename = next(iter(module.SOURCE_FILES))
    (tmp_path / filename).write_text("[]")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.read_source(filename, tmp_path)


@pytest.mark.parametrize("style", ["assertions", "unittest"])
def test_original_scoring_pass_bad_invalid_and_timeout_fixture_only(style, tmp_path):
    verifier = verify
    if style == "assertions":
        tests = "def base(value):\n    return value + 1\nassert combined(1, 2) == 5\n"
    else:
        tests = (
            "import unittest\nclass TestCases(unittest.TestCase):\n"
            "    def test_value(self):\n        self.assertEqual(combined(1, 2), 5)\n"
        )
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"test_code": tests}))
    answer = tmp_path / "answer.py"
    correct = "def combined(left,right):\n    return left + right + 2\n"
    program = verifier.assemble_program(correct, tests)
    assert correct + "\n\n" + tests in program
    answer.write_text(correct)
    assert verifier.grade(task, answer, 2, isolate_user=False)["reward"] == 1
    answer.write_text("def combined(left,right):\n    return 0\n")
    failed = verifier.grade(task, answer, 2, isolate_user=False)
    assert failed["reward"] == 0 and failed["status"] == "failed"
    assert "combined(1, 2)" not in failed["feedback"]
    answer.write_text("import sys\nsys.exit(0)\n")
    assert verifier.grade(task, answer, 2, isolate_user=False)["reward"] == 0
    answer.write_text("import os\nos._exit(0)\n")
    assert verifier.grade(task, answer, 2, isolate_user=False)["reward"] == 0
    answer.write_text("not valid python !")
    assert verifier.grade(task, answer, 2, isolate_user=False)["status"] == "invalid"
    answer.write_text("def combined(left,right):\n    while True:\n        pass\n")
    assert verifier.grade(task, answer, 0.05, isolate_user=False)["status"] == "timeout"
    answer.unlink()
    assert verifier.grade(task, answer, 2, isolate_user=False)["status"] == "invalid"
    task.write_text(json.dumps({"test_code": "not valid python !"}))
    with pytest.raises(verifier.VerifierFault, match="hidden test code"):
        verifier.grade(task, answer, 2, isolate_user=False)
    task.write_text(json.dumps({"test_code": "combined(1, 2)\n"}))
    answer.write_text(correct)
    assert verifier.grade(task, answer, 2, isolate_user=False)["reward"] == 1
    answer.write_text("def combined(left,right):\n    raise ValueError('invalid')\n")
    assert verifier.grade(task, answer, 2, isolate_user=False)["reward"] == 0


def test_harness_and_verifier_share_the_final_answer_contract():
    example = ROOT / "recipes/agentcl"
    assert (example / "harness/answer_contract.py").read_bytes() == (
        example / "harbor/tests/answer_contract.py"
    ).read_bytes()


@pytest.mark.parametrize(
    "attack",
    [
        "import sys,os\nos.write(1,sys._getframe(1).f_locals['completion_key'])\nos._exit(0)\n",
        "import builtins\nbuiltins.len = lambda _: 0\n",
        "import unittest\nunittest.TestCase.assertEqual = lambda *args: None\n",
        "import inspect\ninspect.currentframe()\n",
        "__import__('os').system('true')\n",
        "exec('pass')\n",
        "eval('1')\n",
        "getattr((), '__class__')\n",
        "value = ().__class__.__base__.__subclasses__()\n",
        "raise SystemExit(0)\n",
        "globals()['combined'] = lambda *args: 5\n",
        "len = lambda *args: 0\n",
        "def isinstance(*args):\n    return True\n",
    ],
)
def test_restricted_final_contract_rejects_known_grader_attacks(attack, tmp_path):
    verifier = verify
    task = tmp_path / "task.json"
    answer = tmp_path / "answer.py"
    task.write_text(json.dumps({"test_code": "assert False\n"}))
    answer.write_text("def combined(left, right):\n    return 0\n" + attack)
    result = verifier.grade(task, answer, 2, isolate_user=False)
    assert result["reward"] == 0 and result["status"] == "invalid"


def test_versioned_definition_order_preserves_assertions_and_candidate_authority(tmp_path):
    verifier = verify
    tests = "def collision(n):\n    return n ** 2\n\nassert collision(3) + collision(3) == 6\n"
    task = tmp_path / "task.json"
    answer = tmp_path / "answer.py"
    task.write_text(json.dumps({"test_code": tests, "tests_before_candidate_definition": True}))
    correct = "def collision(n):\n    return n * (n - 1) // 2\n"
    answer.write_text(correct)
    assert verifier.grade(task, answer, 2, isolate_user=False)["reward"] == 1
    program = verifier.assemble_program(correct, tests, tests_before_candidate_definition=True)
    assert "assert collision(3) + collision(3) == 6" in program
    answer.write_text("def collision(n):\n    return 0\n")
    assert verifier.grade(task, answer, 2, isolate_user=False)["reward"] == 0


def test_gold_only_annotation_import_prefix_and_explicit_separator():
    module = taskexport
    row = {
        "id": "Fixture",
        "category": "new",
        "problem": "def combine(values: List[int]) -> List[int]:\n",
        "solution": "    return values\n",
        "test_code": "from typing import List\nassert combine([1]) == [1]\n",
    }
    tests, reference, definitions_first, changes = module.corrected_task(row, "independent")
    assert tests == row["test_code"] and not definitions_first
    assert reference.startswith("from typing import List\n")
    assert row["problem"] + "\n" + row["solution"] in reference
    assert "assert combine" not in reference
    assert changes[-1]["operation"] == "gold_only_test_import_prefix_and_problem_separator"


@pytest.mark.parametrize("style", ["loop", "test_function"])
def test_original_nested_and_named_test_checks_are_executed(style, tmp_path):
    verifier = verify
    if style == "loop":
        tests = "for value in [1,2]:\n    assert answer(value) == value + 1\n"
    else:
        tests = "def test_answer():\n    assert answer(1) == 2\n"
    task = tmp_path / "task.json"
    answer = tmp_path / "answer.py"
    task.write_text(json.dumps({"test_code": tests}))
    answer.write_text("def answer(value):\n    return value + 1\n")
    assert verifier.grade(task, answer, 2, isolate_user=False)["reward"] == 1
    answer.write_text("def answer(value):\n    return 0\n")
    assert verifier.grade(task, answer, 2, isolate_user=False)["reward"] == 0


@pytest.mark.parametrize(
    "change", ["instruction", "verifier", "image", "missing", "extra", "empty", "outside", "symlink"]
)
def test_manifest_rejects_changed_or_incompletely_hashed_task_files(change, tmp_path):
    manifest = manifest_fixture(tmp_path)
    run.load_manifest(tmp_path)
    row = manifest["training"][0]
    task = tmp_path / row["task_path"]
    if change in ("instruction", "verifier", "image"):
        relative = {"instruction": "instruction.md", "verifier": "tests/task.json", "image": "environment/Dockerfile"}[
            change
        ]
        (task / relative).write_bytes(b"changed")
    elif change == "missing":
        (task / "instruction.md").unlink()
    elif change == "extra":
        (task / "unlisted.txt").write_bytes(b"extra")
    elif change == "empty":
        row["files"] = {}
    elif change == "outside":
        row["files"]["../manifest.json"] = hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()
    else:
        (task / "instruction.md").unlink()
        (task / "instruction.md").symlink_to(tmp_path / row["reference_path"])
    report.write_object(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="exported task"):
        run.load_manifest(tmp_path)


@pytest.mark.parametrize("change", ["missing", "legacy", "manifest", "test", "files", "key", "task-bytes"])
def test_reference_prerequisite_blocks_sampling_for_unverified_or_changed_tasks(change, tmp_path):
    manifest = manifest_fixture(tmp_path)
    path = tmp_path / "reference-verification.json"
    if change == "missing":
        path.unlink()
    elif change == "task-bytes":
        task = tmp_path / manifest["training"][0]["task_path"]
        (task / "instruction.md").write_bytes(b"changed")
    else:
        qualification = report.read_object(path)
        if change == "legacy":
            qualification["schema_version"] = 1
        elif change == "manifest":
            qualification["manifest_sha256"] = "wrong"
        elif change == "test":
            qualification["results"][0]["test_sha256"] = "wrong"
        elif change == "files":
            qualification["results"][0]["task_files"] = {}
        else:
            qualification["results"][0]["reference_key"] = "wrong"
        report.write_object(path, qualification)
    arguments = argparse.Namespace(
        run_root=tmp_path / "run",
        data_root=tmp_path,
        method="sdpo",
        run_id="fixture",
        model_path="fixture",
        teacher_checkpoint="",
        profile="smoke",
        steps=2,
        attempts=2,
        seed=42,
        max_turns=8,
        scenario="fixture",
        service_url="http://fixture",
        wandb_project=None,
        wandb_entity=None,
    )
    api = Mock(spec=run.ReefApi)
    api.commits.return_value = []
    api.current_release.return_value = {"release_id": "base", "operation": "creation"}
    backend = Mock(spec=run.EpisodeBackend)
    campaign = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises((RuntimeError, ValueError), match=r"reference|exported task"):
        asyncio.run(campaign.execute_phase("train"))
    backend.run.assert_not_called()
    api.report.assert_not_called()
    api.commits.assert_called_once_with([])


def test_reference_results_reject_changed_manifest_even_with_matching_reference(tmp_path):
    manifest = manifest_fixture(tmp_path)
    task = manifest["training"][0]
    verifier = tmp_path / task["task_path"] / "tests/task.json"
    test_code = "assert candidate() == 2\n"
    verifier.write_text(json.dumps({"test_code": test_code}))
    task["files"]["tests/task.json"] = hashlib.sha256(verifier.read_bytes()).hexdigest()
    task["test_sha256"] = hashlib.sha256(test_code.encode()).hexdigest()
    report.write_object(tmp_path / "manifest.json", manifest)
    run.load_manifest(tmp_path)
    with pytest.raises(RuntimeError, match="fresh qualification"):
        run.validate_reference_verification(tmp_path, manifest, "sdpo")


class ReferenceFixtureExecutor:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, spec: EpisodeSpec) -> EpisodeResult:
        self.calls += 1
        return EpisodeResult(rewards={"reward": 1.0})


@pytest.mark.parametrize("change", ["verifier", "image"])
def test_reference_cache_is_bound_to_changed_verifier_or_image(change, tmp_path, monkeypatch):
    import reef_eval

    module = taskexport
    root = tmp_path / "data"
    manifest = manifest_fixture(root)
    executor = ReferenceFixtureExecutor()
    lab = Lab(tmp_path / "lab", executor=executor)

    def lab_fixture(_root: Path) -> Lab:
        return lab

    monkeypatch.setattr(reef_eval, "Lab", lab_fixture)
    first = asyncio.run(module.verify_references(root, tmp_path / "lab"))
    assert first["status"] == "passed" and executor.calls == 216
    same = asyncio.run(module.verify_references(root, tmp_path / "lab"))
    assert same["results"] == first["results"] and executor.calls == 216
    task = cast(JsonObject, manifest["training"][0])
    if change == "verifier":
        relative = "tests/task.json"
        content = json.dumps({"test_code": "assert candidate() == 2\n"}).encode()
        task["test_sha256"] = hashlib.sha256(b"assert candidate() == 2\n").hexdigest()
    else:
        relative = "environment/Dockerfile"
        content = b"FROM python:3.12-bookworm\n"
    path = root / str(task["task_path"]) / relative
    path.write_bytes(content)
    cast(JsonObject, task["files"])[relative] = hashlib.sha256(content).hexdigest()
    report.write_object(root / "manifest.json", manifest)
    second = asyncio.run(module.verify_references(root, tmp_path / "lab"))
    assert second["status"] == "passed" and executor.calls == 432
    assert first["results"][0]["reference_key"] != second["results"][0]["reference_key"]
    assert first["manifest_sha256"] != second["manifest_sha256"]
    run.validate_reference_verification(root, manifest, "sdpo")
