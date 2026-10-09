"""Pinned AgentCL export and original-scoring contracts using synthetic fixtures."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import tomllib
from pathlib import Path
from typing import cast

import pytest
from harbor.models.task.config import TaskConfig
from reef_eval import Lab
from reef_eval.types import EpisodeResult, EpisodeSpec

from recipes.opd.examples.agentcl.report import JsonObject

ROOT = Path(__file__).resolve().parents[1]


def exporter(method: str):
    return importlib.import_module(f"recipes.{method}.examples.agentcl.taskexport")


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


@pytest.mark.parametrize("method", ["sdft", "sdpo", "opd"])
def test_export_preserves_order_and_keeps_hidden_assets_out_of_student_image(method, tmp_path, monkeypatch):
    module = exporter(method)
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


@pytest.mark.parametrize("method", ["sdft", "sdpo", "opd"])
def test_stream_validation_rejects_reordering_bad_pairs_and_independent_roles(method):
    module = exporter(method)
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


@pytest.mark.parametrize("method", ["sdft", "sdpo", "opd"])
def test_pinned_source_rejects_tampered_cache_without_network(method, tmp_path):
    module = exporter(method)
    filename = next(iter(module.SOURCE_FILES))
    (tmp_path / filename).write_text("[]")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.read_source(filename, tmp_path)


def load_verifier(method: str):
    return importlib.import_module(f"recipes.{method}.examples.agentcl.harbor.tests.verify")


@pytest.mark.parametrize("method", ["sdft", "sdpo", "opd"])
@pytest.mark.parametrize("style", ["assertions", "unittest"])
def test_original_scoring_pass_bad_invalid_and_timeout_fixture_only(method, style, tmp_path):
    verifier = load_verifier(method)
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


def test_method_local_data_harness_and_verifier_assets_are_identical():
    sdft = ROOT / "recipes/sdft/examples/agentcl"
    sdpo = ROOT / "recipes/sdpo/examples/agentcl"
    opd = ROOT / "recipes/opd/examples/agentcl"
    paths = [Path("taskexport.py")]
    paths.extend(
        path.relative_to(sdft)
        for directory in ("harness", "harbor")
        for path in (sdft / directory).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for relative in paths:
        if relative.name == "requirements.txt":
            assert (
                sorted((sdft / relative).read_text().splitlines())
                == sorted((sdpo / relative).read_text().splitlines())
                == (opd / relative).read_text().splitlines()
            ), relative
        else:
            assert (
                (sdft / relative).read_bytes() == (sdpo / relative).read_bytes() == (opd / relative).read_bytes()
            ), relative


@pytest.mark.parametrize("method", ["sdft", "sdpo", "opd"])
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
def test_restricted_final_contract_rejects_known_grader_attacks(method, attack, tmp_path):
    verifier = load_verifier(method)
    task = tmp_path / "task.json"
    answer = tmp_path / "answer.py"
    task.write_text(json.dumps({"test_code": "assert False\n"}))
    answer.write_text("def combined(left, right):\n    return 0\n" + attack)
    result = verifier.grade(task, answer, 2, isolate_user=False)
    assert result["reward"] == 0 and result["status"] == "invalid"


@pytest.mark.parametrize("method", ["sdft", "sdpo", "opd"])
def test_versioned_definition_order_preserves_assertions_and_candidate_authority(method, tmp_path):
    verifier = load_verifier(method)
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


@pytest.mark.parametrize("method", ["sdft", "sdpo", "opd"])
def test_gold_only_annotation_import_prefix_and_explicit_separator(method):
    module = exporter(method)
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


@pytest.mark.parametrize("method", ["sdft", "sdpo", "opd"])
@pytest.mark.parametrize("style", ["loop", "test_function"])
def test_original_nested_and_named_test_checks_are_executed(method, style, tmp_path):
    verifier = load_verifier(method)
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


@pytest.mark.parametrize("method", ["opd", "sdft", "sdpo"])
@pytest.mark.parametrize(
    "change", ["instruction", "verifier", "image", "missing", "extra", "empty", "outside", "symlink"]
)
def test_manifest_rejects_changed_or_incompletely_hashed_task_files(method, change, tmp_path):
    workflow = importlib.import_module(f"tests.test_{method}_agentcl")
    manifest = workflow.manifest_fixture(tmp_path)
    workflow.run.load_manifest(tmp_path)
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
    workflow.report.write_object(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="exported task"):
        workflow.run.load_manifest(tmp_path)


@pytest.mark.parametrize("method", ["opd", "sdft", "sdpo"])
@pytest.mark.parametrize("change", ["missing", "legacy", "manifest", "test", "files", "key", "task-bytes"])
def test_reference_prerequisite_blocks_all_methods_before_sampling(method, change, tmp_path):
    workflow = importlib.import_module(f"tests.test_{method}_agentcl")
    arguments = workflow.arguments_fixture(tmp_path)
    manifest = workflow.manifest_fixture(arguments.data_root)
    path = arguments.data_root / "reference-verification.json"
    if change == "missing":
        path.unlink()
    elif change == "task-bytes":
        task = arguments.data_root / manifest["training"][0]["task_path"]
        (task / "instruction.md").write_bytes(b"changed")
    else:
        qualification = workflow.report.read_object(path)
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
        workflow.report.write_object(path, qualification)
    if method == "opd":
        api = workflow.FakeApi()
        backend = workflow.FakeEpisodes(api)
    else:
        api = workflow.FakeApi(attempts=arguments.attempts)
        backend = workflow.FakeEpisodes()
    campaign = workflow.run.Campaign(arguments, api, backend, manifest)
    with pytest.raises((RuntimeError, ValueError), match=r"reference|exported task"):
        asyncio.run(campaign.execute_phase("train"))
    assert not backend.calls and not api.reports and not api.history


@pytest.mark.parametrize("method", ["opd", "sdft", "sdpo"])
def test_reference_results_reject_changed_manifest_even_with_matching_reference(method, tmp_path):
    workflow = importlib.import_module(f"tests.test_{method}_agentcl")
    manifest = workflow.manifest_fixture(tmp_path)
    task = manifest["training"][0]
    verifier = tmp_path / task["task_path"] / "tests/task.json"
    test_code = "assert candidate() == 2\n"
    verifier.write_text(json.dumps({"test_code": test_code}))
    task["files"]["tests/task.json"] = hashlib.sha256(verifier.read_bytes()).hexdigest()
    task["test_sha256"] = hashlib.sha256(test_code.encode()).hexdigest()
    workflow.report.write_object(tmp_path / "manifest.json", manifest)
    workflow.run.load_manifest(tmp_path)
    with pytest.raises(RuntimeError, match="fresh qualification"):
        workflow.run.validate_reference_verification(tmp_path, manifest)


class ReferenceFixtureExecutor:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, spec: EpisodeSpec) -> EpisodeResult:
        self.calls += 1
        return EpisodeResult(rewards={"reward": 1.0})


@pytest.mark.parametrize("method", ["opd", "sdft", "sdpo"])
@pytest.mark.parametrize("change", ["verifier", "image"])
def test_reference_cache_is_bound_to_changed_verifier_or_image(method, change, tmp_path, monkeypatch):
    import reef_eval

    workflow = importlib.import_module(f"tests.test_{method}_agentcl")
    module = exporter(method)
    root = tmp_path / "data"
    manifest = workflow.manifest_fixture(root)
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
    workflow.report.write_object(root / "manifest.json", manifest)
    second = asyncio.run(module.verify_references(root, tmp_path / "lab"))
    assert second["status"] == "passed" and executor.calls == 432
    assert first["results"][0]["reference_key"] != second["results"][0]["reference_key"]
    assert first["manifest_sha256"] != second["manifest_sha256"]
    workflow.run.validate_reference_verification(root, manifest)
