"""Export the pinned AgentCL coding streams without exposing hidden assets to agents."""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import cast

if __package__:
    from .harness.answer_contract import CONTRACT_VERSION, validate_final_module
    from .report import JsonObject
else:
    from report import JsonObject

    from harness.answer_contract import CONTRACT_VERSION, validate_final_module

DATASET = "osunlp/AgentCL"
REVISION = "a01e2ca6e33fd07d9cf80e4bd69a5b3585d3400f"
SOURCE_FILES = {
    "bigcodebench_lite_pro.dependent.json": "cd462894801f087866019d12d85f669d1ddd72b97ef722dcd180587868ad6df2",
    "bigcodebench_lite_pro.conventional.json": "55d18f52f7e2562514dfc5bb4d75b161ce58861b1e50ea298e4e0033bbe0b636",
    "humaneval_pro.conventional.json": "c15f68d46795087d77dc6650b894fbba4d94e2e85fe8cc47f998c3cfdb79075b",
}


def validate_task_files(root: Path, row: JsonObject) -> None:
    """Require every exported task file to match its manifest hash inside the data root."""
    relative_task = row.get("task_path")
    files = row.get("files")
    if not isinstance(relative_task, str) or not relative_task:
        raise ValueError("exported task requires a relative task path")
    task_relative_path = Path(relative_task)
    if task_relative_path.is_absolute() or ".." in task_relative_path.parts or not task_relative_path.parts:
        raise ValueError("exported task paths must remain under the data root")
    if not isinstance(files, dict) or not files:
        raise ValueError("exported task requires a nonempty file checksum map")
    root = root.resolve()
    task_path = root / task_relative_path
    if any(
        (root / Path(*task_relative_path.parts[:index])).is_symlink()
        for index in range(1, len(task_relative_path.parts) + 1)
    ):
        raise ValueError("exported task paths must not use symlinks")
    if not task_path.resolve().is_relative_to(root) or not task_path.is_dir():
        raise ValueError("exported task directory is missing or outside the data root")
    for relative, checksum in files.items():
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
            or not isinstance(checksum, str)
        ):
            raise ValueError("exported task file checksum paths must stay inside the task directory")
    actual_files: dict[str, str] = {}
    for path in sorted(task_path.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(task_path):
            raise ValueError("exported task files must not use symlinks or leave the task directory")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("exported task entries must be regular files or directories")
        actual_files[path.relative_to(task_path).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_files != files:
        raise ValueError("exported task checksum mismatch or incomplete file coverage")


def reference_key(manifest_checksum: str, row: JsonObject) -> str:
    """Bind cached reference results to the export, task files, and reference bytes."""
    input_checksum = hashlib.sha256(
        json.dumps(
            {
                "manifest_sha256": manifest_checksum,
                "task_files": row["files"],
                "reference_sha256": row["reference_sha256"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return f"reference/v2/{REVISION}/{row['role']}/{row['position']}/{input_checksum}"


def read_source(filename: str, cache_dir: Path) -> list[dict[str, str]]:
    """Fetch a small public file at its immutable revision and verify every byte."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    if path.exists():
        content = path.read_bytes()
    else:
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/codeeval-pro/{filename}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read(2_000_000)
    if hashlib.sha256(content).hexdigest() != SOURCE_FILES[filename]:
        raise ValueError(f"dataset checksum mismatch: {filename}")
    if not path.exists():
        path.write_bytes(content)
    rows = json.loads(content)
    if not isinstance(rows, list):
        raise ValueError(f"expected an ordered JSON task array: {filename}")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"invalid task row: {filename}")
        for key in ("id", "category", "problem", "solution", "test_code"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"invalid task {key}: {filename}")
        ast.parse(row["problem"] + row["solution"])
        ast.parse(row["test_code"])
    return rows


def validate_streams(
    dependent: list[dict[str, str]], conventional: list[dict[str, str]], independent: list[dict[str, str]]
) -> None:
    """Require the supplied order, full task counts and one-to-one pair links."""
    if len(dependent) != 96 or [row["category"] for row in dependent] != ["raw"] * 48 + ["new"] * 48:
        raise ValueError("dependent stream must contain 48 raw tasks followed by 48 new tasks")
    if len(conventional) != 48 or dependent[48:] != conventional:
        raise ValueError("dependent complex tasks must match the conventional stream exactly in order")
    if len(independent) != 120 or any(row["category"] != "new" for row in independent):
        raise ValueError("independent stream must contain 120 complex HumanEval tasks")
    for rows in (dependent, independent):
        identities = {(row["category"], row["id"]) for row in rows}
        if len(identities) != len(rows):
            raise ValueError("duplicate (category, id) task identity")
    raw_ids = {row["id"] for row in dependent[:48]}
    new_ids = {row["id"] for row in dependent[48:]}
    if raw_ids != new_ids:
        raise ValueError("dependent task pair IDs differ")
    for row in dependent:
        if row["category"] == "raw":
            if row.get("corresponding_new_id") != row["id"]:
                raise ValueError("invalid dependent pair link")
        elif row.get("corresponding_raw_id") != row["id"]:
            raise ValueError("invalid dependent pair link")


def corrected_task(row: dict[str, str], role: str) -> tuple[str, str, bool, list[dict[str, object]]]:
    """Apply only approved hash-checked scoring corrections and gold-only imports."""
    test_code = row["test_code"]
    changes: list[dict[str, object]] = []
    definitions_first = role == "independent" and row["category"] == "new" and row["id"] == "41"
    if row["category"] == "raw" and row["id"] == "BigCodeBench/269":
        corrections_dir = Path(__file__).parent / "harbor" / "corrections"
        overlay = json.loads((corrections_dir / "manifest.json").read_text())
        correction = next(change for change in overlay["changes"] if change["id"] == "raw-269-tests")
        if hashlib.sha256(test_code.encode()).hexdigest() != correction["original_sha256"]:
            raise ValueError("raw269 original test checksum differs from the approved overlay")
        test_code = (corrections_dir / correction["replacement_file"]).read_text()
        if hashlib.sha256(test_code.encode()).hexdigest() != correction["replacement_sha256"]:
            raise ValueError("raw269 upstream replacement checksum mismatch")
        changes.append(correction)
    reference_override: str | None = None
    if row["category"] == "new" and row["id"] == "BigCodeBench/217":
        corrections_dir = Path(__file__).parent / "harbor" / "corrections"
        overlay = json.loads((corrections_dir / "manifest.json").read_text())
        correction = next(change for change in overlay["changes"] if change["id"] == "new-217-test-and-reference")
        if hashlib.sha256(test_code.encode()).hexdigest() != correction["original_sha256"]:
            raise ValueError("task217 original test checksum differs from the recorded correction")
        test_code = (corrections_dir / correction["replacement_file"]).read_text()
        reference_override = (corrections_dir / correction["reference_file"]).read_text()
        if hashlib.sha256(test_code.encode()).hexdigest() != correction["replacement_sha256"]:
            raise ValueError("task217 corrected test checksum differs")
        if hashlib.sha256(reference_override.encode()).hexdigest() != correction["reference_sha256"]:
            raise ValueError("task217 corrected reference checksum differs")
        changes.append(correction)
    if definitions_first:
        changes.append({"id": "humaneval-41-definition-order", "operation": "tests_before_candidate_definition"})
    tree = ast.parse(test_code)
    original_reference = (
        reference_override if reference_override is not None else row["problem"] + "\n" + row["solution"]
    )
    reference_tree = ast.parse(original_reference)
    annotation_names: set[str] = set()
    imported_names: set[str] = set()
    for node in ast.walk(reference_tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported_names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.arg) and node.annotation is not None:
            annotation_names.update(part.id for part in ast.walk(node.annotation) if isinstance(part, ast.Name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            annotation_names.update(part.id for part in ast.walk(node.returns) if isinstance(part, ast.Name))
    missing_annotation_names = annotation_names - imported_names
    imports = [
        ast.get_source_segment(test_code, node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any((alias.asname or alias.name.split(".")[0]) in missing_annotation_names for alias in node.names)
    ]
    prefix = "\n".join(statement for statement in imports if statement is not None)
    reference = prefix + "\n\n" + original_reference
    validate_final_module(reference)
    changes.append(
        {
            "id": "reference-assembly",
            "operation": "gold_only_test_import_prefix_and_problem_separator",
            "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
            "original_reference_sha256": hashlib.sha256((row["problem"] + row["solution"]).encode()).hexdigest(),
            "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
        }
    )
    return test_code, reference, definitions_first, changes


def export_row(row: dict[str, str], role: str, position: int, output_dir: Path) -> dict[str, object]:
    """Materialize student-only build inputs and a separate hidden verifier image."""
    relative_task = Path("tasks") / role / f"{position:03d}"
    task_path = output_dir / relative_task
    task_path.mkdir(parents=True, exist_ok=False)
    template = Path(__file__).parent / "harbor"
    shutil.copytree(template / "environment", task_path / "environment")
    shutil.copytree(template / "tests", task_path / "tests")
    shutil.copyfile(template / "task.toml", task_path / "task.toml")
    test_code, reference, definitions_first, corrections = corrected_task(row, role)
    instruction = (
        "Implement the following Python problem. Submit a complete Python module, including its imports and "
        "function definition, using FINAL followed by one fenced python code block. During development you may "
        "execute one fenced python code block per response. The interpreter retains variables and files during "
        "this episode only. Hidden verification supplies any base helper used by the original task, but that "
        "helper and hidden tests are not available in your development interpreter. Final modules may use standard "
        "numeric, scientific and collection libraries, but not process-control or interpreter-introspection modules, "
        "dynamic imports, eval/exec, private or dunder attribute access, or attribute/global namespace mutation. "
        "Submit a top-level function. This restricted syntax is checked before submission and grading; it is not "
        "a general proof against malicious Python. Do not run background processes.\n\n" + row["problem"]
    )
    (task_path / "instruction.md").write_text(instruction, encoding="utf-8")
    verifier = {
        "problem": row["problem"],
        "test_code": test_code,
        "category": row["category"],
        "tests_before_candidate_definition": definitions_first,
        "answer_contract": CONTRACT_VERSION,
    }
    (task_path / "tests" / "task.json").write_text(json.dumps(verifier, ensure_ascii=False), encoding="utf-8")
    reference_path = Path("privileged") / role / f"{position:03d}.py"
    (output_dir / reference_path).parent.mkdir(parents=True, exist_ok=True)
    (output_dir / reference_path).write_text(reference, encoding="utf-8")
    (output_dir / reference_path).chmod(0o600)
    checksums = {
        path.relative_to(task_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(task_path.rglob("*"))
        if path.is_file()
    }
    return {
        "id": row["id"],
        "category": row["category"],
        "pair_id": row["id"],
        "position": position,
        "role": role,
        "task_path": relative_task.as_posix(),
        "reference_path": reference_path.as_posix(),
        "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
        "original_test_sha256": hashlib.sha256(row["test_code"].encode()).hexdigest(),
        "test_sha256": hashlib.sha256(test_code.encode()).hexdigest(),
        "check_kind": (
            "unittest"
            if any(isinstance(node, ast.ClassDef) for node in ast.parse(test_code).body)
            else "assertions" if any(isinstance(node, ast.Assert) for node in ast.parse(test_code).body) else "calls"
        ),
        "corrections": corrections,
        "files": checksums,
    }


def export_tasks(output_dir: Path, cache_dir: Path) -> dict[str, object]:
    """Export 96 training tasks and 120 held-out tasks; never execute benchmark code."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("export output directory must be empty; use a new pinned export root")
    dependent, conventional, independent = [read_source(filename, Path(cache_dir)) for filename in SOURCE_FILES]
    validate_streams(dependent, conventional, independent)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "privileged").mkdir(mode=0o700)
    (output_dir / "original").mkdir(mode=0o700)
    for filename in SOURCE_FILES:
        shutil.copyfile(Path(cache_dir) / filename, output_dir / "original" / filename)
    correction_manifest = Path(__file__).parent / "harbor" / "corrections" / "manifest.json"
    shutil.copyfile(correction_manifest, output_dir / "corrections.json")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": DATASET,
        "revision": REVISION,
        "license": "CC-BY-NC-4.0; underlying source terms also apply; redistribution/commercial use needs review",
        "source_files": dict(SOURCE_FILES),
        "scoring": "restricted candidate then original checks, with explicit raw269 upstream test and task41 definition-order corrections",
        "overlay_version": "agentcl-coding-overlay-v1",
        "corrections_sha256": hashlib.sha256(correction_manifest.read_bytes()).hexdigest(),
        "answer_contract": CONTRACT_VERSION,
        "task_counts": {"training": 96, "independent": 120},
        "reference_verification": "not-run",
        "training": [export_row(row, "training", position, output_dir) for position, row in enumerate(dependent)],
        "independent": [
            export_row(row, "independent", position, output_dir) for position, row in enumerate(independent)
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


async def verify_references(output_dir: Path, lab_dir: Path) -> dict[str, object]:
    """Verify all gold modules in fresh Harbor containers without model requests."""
    from reef_eval import Lab

    manifest_bytes = (output_dir / "manifest.json").read_bytes()
    manifest = cast(JsonObject, json.loads(manifest_bytes))
    manifest_checksum = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest.get("revision") != REVISION:
        raise ValueError("reference verification requires the pinned AgentCL export")
    rows = cast(list[JsonObject], manifest["training"]) + cast(list[JsonObject], manifest["independent"])
    if len(rows) != 216:
        raise ValueError("reference verification requires all 216 exported tasks")
    lab = Lab(lab_dir)
    if __package__:
        agent_import = f"{__package__}.harness.reference:ReferenceAgent"
    else:
        agent_import = "harness.reference:ReferenceAgent"
    results: list[dict[str, object]] = []
    verification: dict[str, object] = {
        "schema_version": 2,
        "revision": REVISION,
        "overlay_version": manifest["overlay_version"],
        "answer_contract": manifest["answer_contract"],
        "corrections_sha256": manifest["corrections_sha256"],
        "manifest_sha256": manifest_checksum,
        "status": "running",
        "total": 216,
        "passed": 0,
        "results": results,
    }
    artifact_path = output_dir / "reference-verification.json"
    for row in rows:
        validate_task_files(output_dir, row)
        reference_path = (output_dir / str(row["reference_path"])).resolve()
        if not reference_path.is_relative_to(output_dir.resolve()):
            raise ValueError("reference paths must remain under the data root")
        reference_checksum = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        if reference_checksum != row["reference_sha256"]:
            raise ValueError("reference checksum mismatch")
        task_path = output_dir / str(row["task_path"])
        key = reference_key(manifest_checksum, row)
        result = await lab.run(
            str(task_path),
            agent={
                "import_path": agent_import,
                "kwargs": {"reference_path": str(reference_path)},
            },
            key=key,
        )
        reward = result.rewards.get("reward")
        error = result.tags.get("error")
        results.append(
            {
                "id": row["id"],
                "category": row["category"],
                "role": row["role"],
                "position": row["position"],
                "reference_sha256": reference_checksum,
                "reference_key": key,
                "test_sha256": row["test_sha256"],
                "task_files": row["files"],
                "reward": reward,
                "error": error,
                "uri": result.uri,
            }
        )
        verification["passed"] = sum(row["reward"] == 1 and not row["error"] for row in results)
        if len(results) == 216:
            verification["status"] = "passed" if verification["passed"] == 216 else "failed"
        temporary = artifact_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
        temporary.replace(artifact_path)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--verify-references", action="store_true")
    parser.add_argument("--lab-dir", type=Path)
    arguments = parser.parse_args()
    if arguments.verify_references:
        if arguments.lab_dir is None:
            parser.error("--verify-references requires --lab-dir")
        verification = asyncio.run(verify_references(arguments.output_dir, arguments.lab_dir))
        print(json.dumps(verification, indent=2))
        if verification["status"] != "passed":
            raise SystemExit(1)
    else:
        if arguments.cache_dir is None:
            parser.error("export requires --cache-dir")
        manifest = export_tasks(arguments.output_dir, arguments.cache_dir)
        print(json.dumps({key: manifest[key] for key in ("dataset", "revision", "source_files", "license")}, indent=2))
        print(f"Exported 96 training and 120 independent tasks to {arguments.output_dir}")


if __name__ == "__main__":
    main()
