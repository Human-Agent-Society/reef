"""Run the original CodeEval-Pro scoring assembly in the isolated verifier."""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import pwd
import secrets
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__:
    from .answer_contract import AnswerContractError, validate_final_module
else:
    from answer_contract import AnswerContractError, validate_final_module


class VerifierFault(RuntimeError):
    """The grading infrastructure or original hidden tests are invalid."""


def assemble_program(answer: str, test_code: str, *, tests_before_candidate_definition: bool = False) -> str:
    """Preserve original checks, with the versioned task41 definition-order correction."""
    tree = ast.parse(test_code)
    if tests_before_candidate_definition:
        definitions_end = max(
            node.end_lineno or node.lineno
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        )
        lines = test_code.splitlines(keepends=True)
        definitions = "".join(lines[:definitions_end])
        assertions = "".join(lines[definitions_end:])
        program = definitions + "\n\n" + answer + "\n\n" + assertions + "\n"
    else:
        program = answer + "\n\n" + test_code + "\n"
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    test_functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    executable = [
        node
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if (
        "TestCases" not in classes
        and not test_functions
        and not any(
            isinstance(node, (ast.Assert, ast.Call)) for statement in executable for node in ast.walk(statement)
        )
    ):
        raise VerifierFault("original hidden tests contain no executable checks")
    direct_calls = {
        node.value.func.id
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
    }
    for function in test_functions:
        if function.args.args or function.args.posonlyargs or function.args.kwonlyargs:
            raise VerifierFault("original test functions require unsupported fixture arguments")
        if function.name not in direct_calls:
            program += f"\n{function.name}()\n"
    if "TestCases" in classes:
        program += (
            "\nif __name__ == '__main__':\n"
            "    import unittest\n"
            "    unittest.TestCase.assertEquals = unittest.TestCase.assertEqual\n"
            "    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCases)\n"
            "    result = unittest.TextTestRunner().run(suite)\n"
            "    if result.testsRun == 0 or not result.wasSuccessful():\n"
            "        raise SystemExit(1)\n"
        )
    return program


def grade(
    task_path: Path, answer_path: Path, timeout_seconds: float, *, isolate_user: bool = True
) -> dict[str, object]:
    """Score a candidate without executing it in the privileged verifier process."""
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if not isinstance(task, dict) or not isinstance(task.get("test_code"), str):
        raise VerifierFault("missing original test code")
    try:
        ast.parse(task["test_code"])
    except SyntaxError as error:
        raise VerifierFault("original hidden test code is invalid") from error
    if not answer_path.is_file():
        return {"reward": 0.0, "status": "invalid", "feedback": "No Python module was submitted."}
    answer = answer_path.read_text(encoding="utf-8")
    try:
        validate_final_module(answer)
    except AnswerContractError as error:
        return {"reward": 0.0, "status": "invalid", "feedback": str(error)}
    program = assemble_program(
        answer,
        task["test_code"],
        tests_before_candidate_definition=task.get("tests_before_candidate_definition", False),
    )
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "MPLBACKEND": "Agg",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    }
    with tempfile.TemporaryDirectory(prefix="agentcl-grade-") as directory:
        root = Path(directory)
        root.chmod(0o755)
        source_path = root / "submission.py"
        source_path.write_text(program, encoding="utf-8")
        source_path.chmod(0o644)
        # A separate wrapper emits completion only after the whole original assembly returns.
        completion_key = secrets.token_hex(32)
        wrapper_path = root / "runner.py"
        wrapper_path.write_text(
            "import builtins, os, pathlib, sys\n"
            "source_path = pathlib.Path(sys.argv[1])\n"
            "completion_key = sys.argv[2].encode()\n"
            "sys.argv = [str(source_path)]\n"
            "execute = builtins.exec\n"
            "write = os.write\n"
            "exit_process = os._exit\n"
            "namespace = {'__name__': '__main__', '__file__': str(source_path)}\n"
            "try:\n"
            "    execute(compile(source_path.read_text(), str(source_path), 'exec'), namespace)\n"
            "except BaseException:\n"
            "    exit_process(1)\n"
            "write(1, completion_key)\n"
            "exit_process(0)\n",
            encoding="utf-8",
        )
        wrapper_path.chmod(0o644)
        options: dict[str, object] = {}
        if isolate_user:
            if os.geteuid() != 0:
                raise VerifierFault("isolated verifier must run as root before dropping grader privileges")
            grader = pwd.getpwnam("grader")
            options = {"user": grader.pw_uid, "group": grader.pw_gid, "extra_groups": []}
            os.chown(root, grader.pw_uid, grader.pw_gid)
        with (
            tempfile.TemporaryFile() as completion_output,
            subprocess.Popen(
                [sys.executable, "-I", str(wrapper_path), str(source_path), completion_key],
                cwd=root,
                env=environment,
                stdout=completion_output,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                **options,
            ) as process,
        ):
            try:
                return_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return {
                    "reward": 0.0,
                    "status": "timeout",
                    "feedback": "The submitted module exceeded the time limit.",
                }
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            completion_output.seek(0, os.SEEK_END)
            output_length = completion_output.tell()
            completion_output.seek(max(0, output_length - len(completion_key)))
            completed = completion_output.read() == completion_key.encode()
    if return_code == 0 and completed:
        return {"reward": 1.0, "status": "passed", "feedback": "The submitted module passed the hidden checks."}
    return {"reward": 0.0, "status": "failed", "feedback": "The submitted module did not pass the hidden checks."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--answer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    result = grade(arguments.task, arguments.answer, arguments.timeout_seconds)
    (arguments.output_dir / "result.json").write_text(json.dumps(result) + "\n", encoding="utf-8")
    (arguments.output_dir / "reward.txt").write_text(str(result["reward"]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
