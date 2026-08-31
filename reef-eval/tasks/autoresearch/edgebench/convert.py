"""Convert published EdgeBench specs into stock Harbor task directories.

EdgeBench (ByteDance-Seed, 51 released tasks) publishes each task as a JSON
spec with a ``work`` half (the agent's world) and a ``judge`` half (the
grader), plus a shared ``BENCHMARK.yaml`` defining base images. Its
two-container design maps one-to-one onto Harbor's separate-verifier mode:

| EdgeBench | Harbor task dir |
|---|---|
| ``work.agent_query`` | ``instruction.md`` |
| work image (prebuilt tag or ``setup_cmds`` on a base image) | ``environment/Dockerfile`` |
| judge image + ``eval_cmd`` | ``tests/Dockerfile`` + ``tests/test.sh`` (separate verifier) |
| ``submit_paths`` | declared ``artifacts`` (restored into the judge workspace before eval) |
| ``internet`` | ``[environment] network_mode`` |
| ``judge.eval_timeout`` | ``[verifier] timeout_sec`` |
| ``rescale`` / ``parser`` / category | ``[metadata]`` (normalization stays query-time) |

The interaction-time budget (EdgeBench evaluates at 2 to 12 h) is a *run*
parameter, not a task property: set ``[agent] timeout_sec`` per run via
``lab.run(..., agent={"override_timeout_sec": ...})`` and tag the row with
``{"budget": h}``; ``metrics.scaling`` reads the curve.

Scoring: the judge's stdout carries either a ``TOTAL_SCORE <n>`` line or a
structured JSON object with a ``score`` field (this mirrors EdgeBench's own
``extract_score`` / ``structured_json`` extraction). The generated
``parse_score.py`` handles both and writes Harbor's ``reward.json``.
``submit_exclude`` patterns are ignored (declared artifacts are copied
whole), and upstream's live judge loop (an auto-eval thread submitting to
a judge server during the session) is not reproduced; see
README.md in this folder for what that changes and the roadmap item that
addresses it.

Spec files: ``https://huggingface.co/datasets/ByteDance-Seed/EdgeBench``
(one ``<task_id>.json`` per task + ``BENCHMARK.yaml``). Prebuilt images are
pulled from the EdgeBench registry (``--registry`` in their docs); tasks
whose spec carries ``setup_cmds`` are built from the base image instead and
need no registry access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

TEMPLATES = Path(__file__).with_name("templates")


def convert_task(
    task_json: Path | str,
    benchmark_yaml: Path | str,
    out_dir: Path | str,
    *,
    registry: str = "seededge",
) -> Path:
    """Convert one EdgeBench spec and return its generated task directory."""
    spec: dict[str, Any] = json.loads(Path(task_json).read_text())
    benchmark: dict[str, Any] = yaml.safe_load(Path(benchmark_yaml).read_text())
    bench_name = benchmark.get("name", "edgebench")
    task_id = spec["task_id"]

    task_dir = Path(out_dir) / task_id
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)

    cwd = spec["cwd"]
    submit_abs = [_join_cwd(cwd, p) for p in spec.get("submit_paths", [])]

    (task_dir / "instruction.md").write_text(spec["work"]["agent_query"])
    (task_dir / "task.toml").write_text(_render_task_toml(spec, submit_abs=submit_abs))
    (task_dir / "environment" / "Dockerfile").write_text(
        _render_env_dockerfile(spec, benchmark, bench_name, registry)
    )
    (task_dir / "tests" / "Dockerfile").write_text(
        _render_judge_dockerfile(spec, benchmark, bench_name, registry)
    )
    (task_dir / "tests" / "test.sh").write_text(
        _render_test_sh(spec, submit_abs=submit_abs)
    )
    (task_dir / "tests" / "parse_score.py").write_text(_template("parse_score.py"))
    (task_dir / "tests" / "test.sh").chmod(0o755)
    return task_dir


# --------------------------------------------------------------- rendering


def _join_cwd(cwd: str, path: str) -> str:
    path = path.rstrip("/")
    return path if path.startswith("/") else f"{cwd.rstrip('/')}/{path}"


def _toml_str(s: str) -> str:
    return json.dumps(s)  # JSON string escaping is valid TOML basic-string


def _template(name: str) -> str:
    return (TEMPLATES / name).read_text()


def _render_task_toml(spec: dict, *, submit_abs: list[str]) -> str:
    judge = spec["judge"]
    network = "public" if spec.get("internet", True) else "no-network"
    artifacts = ", ".join(_toml_str(p) for p in submit_abs)

    metadata = {
        "benchmark": "edgebench",
        "category": spec.get("category", ""),
        "judge_parser": judge.get("parser", ""),
        "score_direction": judge.get("score_direction", "maximize"),
    }
    for key, value in (judge.get("rescale") or {}).items():
        metadata[f"rescale_{key}"] = value
    metadata_lines = "\n".join(
        f"{k} = {json.dumps(v)}" for k, v in metadata.items() if v != ""
    )

    return f"""schema_version = "1.4"

artifacts = [{artifacts}]

[task]
name = {_toml_str(f"edgebench/{spec['task_id']}")}
description = {_toml_str(spec.get("name", spec["task_id"]))}

[metadata]
{metadata_lines}

[agent]
# EdgeBench's official budgets are 2-12 h; the budget is a RUN parameter.
# Override per run: lab.run(..., agent={{"override_timeout_sec": h * 3600}},
# tags={{"budget": h}}).
timeout_sec = 43200.0

[verifier]
timeout_sec = {float(judge["eval_timeout"])}
environment_mode = "separate"

[environment]
build_timeout_sec = 1800.0
network_mode = "{network}"
"""


def _base_image_lines(base_key: str, benchmark: dict) -> list[str]:
    base = benchmark["base_images"][base_key]
    lines = [f"FROM {base['official_image']}"]
    packages = base.get("extra_packages") or []
    if packages:
        lines.append(
            "RUN apt-get update && apt-get install -y "
            + " ".join(packages)
            + " && rm -rf /var/lib/apt/lists/*"
        )
    return lines


def _render_role_dockerfile(
    role: str,
    role_spec: dict,
    spec: dict,
    benchmark: dict,
    bench_name: str,
    registry: str,
) -> list[str]:
    tag = role_spec.get("image_tag")
    if tag:
        # Prebuilt path: EdgeBench publishes hashed work/judge images.
        return [f"FROM {registry}/{bench_name}.{role}.{spec['task_id']}:{tag}"]
    lines = _base_image_lines(spec["base_image"], benchmark)
    for cmd in role_spec.get("setup_cmds") or []:
        lines.append(f"RUN {cmd}")
    return lines


def _render_env_dockerfile(
    spec: dict, benchmark: dict, bench_name: str, registry: str
) -> str:
    lines = _render_role_dockerfile(
        "work", spec["work"], spec, benchmark, bench_name, registry
    )
    lines.append(f"WORKDIR {spec['cwd']}")
    return "\n".join(lines) + "\n"


def _render_judge_dockerfile(
    spec: dict, benchmark: dict, bench_name: str, registry: str
) -> str:
    lines = _render_role_dockerfile(
        "judge", spec["judge"], spec, benchmark, bench_name, registry
    )
    lines += [
        "COPY . /tests",
        "RUN chmod +x /tests/test.sh",
    ]
    return "\n".join(lines) + "\n"


def _render_test_sh(spec: dict, *, submit_abs: list[str]) -> str:
    check_lines = [
        f'[ -e {json.dumps(p)} ] || echo "WARNING: submission missing: {p}"'
        for p in submit_abs
    ]
    return _template("test.sh").format(
        check_lines="\n".join(check_lines),
        cwd=spec["cwd"],
        eval_cmd=spec["judge"]["eval_cmd"],
    )
