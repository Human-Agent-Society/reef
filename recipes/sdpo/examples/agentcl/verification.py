"""Recompute local coverage, exact consumption and score summaries independently."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import cast

if __package__:
    from .metrics import build_summary, is_scored_episode
    from .report import JsonObject, read_object, stable_id, terminal_report
else:
    from metrics import build_summary, is_scored_episode
    from report import JsonObject, read_object, stable_id, terminal_report


def verify_run(run_root: Path) -> JsonObject:
    specification = read_object(run_root / "run-manifest.json")
    cursor = read_object(run_root / "cursor.json")
    initial = cast(JsonObject, cursor["initial_release"])
    final = cast(JsonObject, cursor["training_release"])
    records = [read_object(path) for path in sorted((run_root / "episodes").glob("*.json"))]
    commits = [read_object(path) for path in sorted((run_root / "commits").glob("*.json"))]
    failures: list[str] = []
    pending: list[str] = []
    run_id = str(specification["run_id"])
    method = str(specification["method"])
    steps = int(cast(int, specification["steps"]))
    attempts = int(cast(int, specification["attempts_per_training_task"]))
    independent_count = int(cast(int, specification["independent_task_count"]))
    declared_tasks = cast(JsonObject, specification["tasks"])
    selected_positions = cast(list[int], specification["selected_training_positions"])
    data_root = Path(str(specification["data_root"]))
    data_manifest = read_object(data_root / "manifest.json")
    if (
        hashlib.sha256(json.dumps(data_manifest, sort_keys=True).encode()).hexdigest()
        != specification["manifest_sha256"]
    ):
        failures.append("export manifest differs from the immutable run specification")
    expected_counts = {
        "baseline": steps,
        "baseline-independent": independent_count,
        "train": steps * attempts,
        "frozen-repeat": steps,
        "independent": independent_count,
    }
    for phase, expected in expected_counts.items():
        rows = [row for row in records if row["phase"] == phase]
        role = "independent" if phase in ("independent", "baseline-independent") else "training"
        declared = cast(list[JsonObject], declared_tasks[role])
        if role == "training":
            declared = [row for row in declared if row["position"] in selected_positions]
            declared.sort(key=lambda row: selected_positions.index(int(cast(int, row["position"]))))
        else:
            declared = declared[:independent_count]
        expected_identities = {
            (row["category"], row["id"], attempt, index, row["position"])
            for index, row in enumerate(declared)
            for attempt in range(attempts if phase == "train" else 1)
        }
        actual_identities = {
            (row["category"], row["task_id"], row["attempt"], row["campaign_position"], row["position"])
            for row in rows
        }
        if actual_identities != expected_identities:
            failures.append(f"{phase}: task identities or supplied order differ from run manifest")
        if len(rows) != expected:
            failures.append(f"{phase}: {len(rows)} episodes, expected {expected}")
        if len({(row["category"], row["task_id"], row["attempt"]) for row in rows}) != len(rows):
            failures.append(f"{phase}: repeated task/attempt identities")
    for record in records:
        phase = str(record["phase"])
        episode_id = stable_id(
            run_id,
            phase,
            str(record["category"]),
            str(record["task_id"]),
            int(cast(int, record["attempt"])),
            "episode",
        )
        if record["episode_id"] != episode_id:
            failures.append("episode UUID does not match its declared run/task/attempt")
        if not is_scored_episode(record):
            failures.append(f"{episode_id}: fault or truncation")
        score = record.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or score not in (0, 1)
        ):
            failures.append(f"{episode_id}: missing or invalid binary verifier score")
        references = cast(list[str], record["references"])
        turns = cast(list[JsonObject], record["turns"])
        if (
            references != [turn.get("receipt") for turn in turns]
            or not references
            or len(set(references)) != len(references)
        ):
            failures.append(f"{episode_id}: missing or repeated turn receipts")
        if any(turn.get("release_id") != record["release_id"] for turn in turns):
            failures.append(f"{episode_id}: mixed or missing turn releases")
        if phase in ("baseline", "baseline-independent", "frozen-repeat", "independent"):
            sampled_release = initial if phase.startswith("baseline") else final
            phase_state = cast(JsonObject, cast(JsonObject, cursor["phases"]).get(phase, {}))
            runtime_bindings = cast(JsonObject, cursor.get("evaluation_runtime_load_ids", {}))
            runtime_load_id = (
                sampled_release.get("runtime_load_id")
                or runtime_bindings.get(str(sampled_release["release_id"]))
                or phase_state.get("runtime_load_id")
            )
            if (
                record["outcome"] == "truncated"
                or "runtime_load_id" in record
                or any("runtime_load_id" in turn for turn in turns)
            ) and (
                not isinstance(runtime_load_id, str)
                or not runtime_load_id
                or record.get("runtime_load_id") != runtime_load_id
                or any(turn.get("runtime_load_id") != runtime_load_id for turn in turns)
            ):
                failures.append(f"{episode_id}: mixed or missing turn runtime loads")
        report_path = run_root / "reports" / f"{record['report_id']}.json"
        if phase == "train":
            if not report_path.exists():
                failures.append(f"{episode_id}: missing terminal report")
            else:
                report_id = stable_id(
                    run_id,
                    phase,
                    str(record["category"]),
                    str(record["task_id"]),
                    int(cast(int, record["attempt"])),
                    "report",
                )
                if record["report_id"] != report_id:
                    failures.append(f"{episode_id}: terminal report UUID differs from the declared episode")
                context = ""
                if method == "sdft":
                    tasks = cast(list[JsonObject], data_manifest["training"])
                    task = next(
                        (
                            row
                            for row in tasks
                            if row["category"] == record["category"]
                            and row["id"] == record["task_id"]
                            and row["position"] == record["position"]
                        ),
                        None,
                    )
                    if task is None:
                        failures.append(f"{episode_id}: demonstration task is absent from the manifest")
                    else:
                        demonstration = data_root / str(task["reference_path"])
                        if (
                            not demonstration.is_file()
                            or hashlib.sha256(demonstration.read_bytes()).hexdigest() != task["reference_sha256"]
                        ):
                            failures.append(f"{episode_id}: verified demonstration bytes differ")
                        else:
                            context = demonstration.read_text()
                elif method == "sdpo":
                    context = (
                        "The submitted solution passed the verifier."
                        if record["score"] == 1
                        else "The submitted solution did not pass the verifier."
                    )
                try:
                    expected_report = terminal_report(
                        method, record, context, int(cast(int, record["campaign_position"])) + 1
                    )
                except ValueError as error:
                    failures.append(f"{episode_id}: invalid terminal report source: {error}")
                else:
                    if read_object(report_path) != expected_report:
                        failures.append(
                            f"{episode_id}: terminal report differs from the canonical episode and teacher context"
                        )
        elif report_path.exists():
            failures.append(f"{episode_id}: evaluation produced a training report")
    expected_reports = {str(row["report_id"]) for row in records if row["phase"] == "train"}
    actual_reports = {path.stem for path in (run_root / "reports").glob("*.json")}
    if actual_reports != expected_reports:
        failures.append("unexpected or missing report artifacts; evaluation must produce zero reports")
    for phase in ("baseline", "baseline-independent", "frozen-repeat", "independent"):
        required = initial["release_id"] if phase.startswith("baseline") else final["release_id"]
        if any(row["release_id"] != required for row in records if row["phase"] == phase):
            failures.append(f"{phase}: model release was not frozen")
    if len(commits) != steps or cursor["training_step"] != steps:
        failures.append("missing expected committed training updates")
    prior = initial["release_id"]
    checkpoint_parent = initial["release_id"]
    for position, commit in enumerate(commits):
        if (
            commit["step"] != position + 1
            or commit["operation"] != "training"
            or commit["operation_verified"] is not True
            or commit["pending"]
        ):
            failures.append(f"step {position + 1}: invalid or pending training commit")
        artifact = cast(JsonObject, commit["artifact_ref"])
        if artifact["parent_release_id"] != checkpoint_parent or artifact["release_id"] == prior:
            failures.append(f"step {position + 1}: broken publication chain")
        sampled_release = prior
        prior = artifact["release_id"]
        if artifact["kind"] != "live_weights":
            checkpoint_parent = artifact["release_id"]
        rows = [row for row in records if row["phase"] == "train" and row["campaign_position"] == position]
        if len(rows) != attempts or any(row["release_id"] != sampled_release for row in rows):
            failures.append(f"step {position + 1}: training grid was not sampled from its prior committed release")
        expected = {str(row["report_id"]) for row in rows}
        expected.update(reference for row in rows for reference in cast(list[str], row["references"]))
        if set(cast(list[str], commit["consumed_ids"])) != expected:
            failures.append(f"step {position + 1}: complete-grid consumption mismatch")
        metrics = commit.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            pending.append(f"step {position + 1}: optimizer metrics unavailable")
        elif any(isinstance(value, (int, float)) and not math.isfinite(value) for value in metrics.values()):
            failures.append(f"step {position + 1}: non-finite optimizer metric")
    if prior != final["release_id"]:
        failures.append("cursor final release differs from final commit")
    records.sort(
        key=lambda record: (
            str(record["phase"]),
            int(cast(int, record["campaign_position"])),
            int(cast(int, record["attempt"])),
        )
    )
    summary = build_summary(records)
    if (run_root / "summary.json").exists() and read_object(run_root / "summary.json") != summary:
        failures.append("stored summary differs from independently recomputed scores")
    # These claims require external artifacts, not inference from a release name.
    qualification_path = run_root / "qualification.json"
    required_checks = (
        "real_gpu_training",
        "finite_losses_and_gradients",
        "nonzero_gradient_weight_change",
        "all_assistant_tokens_selected",
        "tool_context_zero_loss",
        "teacher_suffix_identity",
        "final_export_reload",
        "owned_cleanup_gpu_drain",
        "hidden_assets_not_exposed",
        "reference_solutions_verified",
        "trace_browser_verified",
        "authenticated_toolbox_bytes_verified",
    )
    if qualification_path.exists():
        qualification = read_object(qualification_path)
        for key in required_checks:
            check = qualification.get(key)
            if (
                not isinstance(check, dict)
                or check.get("passed") is not True
                or not isinstance(check.get("artifact"), str)
            ):
                pending.append(f"qualification required: {key}")
            elif not (run_root / str(check["artifact"])).is_file():
                pending.append(f"qualification artifact missing: {key}")
    else:
        pending.extend(f"qualification required: {key}" for key in required_checks)
    pending.append("external qualification artifacts require independent lead review; declarations are not validation")
    return {
        "method": method,
        "profile": specification["profile"],
        "complete": not failures and not pending and specification["profile"] == "full",
        "local_checks_passed": not failures,
        "failures": cast(list, failures),
        "unverified": cast(list, pending),
        "summary": summary,
    }
