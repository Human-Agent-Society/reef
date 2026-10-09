"""Ordered AgentCL tasks through Harbor and native Reef commits, with fail-closed resume."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shlex
import sys
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

if __package__:
    from .harness import answer_contract
    from .metrics import build_summary, is_scored_episode
    from .qualification import check_native_sample, qualify_inputs
    from .report import JsonObject, read_object, stable_id, terminal_report, write_object
else:
    from metrics import build_summary, is_scored_episode
    from qualification import check_native_sample, qualify_inputs
    from report import JsonObject, read_object, stable_id, terminal_report, write_object

    from harness import answer_contract

METHOD = "opd"
HERE = Path(__file__).resolve().parent
DATASET_REVISION = "a01e2ca6e33fd07d9cf80e4bd69a5b3585d3400f"
BASE_COMMIT = "daeaf4d42fb2f9e20445df6618afdc2a4097bdeb"
OPD_IMPLEMENTATION_BRANCH = "origin/codex/opd-slime-reproduction"
OPD_IMPLEMENTATION_REVISION = "a06415d7c417dc0f180b3301e64d256aa91dcb40"
OPD_TRAINING_OPTIONS = {
    "opd-teacher": "separate",
    "opd-divergence": "reverse",
    "opd-top-k": "1",
    "opd-skip-response-tokens": "0",
    "opd-importance-sampling-cap": "0.0",
    "opd-importance-sampling-level": "token",
    "opd-teacher-update-rate": "0.0",
}


class ReefApi(ABC):
    """The receipt/commit contract used by the resumable driver."""

    @abstractmethod
    def current_release(self) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def commits(self, record_ids: list[str]) -> list[JsonObject]:
        raise NotImplementedError

    @abstractmethod
    def record(self, record_id: str) -> JsonObject | None:
        raise NotImplementedError

    @abstractmethod
    def report(self, payload: JsonObject) -> None:
        raise NotImplementedError


class HttpReefApi(ReefApi):
    def __init__(self, service_url: str, scenario: str, token: str | None, timeout_seconds: float = 60):
        self.service_url = service_url.rstrip("/")
        self.scenario = scenario
        self.timeout_seconds = timeout_seconds
        self.headers = {"Content-Type": "application/json", "x-reef-scenario": scenario}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.scenario_path = f"/reef/scenarios/{quote(scenario, safe='')}"

    def request(self, path: str, payload: JsonObject | None = None) -> JsonObject:
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = Request(self.service_url + path, data=body, headers=self.headers)
        with urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise ValueError("Reef response must be a JSON object")
        return cast(JsonObject, value)

    def ensure_scenario(self) -> None:
        """Create this named scenario only when it is absent; never reset existing state."""
        try:
            self.current_release()
        except HTTPError as error:
            if error.code != 404:
                raise
            self.request("/reef/scenarios", {"name": self.scenario})

    def current_release(self) -> JsonObject:
        releases = cast(list[JsonObject], self.request(self.scenario_path + "/releases")["releases"])
        current = [release for release in releases if release["current"] is True]
        if len(current) != 1 or current[0].get("pending"):
            raise RuntimeError("scenario must expose exactly one non-pending current release")
        return current[0]

    def commits(self, record_ids: list[str]) -> list[JsonObject]:
        commits: list[JsonObject] = []
        after_step = 0
        while True:
            query = urlencode(
                [("after_step", after_step), ("limit", 100), *(("record_id", value) for value in record_ids)]
            )
            page = self.request(self.scenario_path + "/commits?" + query)
            commits.extend(cast(list[JsonObject], page["commits"]))
            next_step = page["next_after_step"]
            if next_step is None:
                return commits
            after_step = int(cast(int, next_step))

    def record(self, record_id: str) -> JsonObject | None:
        try:
            return self.request(self.scenario_path + "/records/" + quote(record_id, safe=""))
        except HTTPError as error:
            if error.code == 404:
                return None
            raise

    def report(self, payload: JsonObject) -> None:
        result = self.request("/reef/report", payload)
        if result["agent_record_id"] != payload["agent_record_id"]:
            raise RuntimeError("Reef did not retain the deterministic report ID")


class EpisodeBackend(ABC):
    """An isolated sandbox episode and its trusted verifier result."""

    @abstractmethod
    async def run(self, task: JsonObject, episode_id: str, phase: str, release: JsonObject) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def recover(self, episode_id: str) -> JsonObject | None:
        raise NotImplementedError


class HarborBackend(EpisodeBackend):
    def __init__(self, arguments: argparse.Namespace):
        from reef_eval import Lab

        self.arguments = arguments
        self.lab = Lab(arguments.run_root / "lab")

    def recover(self, episode_id: str) -> JsonObject | None:
        row = self.lab.store.get(episode_id)
        path = self.arguments.run_root / "harbor-episodes" / f"{episode_id}.json"
        if row is None or not path.exists():
            return None
        result = read_object(path)
        if row.tags.get("error") or "reward" not in row.rewards:
            result.update(outcome="fault", fault="missing verifier reward or Harbor infrastructure error", score=None)
        else:
            score = float(row.rewards["reward"])
            if score not in (0.0, 1.0):
                raise ValueError("AgentCL requires a finite binary verifier reward")
            result["score"] = score
        return result

    async def run(self, task: JsonObject, episode_id: str, phase: str, release: JsonObject) -> JsonObject:
        output = self.arguments.run_root / "harbor-episodes" / f"{episode_id}.json"
        kwargs: JsonObject = {
            "service_url": self.arguments.service_url,
            "scenario": self.arguments.scenario,
            "episode_id": episode_id,
            "episode_output": str(output),
            "expected_release": release["release_id"],
            "expected_runtime_load_id": release.get("runtime_load_id"),
            "phase": "baseline" if phase == "baseline-independent" else phase,
            "max_turns": self.arguments.max_turns,
            "max_response_tokens": 2048,
            "max_episode_tokens": 8192,
            "tool_timeout_seconds": 30,
            "max_tool_output_chars": 8000,
            "temperature": 0.7,
            "seed": task["sampling_seed"],
        }
        # The harness reads credentials from the host environment, not persisted agent kwargs.
        await self.lab.run(
            str(self.arguments.data_root / str(task["task_path"])),
            {"name": "harness:HarborAgent", "model_name": "reef", "kwargs": kwargs},
            key=episode_id,
            environment={"import_path": "harness.detached:DetachedNoNetworkEnvironment"},
            tags={"phase": phase, "category": task["category"], "task_id": task["id"]},
        )
        result = self.recover(episode_id)
        if result is None:
            raise RuntimeError("Harbor returned without both episode.json and a durable verifier row")
        return result


def matched_commit(
    api: ReefApi,
    report_ids: list[str],
    references: list[str],
    base_release: str,
    expected_step: int,
    checkpoint_parent: str | None = None,
) -> JsonObject | None:
    """One exact native training commit, not merely a newer release."""
    commits = api.commits(report_ids)
    if not commits:
        return None
    if len(commits) != 1:
        raise RuntimeError("task reports were consumed by more than one commit")
    commit = commits[0]
    if commit["operation"] != "training" or commit["operation_verified"] is not True:
        raise RuntimeError("task consumption is not a verified training operation")
    if set(cast(list[str], commit["consumed_ids"])) != set(report_ids + references):
        raise RuntimeError("training commit did not consume exactly the complete task grid and all receipts")
    if commit["step"] != expected_step:
        raise RuntimeError("unexpected scenario step; another writer may own this scenario")
    artifact = cast(JsonObject, commit["artifact_ref"])
    expected_parent = base_release if checkpoint_parent is None else checkpoint_parent
    if artifact["release_id"] == base_release or artifact["parent_release_id"] != expected_parent:
        raise RuntimeError("training commit must publish a new release anchored to the durable checkpoint")
    if commit["pending"] is True:
        return None
    current = api.current_release()
    if current["release_id"] != artifact["release_id"]:
        raise RuntimeError("committed task release is not the currently served release")
    return commit


def read_reviewed_episode(
    run_root: Path, identifier: str, expected: JsonObject, release: JsonObject, runtime_load_id: str | None
) -> JsonObject | None:
    """Read parent-reviewed results without replaying inference or changing archived originals."""
    root = run_root.resolve()

    def _path(relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError("recovery artifact must stay inside the run root")
        candidate = root / path
        if any((root / Path(*path.parts[:index])).is_symlink() for index in range(1, len(path.parts) + 1)):
            raise RuntimeError("recovery artifact must not use symlinks")
        if not candidate.resolve(strict=True).is_relative_to(root) or not candidate.is_file():
            raise RuntimeError("recovery artifact must be a file inside the run root")
        return candidate

    def _artifact(descriptor: JsonObject) -> Path:
        artifact = _path(str(descriptor["artifact"]))
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != descriptor["sha256"]:
            raise RuntimeError("recovery artifact checksum changed")
        return artifact

    relative = f"episode-recoveries/{identifier}.json"
    if (root / "episode-recoveries").is_symlink():
        raise RuntimeError("recovery artifact must not use symlinks")
    if not (root / relative).exists() and not (root / relative).is_symlink():
        return None
    record = read_object(_path(relative))
    kind = record.get("recovery_kind")
    if record.get("reviewed") is not True or kind not in (
        "verifier_only_saved_final",
        "confirmed_zero_call_retry",
        "verifier_only_saved_episode",
    ):
        raise RuntimeError("episode recovery needs explicit parent review and a supported recovery kind")
    original_path = _artifact({"artifact": record["original_artifact"], "sha256": record["original_sha256"]})
    if not original_path.is_relative_to(root / "recovery-originals"):
        raise RuntimeError("original recovery artifact must be archived under recovery-originals")
    original = read_object(original_path)
    result = cast(JsonObject, record["recovered_episode"])
    if not isinstance(result, dict) or any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError("recovered episode task or logical identity changed")
    if (
        record.get("sampled_release") != release
        or original.get("episode_id") != identifier
        or (original.get("release_id") != release["release_id"] or original.get("outcome") != "fault")
    ):
        raise RuntimeError("original episode fault or sampled release changed")
    references = result.get("references")
    turns = result.get("turns")
    if (
        not isinstance(references, list)
        or not references
        or any(not isinstance(value, str) or not value for value in references)
    ):
        raise RuntimeError("recovery needs ordered unique turn receipts")
    if (
        not isinstance(turns, list)
        or any(not isinstance(turn, dict) for turn in turns)
        or (
            references != [turn.get("receipt") for turn in cast(list[JsonObject], turns)]
            or len(set(references)) != len(references)
        )
    ):
        raise RuntimeError("recovery needs ordered unique turn receipts")
    if (
        not isinstance(runtime_load_id, str)
        or not runtime_load_id
        or result.get("runtime_load_id") != runtime_load_id
        or any(
            turn.get("release_id") != release["release_id"] or turn.get("runtime_load_id") != runtime_load_id
            for turn in cast(list[JsonObject], turns)
        )
    ):
        raise RuntimeError("recovered episode release or native runtime binding changed")
    if (
        result.get("outcome") not in ("completed", "truncated")
        or (result.get("outcome") == "truncated" and not is_scored_episode(result))
        or type(result.get("score")) not in (int, float)
        or result["score"] not in (0, 1)
    ):
        raise RuntimeError("recovery must retain a known binary verifier score")
    if kind == "verifier_only_saved_episode":
        if expected.get("phase") not in ("baseline", "baseline-independent", "frozen-repeat", "independent"):
            raise RuntimeError("saved episode recovery is evaluation-only")
        if original.get("score") is not None or original.get("fault") != (
            "missing verifier reward or Harbor infrastructure error"
        ):
            raise RuntimeError("saved episode recovery requires the original Harbor infrastructure fault")
        snapshot_path = _artifact(cast(JsonObject, record["original_snapshot"]))
        if not snapshot_path.is_relative_to(root / "recovery-originals") or snapshot_path == original_path:
            raise RuntimeError("original snapshot must be separately archived under recovery-originals")
        snapshot = read_object(snapshot_path)
        # Cleanup may replace outcomes, never inference records or generated source bytes.
        mutable = {"outcome", "fault", "score", "metadata"}
        original_immutable = {key: value for key, value in original.items() if key not in mutable}
        if any(
            {key: value for key, value in episode.items() if key not in mutable} != original_immutable
            for episode in (snapshot, result)
        ):
            raise RuntimeError("saved episode recovery must preserve every original immutable field")
        for previous, following in ((original, snapshot), (original, result), (snapshot, result)):
            previous_metadata = previous.get("metadata", {})
            following_metadata = following.get("metadata", {})
            if (
                not isinstance(previous_metadata, dict)
                or not isinstance(following_metadata, dict)
                or any(following_metadata.get(key) != value for key, value in previous_metadata.items())
            ):
                raise RuntimeError("saved episode recovery permits only metadata additions")
        if (
            snapshot.get("outcome") not in ("completed", "truncated")
            or result.get("outcome") != snapshot.get("outcome")
            or result.get("fault") != snapshot.get("fault")
            or not is_scored_episode({**snapshot, "score": 0})
        ):
            raise RuntimeError("saved episode recovery must preserve the original completion or bounded truncation")
        proof = read_object(_artifact(cast(JsonObject, record["verifier_result"])))
        physical_key = record.get("harbor_key")
        rewards = proof.get("rewards")
        tags = proof.get("tags")
        if (
            not isinstance(physical_key, str)
            or not physical_key
            or physical_key == identifier
            or proof.get("key") != physical_key
            or not isinstance(rewards, dict)
            or type(rewards.get("reward")) not in (int, float)
            or rewards.get("reward") != result["score"]
            or not isinstance(tags, dict)
            or tags.get("error")
            or tags.get("episode_id") != identifier
            or tags.get("recovery_kind") != kind
            or tags.get("parent_checked") is not True
            or tags.get("fresh_verifier") is not True
        ):
            raise RuntimeError("saved episode recovery needs matching fresh parent-checked binary verifier proof")
        if snapshot["outcome"] == "completed":
            messages = original.get("messages")
            if not isinstance(messages, list):
                raise RuntimeError("saved completed episode requires the original assistant FINAL")
            assistants = [
                message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"
            ]
            text = assistants[-1].get("content") if assistants else None
            if not isinstance(text, str) or not text.lstrip().startswith("FINAL"):
                raise RuntimeError("saved completed episode requires the original assistant FINAL")
            blocks = re.findall(r"```python\s*\n(.*?)```", text, flags=re.DOTALL)
            if len(blocks) != 1:
                raise RuntimeError("saved completed FINAL requires exactly one Python code block")
            try:
                answer_contract.validate_final_module(blocks[0])
            except answer_contract.AnswerContractError as error:
                raise RuntimeError("saved completed FINAL violates the answer contract") from error
            answer = _artifact(cast(JsonObject, record["answer_blob"]))
            if answer.read_bytes() != blocks[0].encode("utf-8"):
                raise RuntimeError("saved FINAL answer bytes differ from the original model output")
        else:
            if result["score"] != 0:
                raise RuntimeError("saved truncated episode cannot gain a successful score")
            answer_descriptor = record.get("answer_blob")
            if answer_descriptor is not None and read_object(_artifact(cast(JsonObject, answer_descriptor))) != {
                "answer_present": False,
                "reason": "original_episode_did_not_submit",
            }:
                raise RuntimeError("saved truncated episode must not fabricate a final answer")
    elif kind == "verifier_only_saved_final":
        _artifact(cast(JsonObject, record["verifier_result"]))
        _artifact(cast(JsonObject, record["answer_blob"]))
        immutable = (
            "references",
            "turns",
            "messages",
            "trajectory",
            "runtime_load_id",
            "prompt_tokens",
            "completion_tokens",
        )
        if (
            result.get("outcome") != "completed"
            or record.get("terminal_final_validated") is not True
            or any(result.get(key) != original.get(key) for key in immutable)
        ):
            raise RuntimeError("saved FINAL recovery must preserve all original inference and ATIF records")
    else:
        _artifact(cast(JsonObject, record["new_trial_row"]))
        if (
            original.get("references") != []
            or original.get("turns") != []
            or any(original.get(key) != 0 for key in ("prompt_tokens", "completion_tokens"))
        ):
            raise RuntimeError("retry recovery requires a confirmed zero-call original fault")
        if (
            not isinstance(record.get("harbor_key"), str)
            or not record["harbor_key"]
            or record["harbor_key"] == identifier
        ):
            raise RuntimeError("retry recovery needs a new physical Harbor key")
    if (root / "episodes").is_symlink():
        raise RuntimeError("recovery artifact must not use symlinks")
    canonical = root / "episodes" / f"{identifier}.json"
    if canonical.exists() or canonical.is_symlink():
        canonical = _path(f"episodes/{identifier}.json")
        if (
            read_object(canonical) != result
            and hashlib.sha256(canonical.read_bytes()).hexdigest() != record["original_sha256"]
        ):
            raise RuntimeError("existing canonical episode differs from the reviewed original or recovery")
    return result


class Campaign:
    """Persist local intent before remote work; reconcile uncertain results without replay."""

    def __init__(self, arguments: argparse.Namespace, api: ReefApi, backend: EpisodeBackend, manifest: JsonObject):
        self.arguments = arguments
        self.api = api
        self.backend = backend
        self.manifest = manifest
        if not isinstance(arguments.teacher_checkpoint, str) or not arguments.teacher_checkpoint.strip():
            raise ValueError("OPD campaigns require an explicit frozen teacher checkpoint")
        self.cursor_path = arguments.run_root / "cursor.json"
        self.spec_path = arguments.run_root / "run-manifest.json"
        specification = run_specification(arguments, manifest)
        if self.spec_path.exists():
            if read_object(self.spec_path) != specification:
                raise RuntimeError("run settings or data changed; use a fresh run root")
        else:
            write_object(self.spec_path, specification)
        if self.cursor_path.exists():
            self.cursor = read_object(self.cursor_path)
        else:
            current = api.current_release()
            if current["operation"] != "creation":
                raise RuntimeError("a new campaign must start on a fresh base-model scenario")
            self.cursor = {
                "initial_release": current,
                "training_release": current,
                "training_step": 0,
                "phases": {},
            }
            self.save()

    def save(self) -> None:
        write_object(self.cursor_path, self.cursor)

    async def episode(
        self, task: JsonObject, phase: str, attempt: int, release: JsonObject, task_state: JsonObject
    ) -> JsonObject:
        identifier = stable_id(
            self.arguments.run_id, phase, str(task["category"]), str(task["id"]), attempt, "episode"
        )
        path = self.arguments.run_root / "episodes" / f"{identifier}.json"
        expected: JsonObject = {
            "episode_id": identifier,
            "phase": phase,
            "category": task["category"],
            "task_id": task["id"],
            "position": task["position"],
            "campaign_position": task["campaign_position"],
            "attempt": attempt,
            "release_id": release["release_id"],
            "report_id": stable_id(
                self.arguments.run_id, phase, str(task["category"]), str(task["id"]), attempt, "report"
            ),
        }
        runtime_bindings = cast(JsonObject, self.cursor.get("evaluation_runtime_load_ids", {}))
        runtime_load_id = cast(
            str | None, release.get("runtime_load_id") or runtime_bindings.get(str(release["release_id"]))
        )
        recovered = read_reviewed_episode(self.arguments.run_root, identifier, expected, release, runtime_load_id)
        if recovered is not None:
            if not path.exists():
                write_object(path, recovered)
            return recovered
        if path.exists():
            existing = read_object(path)
            if existing["episode_id"] != identifier or existing["release_id"] != release["release_id"]:
                raise RuntimeError("persisted episode identity or sampled release changed")
            return existing
        started = cast(list[str], task_state.setdefault("started_episodes", []))
        if identifier in started:
            result = self.backend.recover(identifier)
            if result is None:
                raise RuntimeError(
                    f"unknown outcome for episode {identifier}; reconcile its Harbor result, never replay inference"
                )
        else:
            started.append(identifier)
            self.save()
            seed_identity = stable_id("sampling", "all", str(task["category"]), str(task["id"]), attempt, "seed")
            sampling_seed = (self.arguments.seed + int(seed_identity.replace("-", "")[:8], 16)) % (2**31)
            result = await self.backend.run({**task, "sampling_seed": sampling_seed}, identifier, phase, release)
        if result["episode_id"] != identifier or result["release_id"] != release["release_id"]:
            raise RuntimeError("episode backend returned a different identity or release")
        turns = cast(list[JsonObject], result["turns"])
        references = cast(list[str], result["references"])
        if (
            references != [turn["receipt"] for turn in turns]
            or not references
            or len(set(references)) != len(references)
        ):
            raise RuntimeError("episode must retain every ordered unique receipt")
        if any(turn["release_id"] != release["release_id"] for turn in turns):
            raise RuntimeError("episode turns were not sampled from one pinned release")
        result.update(
            episode_id=identifier,
            phase=phase,
            category=task["category"],
            task_id=task["id"],
            position=task["position"],
            campaign_position=task["campaign_position"],
            attempt=attempt,
            release_id=release["release_id"],
            report_id=stable_id(
                self.arguments.run_id, phase, str(task["category"]), str(task["id"]), attempt, "report"
            ),
        )
        write_object(path, result)
        return result

    async def execute_phase(self, phase: str) -> None:
        training = phase == "train"
        if training:
            validate_reference_verification(self.arguments.data_root, self.manifest)
        if phase in ("frozen-repeat", "independent"):
            if self.cursor["training_step"] != self.arguments.steps:
                raise RuntimeError("frozen evaluation requires all configured training commits first")
            release = cast(JsonObject, self.cursor["training_release"])
        elif training:
            release = cast(JsonObject, self.cursor["training_release"])
        else:
            release = cast(JsonObject, self.cursor["initial_release"])
        phases = cast(JsonObject, self.cursor["phases"])
        phase_state = cast(
            JsonObject, phases.setdefault(phase, {"tasks": {}, "complete": False, "release_id": release["release_id"]})
        )
        tasks = cast(
            list[JsonObject],
            self.manifest["independent" if phase in ("independent", "baseline-independent") else "training"],
        )
        if self.arguments.profile == "smoke":
            if phase in ("independent", "baseline-independent"):
                tasks = tasks[:2]
            else:
                first = tasks[0]
                complex_task = next(task for task in tasks[48:] if task["pair_id"] == first["pair_id"])
                tasks = [first, complex_task]
        if phase in ("baseline", "baseline-independent", "frozen-repeat", "independent"):
            await self.execute_evaluation_phase(phase, tasks, release, phase_state)
            return
        states = cast(JsonObject, phase_state["tasks"])
        for campaign_position, task in enumerate(tasks):
            if training and self.arguments.native_input_checks and campaign_position > 0:
                first_reports = [read_object(path) for path in (self.arguments.run_root / "episodes").glob("*.json")]
                first_episodes = [
                    row for row in first_reports if row["phase"] == "train" and row["campaign_position"] == 0
                ]
                if len(first_episodes) != self.arguments.attempts:
                    raise RuntimeError("native first-task qualification requires the complete sampling grid")
                first_checks = []
                for episode in first_episodes:
                    capture = read_object(self.arguments.run_root / "teacher-records" / f"{episode['report_id']}.json")
                    first_checks.append(check_native_sample(capture, episode, require_multi_turn=False))
                if not any(check["active"] for check in first_checks):
                    raise RuntimeError("first native grid has no effective distillation signal")
                write_object(
                    self.arguments.run_root / "native-first-task-checks.json",
                    {"passed": True, "checked_reports": [row["report_id"] for row in first_episodes]},
                )
            key = json.dumps([task["category"], task["id"]], separators=(",", ":"))
            state = cast(
                JsonObject, states.setdefault(key, {"started_episodes": [], "report_attempted": [], "complete": False})
            )
            if state["complete"] is True:
                continue
            release = cast(JsonObject, self.cursor["training_release"]) if training else release
            sampled_release = cast(JsonObject, state.setdefault("sampled_release", release))
            if not training and self.api.current_release()["release_id"] != release["release_id"]:
                raise RuntimeError("frozen evaluation release changed before sampling")
            task = {**task, "campaign_position": campaign_position}
            episodes = [
                await self.episode(task, phase, attempt, sampled_release, state)
                for attempt in range(self.arguments.attempts if training else 1)
            ]
            if any(record["outcome"] != "completed" or record.get("score") is None for record in episodes):
                raise RuntimeError("task fault or truncation; recorded scores cannot establish campaign completion")
            if training:
                await self.commit_task(task, episodes, sampled_release, state)
            else:
                if self.api.current_release()["release_id"] != release["release_id"]:
                    raise RuntimeError("frozen evaluation weights changed")
                state["complete"] = True
                self.save()
        phase_state["complete"] = True
        self.save()

    async def execute_evaluation_phase(
        self, phase: str, tasks: list[JsonObject], release: JsonObject, phase_state: JsonObject
    ) -> None:
        release = deepcopy(release)
        if phase_state["release_id"] != release["release_id"]:
            raise RuntimeError("frozen evaluation phase release changed")
        sampled_release = cast(JsonObject, phase_state.setdefault("sampled_release", deepcopy(release)))
        if sampled_release != release:
            raise RuntimeError("frozen evaluation sampled release changed")
        states = cast(JsonObject, phase_state["tasks"])
        ordered_states: JsonObject = {}
        pending: list[tuple[JsonObject, JsonObject]] = []
        for campaign_position, task in enumerate(tasks):
            key = json.dumps([task["category"], task["id"]], separators=(",", ":"))
            state = cast(
                JsonObject, states.setdefault(key, {"started_episodes": [], "report_attempted": [], "complete": False})
            )
            state["campaign_position"] = campaign_position
            if state.setdefault("sampled_release", deepcopy(release)) != release:
                raise RuntimeError("frozen evaluation task sampled release changed")
            ordered_states[key] = state
            if state["complete"] is not True:
                pending.append(({**task, "campaign_position": campaign_position}, state))
        phase_state["tasks"] = ordered_states
        if pending:
            phase_state["complete"] = False
        self.save()

        def _check_release() -> None:
            current = self.api.current_release()
            if (
                current["release_id"] != release["release_id"]
                or current.get("runtime_load_id") != release.get("runtime_load_id")
                or current.get("pending")
            ):
                raise RuntimeError("frozen evaluation release changed")

        async def _evaluate(task: JsonObject, state: JsonObject) -> None:
            _check_release()
            try:
                episode = await self.episode(task, phase, 0, deepcopy(release), state)
            finally:
                _check_release()
            if (
                episode["release_id"] != release["release_id"]
                or any(episode[key] != task[key] for key in ("category", "position", "campaign_position"))
                or episode["task_id"] != task["id"]
                or episode["phase"] != phase
                or episode["attempt"] != 0
            ):
                raise RuntimeError("frozen evaluation episode identity or sampled release changed")
            turns = cast(list[JsonObject], episode["turns"])
            references = cast(list[str], episode["references"])
            if (
                not references
                or references != [turn["receipt"] for turn in turns]
                or len(set(references)) != len(references)
                or any(turn["release_id"] != release["release_id"] for turn in turns)
            ):
                raise RuntimeError("frozen evaluation episode receipts or sampled release changed")
            runtime_bindings = cast(JsonObject, self.cursor.setdefault("evaluation_runtime_load_ids", {}))
            runtime_load_id = (
                release.get("runtime_load_id")
                or runtime_bindings.get(str(release["release_id"]))
                or phase_state.get("runtime_load_id")
            )
            if runtime_load_id is None and (self.arguments.native_input_checks or "runtime_load_id" in episode):
                runtime_load_id = episode.get("runtime_load_id")
                if not isinstance(runtime_load_id, str) or not runtime_load_id:
                    raise RuntimeError("frozen evaluation needs a receipt-derived runtime load ID")
                runtime_bindings[str(release["release_id"])] = runtime_load_id
                phase_state["runtime_load_id"] = runtime_load_id
                self.save()
            if (
                episode["outcome"] == "truncated"
                or "runtime_load_id" in episode
                or any("runtime_load_id" in turn for turn in turns)
            ) and (
                not isinstance(runtime_load_id, str)
                or not runtime_load_id
                or episode.get("runtime_load_id") != runtime_load_id
                or any(turn.get("runtime_load_id") != runtime_load_id for turn in turns)
            ):
                raise RuntimeError("frozen evaluation episode runtime load changed or is missing")
            if not is_scored_episode(episode):
                raise RuntimeError(
                    "task fault or truncation without a valid score; cannot establish campaign completion"
                )
            state["complete"] = True
            self.save()

        _check_release()
        missing: list[tuple[JsonObject, JsonObject]] = []
        # Reconcile all prior intent before issuing any fresh evaluation request.
        for task, state in pending:
            identifier = stable_id(self.arguments.run_id, phase, str(task["category"]), str(task["id"]), 0, "episode")
            path = self.arguments.run_root / "episodes" / f"{identifier}.json"
            if identifier in cast(list[str], state["started_episodes"]) or path.exists():
                await _evaluate(task, state)
            else:
                missing.append((task, state))
        semaphore = asyncio.Semaphore(4)

        async def _bounded_evaluate(task: JsonObject, state: JsonObject) -> None:
            async with semaphore:
                await _evaluate(task, state)

        evaluations = [asyncio.create_task(_bounded_evaluate(task, state)) for task, state in missing]
        settled = asyncio.gather(*evaluations, return_exceptions=True)
        try:
            outcomes = await asyncio.shield(settled)
        except asyncio.CancelledError:
            for evaluation in evaluations:
                evaluation.cancel()
            try:
                async with asyncio.timeout(30):
                    await asyncio.shield(settled)
            except (TimeoutError, asyncio.CancelledError):
                for evaluation in evaluations:
                    evaluation.cancel()
                await settled
            raise
        release_error: RuntimeError | None = None
        try:
            _check_release()
        except RuntimeError as error:
            release_error = error
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        if release_error is not None:
            raise release_error
        phase_state["complete"] = True
        self.save()

    async def commit_task(
        self, task: JsonObject, episodes: list[JsonObject], release: JsonObject, state: JsonObject
    ) -> None:
        references = [reference for episode in episodes for reference in cast(list[str], episode["references"])]
        report_ids = [str(episode["report_id"]) for episode in episodes]
        step = int(cast(int, self.cursor["training_step"])) + 1
        checkpoint_parent = str(release["release_id"] if release["checkpoint"] else release["parent_release_id"])
        for episode in episodes:
            payload = terminal_report(METHOD, episode, "", step)
            report_path = self.arguments.run_root / "reports" / f"{episode['report_id']}.json"
            if report_path.exists() and read_object(report_path) != payload:
                raise RuntimeError("persisted report differs from the complete OPD episode")
        commit = matched_commit(self.api, report_ids, references, str(release["release_id"]), step, checkpoint_parent)
        if commit is None:
            if self.api.current_release()["release_id"] != release["release_id"]:
                raise RuntimeError("sampling release changed before the complete task committed")
            for episode in episodes:
                report_id = str(episode["report_id"])
                report_path = self.arguments.run_root / "reports" / f"{report_id}.json"
                payload = terminal_report(METHOD, episode, "", step)
                if report_path.exists():
                    if read_object(report_path) != payload:
                        raise RuntimeError("persisted report differs from the complete OPD episode")
                else:
                    write_object(report_path, payload)
                existing = self.api.record(report_id)
                if existing is not None:
                    if existing["payload"] != {
                        key: value for key, value in payload.items() if key != "agent_record_id"
                    }:
                        raise RuntimeError("existing report differs from the persisted deterministic report")
                    continue
                attempted = cast(list[str], state["report_attempted"])
                if report_id in attempted:
                    reconciled = matched_commit(
                        self.api, report_ids, references, str(release["release_id"]), step, checkpoint_parent
                    )
                    if reconciled is None:
                        raise RuntimeError(
                            "unknown report outcome; reconcile retained records or consumption, do not replay"
                        )
                    commit = reconciled
                    break
                attempted.append(report_id)
                self.save()
                self.api.report(payload)
            deadline = time.monotonic() + self.arguments.commit_timeout_seconds
            while commit is None:
                commit = matched_commit(
                    self.api, report_ids, references, str(release["release_id"]), step, checkpoint_parent
                )
                if commit is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("task commit deadline exceeded; resume reconciles this task without replay")
                await asyncio.sleep(self.arguments.poll_seconds)
        write_object(self.arguments.run_root / "commits" / f"{step:03d}.json", commit)
        self.cursor["training_release"] = self.api.current_release()
        self.cursor["training_step"] = step
        state.update(complete=True, commit_step=step)
        self.save()


def validate_reference_verification(root: Path, manifest: JsonObject) -> None:
    if __package__:
        from .taskexport import reference_key, validate_task_files
    else:
        from taskexport import reference_key, validate_task_files

    path = root / "reference-verification.json"
    if not path.exists():
        raise RuntimeError("OPD training requires all 216 isolated reference-solution checks first")
    verification = read_object(path)
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest_checksum = hashlib.sha256(manifest_bytes).hexdigest()
    if (
        verification.get("schema_version") != 2
        or verification.get("manifest_sha256") != manifest_checksum
        or json.loads(manifest_bytes) != manifest
    ):
        raise RuntimeError("reference verification is not bound to this manifest; run fresh qualification")
    if (
        verification.get("revision") != DATASET_REVISION
        or verification.get("status") != "passed"
        or verification.get("total") != 216
        or verification.get("passed") != 216
    ):
        raise RuntimeError("reference verification did not pass all 216 pinned tasks")
    checked = cast(list[JsonObject], verification["results"])
    identities = {(row["role"], row["category"], row["id"]): row for row in checked}
    tasks = cast(list[JsonObject], manifest["training"]) + cast(list[JsonObject], manifest["independent"])
    if len(identities) != 216 or len(checked) != 216:
        raise RuntimeError("reference verification has missing or repeated task identities")
    for task in tasks:
        validate_task_files(root, task)
        row = identities.get((task["role"], task["category"], task["id"]))
        if (
            row is None
            or row.get("position") != task["position"]
            or row.get("reference_sha256") != task["reference_sha256"]
            or row.get("test_sha256") != task["test_sha256"]
            or row.get("task_files") != task["files"]
            or row.get("reference_key") != reference_key(manifest_checksum, task)
            or row.get("reward") != 1
            or row.get("error")
        ):
            raise RuntimeError("reference verification differs from the pinned task or failed its verifier")


def selected_training_positions(arguments: argparse.Namespace, manifest: JsonObject | None) -> list[int] | str:
    if arguments.profile == "full":
        return list(range(96))
    if manifest is None:
        return "first subtask and its matching complex task; export manifest required to resolve positions"
    tasks = cast(list[JsonObject], manifest["training"])
    matching = next(task for task in tasks[48:] if task["pair_id"] == tasks[0]["pair_id"])
    return [0, int(cast(int, matching["position"]))]


def run_specification(arguments: argparse.Namespace, manifest: JsonObject | None) -> JsonObject:
    return {
        "schema_version": 1,
        "method": METHOD,
        "run_id": arguments.run_id,
        "profile": arguments.profile,
        "base_commit": BASE_COMMIT,
        "opd_implementation_branch": OPD_IMPLEMENTATION_BRANCH,
        "opd_implementation_revision": OPD_IMPLEMENTATION_REVISION,
        "dataset": "osunlp/AgentCL",
        "dataset_revision": DATASET_REVISION,
        "manifest_sha256": (
            hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() if manifest else None
        ),
        "tasks": (
            {
                role: [
                    {key: row[key] for key in ("category", "id", "pair_id", "position")}
                    for row in cast(list[JsonObject], manifest[role])
                ]
                for role in ("training", "independent")
            }
            if manifest
            else None
        ),
        "model": arguments.model_path,
        "tokenizer": arguments.model_path,
        "teacher_checkpoint": arguments.teacher_checkpoint,
        "teacher": {
            "source": "separate frozen checkpoint in the actor layout",
            "checkpoint": arguments.teacher_checkpoint,
            "input": "exact recorded student token IDs; no prompt rendering or added context",
            "checkpoint_identity": "immutable checkpoint path; checkpoint contents must not change in place",
        },
        "native_training_options": {
            **OPD_TRAINING_OPTIONS,
            "opd-teacher-checkpoint": arguments.teacher_checkpoint,
        },
        "seed": arguments.seed,
        "sampling_seed_protocol": "base seed plus UUID5(category,id,attempt) prefix modulo 2^31; same task/attempt across phases",
        "scenario": arguments.scenario,
        "service_url": arguments.service_url,
        "steps": arguments.steps,
        "selected_training_positions": selected_training_positions(arguments, manifest),
        "independent_task_count": 2 if arguments.profile == "smoke" else 120,
        "attempts_per_training_task": arguments.attempts,
        "student_window_tokens": 8192,
        "teacher_window_tokens": 16384,
        "response_tokens": 2048,
        "native_input_checks": arguments.native_input_checks,
        "max_turns": arguments.max_turns,
        "tool_timeout_seconds": 30,
        "max_tool_output_chars": 8000,
        "temperature": 0.7,
        "learning_rate": 1e-5,
        "warmup_steps": min(10, arguments.steps - 1),
        "schedule_steps": arguments.steps,
        "ema": 0.0,
        "loss": "reverse KL",
        "top_k": 1,
        "importance_sampling_cap": 0.0,
        "skip_response_tokens": 0,
        "realign_threshold": 0,
        "scaffold_tolerance": 0,
        "gpu_proposal": {
            "count": 4,
            "actor_tensor_parallel": 4,
            "inference_tensor_parallel": 1,
            "inference_engines": 4,
            "colocated": True,
            "separate_teacher_engine": False,
        },
        "data_root": str(arguments.data_root),
        "run_root": str(arguments.run_root),
        "evaluation": "frozen one attempt, no reports, no demonstration lookup, reset sandbox, no external memory",
        "wandb": (
            {
                "project": arguments.wandb_project,
                "entity": arguments.wandb_entity,
                "enabled": bool(arguments.wandb_project),
            }
            if arguments.wandb_project
            else "disabled; online uploads require separate approval"
        ),
        "resume": "same supervisor/service only; unknown inference/report outcomes stop; exact optimizer restart is not promised",
    }


def load_manifest(root: Path) -> JsonObject:
    if __package__:
        from .taskexport import validate_task_files
    else:
        from taskexport import validate_task_files

    manifest = read_object(root / "manifest.json")
    if manifest.get("dataset") != "osunlp/AgentCL":
        raise ValueError("dataset identity is not osunlp/AgentCL")
    expected_sources = {
        "bigcodebench_lite_pro.dependent.json": "cd462894801f087866019d12d85f669d1ddd72b97ef722dcd180587868ad6df2",
        "bigcodebench_lite_pro.conventional.json": "55d18f52f7e2562514dfc5bb4d75b161ce58861b1e50ea298e4e0033bbe0b636",
        "humaneval_pro.conventional.json": "c15f68d46795087d77dc6650b894fbba4d94e2e85fe8cc47f998c3cfdb79075b",
    }
    if manifest.get("source_files") != expected_sources:
        raise ValueError("dataset source checksums differ from the approved pin")
    if manifest["revision"] != DATASET_REVISION:
        raise ValueError("dataset revision is not the approved pin")
    for role, count in (("training", 96), ("independent", 120)):
        rows = cast(list[JsonObject], manifest[role])
        identities = [(row["category"], row["id"]) for row in rows]
        if len(rows) != count or len(set(identities)) != count:
            raise ValueError(f"{role} requires {count} unique category/id tasks")
        if [row["position"] for row in rows] != list(range(count)):
            raise ValueError("task order must match the pinned manifest")
        if any(row["role"] != role for row in rows):
            raise ValueError("task role differs from manifest split")
        for row in rows:
            validate_task_files(root, row)
            for key in ("task_path", "reference_path"):
                path = (root / str(row[key])).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("exported task paths must remain under the data root")
    training = cast(list[JsonObject], manifest["training"])
    if [row["category"] for row in training] != ["raw"] * 48 + ["new"] * 48:
        raise ValueError("training stream must contain the 48 subtasks followed by their complex tasks")
    raw_pairs = {row["pair_id"] for row in training[:48]}
    new_pairs = {row["pair_id"] for row in training[48:]}
    if len(raw_pairs) != 48 or raw_pairs != new_pairs:
        raise ValueError("dependent task pairs differ")
    return manifest


def validate_service_proposal(arguments: argparse.Namespace) -> None:
    from reef.service.deploy.config_utils import load_config
    from reef.service.deploy.service_config import service_config_from_mapping

    if not isinstance(arguments.teacher_checkpoint, str) or not arguments.teacher_checkpoint.strip():
        raise ValueError("a frozen teacher checkpoint is required for the OPD service proposal")
    source_root = str(HERE.parents[3])
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    configuration = load_config(HERE / "serve.yaml", interpolate_env=False)
    values = {
        "AGENTCL_RUN_ROOT": str(arguments.run_root),
        "AGENTCL_MODEL_PATH": arguments.model_path,
        "AGENTCL_TEACHER_CHECKPOINT": arguments.teacher_checkpoint,
        "AGENTCL_STEPS": str(arguments.steps),
        "AGENTCL_WARMUP": str(min(10, arguments.steps - 1)),
        "AGENTCL_BATCH_SIZE": str(arguments.attempts),
        "REEF_PORT": "28902",
        "AGENTCL_ROUTER_PORT": "23002",
        "REEF_TOKEN": "",
    }

    def _substitute(value):
        if isinstance(value, dict):
            return {key: _substitute(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_substitute(item) for item in value]
        if isinstance(value, str):
            for key, replacement in values.items():
                value = value.replace("${" + key + ":?}", replacement).replace("${" + key + "}", replacement)
        return value

    recipe_configuration = cast(JsonObject, cast(JsonObject, configuration["recipe"])["config"])
    if "tokenizer-path" in recipe_configuration or "tokenizer_path" in recipe_configuration:
        raise ValueError("OPD must use recorded token IDs without a recipe tokenizer")
    service = service_config_from_mapping(_substitute(configuration))
    options = service.training_backend_options
    if (
        service.inference_num_gpus != 4
        or service.tensor_parallel_size != 1
        or service.colocate is not True
        or options.get("tensor-model-parallel-size") != "4"
    ):
        raise ValueError("service topology differs from four colocated GPUs and a TP4 actor")
    expected = {**OPD_TRAINING_OPTIONS, "opd-teacher-checkpoint": arguments.teacher_checkpoint}
    if any(options.get(key) != value for key, value in expected.items()):
        raise ValueError("service proposal differs from the frozen checkpoint-backed OPD settings")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("export", "baseline", "train", "evaluate", "verify", "render-traces", "qualify-inputs")
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--phase", choices=("frozen-repeat", "independent"), default="frozen-repeat")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--references", action="store_true", help="verify 216 references in isolated Harbor containers"
    )
    parser.add_argument("--run-id", default=f"agentcl-{METHOD}")
    parser.add_argument("--run-root", type=Path, default=HERE / "work")
    parser.add_argument("--data-root", type=Path, default=HERE / "data")
    parser.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    parser.add_argument("--service-url", default=os.environ.get("REEF_SERVICE_URL", "http://127.0.0.1:28902"))
    parser.add_argument("--scenario", default=os.environ.get("REEF_SCENARIO", f"agentcl-{METHOD}"))
    parser.add_argument(
        "--model-path", default=os.environ.get("AGENTCL_MODEL_PATH", "/root/models/Qwen2.5-7B-Instruct")
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default=os.environ.get("AGENTCL_TEACHER_CHECKPOINT", ""),
        help="immutable frozen teacher checkpoint; required for campaign execution and service proposals",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", help="approval-gated native optimizer and evaluation uploads")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--native-input-checks",
        action="store_true",
        help="check first-task tensor integrity; qualify-inputs requires overall active multi-turn coverage",
    )
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--commit-timeout-seconds", type=float, default=7200)
    parser.add_argument("--poll-seconds", type=float, default=2)
    arguments = parser.parse_args()
    arguments.steps = 2 if arguments.profile == "smoke" else 96
    arguments.attempts = 1
    arguments.teacher_checkpoint = arguments.teacher_checkpoint.strip()
    if (
        arguments.command in ("baseline", "train", "evaluate") or (arguments.dry_run and arguments.command != "export")
    ) and not arguments.teacher_checkpoint:
        parser.error("--teacher-checkpoint or AGENTCL_TEACHER_CHECKPOINT is required; it must not be empty")
    arguments.run_root = arguments.run_root.expanduser().resolve()
    arguments.data_root = arguments.data_root.expanduser().resolve()
    arguments.cache_dir = arguments.cache_dir.expanduser().resolve()
    if arguments.max_turns != 8 or arguments.commit_timeout_seconds <= 0 or arguments.poll_seconds <= 0:
        parser.error("positive timing and exactly eight allowed turns are required")
    return arguments


async def main() -> None:
    arguments = parse_arguments()
    if arguments.command == "export":
        if arguments.dry_run:
            print(json.dumps(run_specification(arguments, None), indent=2))
            return
        if __package__:
            from .taskexport import export_tasks
        else:
            from taskexport import export_tasks
        manifest = export_tasks(arguments.data_root, arguments.cache_dir)
        load_manifest(arguments.data_root)
        print(json.dumps(manifest, indent=2))
        return
    manifest = load_manifest(arguments.data_root) if (arguments.data_root / "manifest.json").exists() else None
    if arguments.dry_run:
        validate_service_proposal(arguments)
        print(
            json.dumps(
                {
                    **run_specification(arguments, manifest),
                    "command": arguments.command,
                    "phase": arguments.phase,
                    "submits": 0,
                },
                indent=2,
            )
        )
        print("Service proposal (start only under an approved external supervisor):")
        wandb_flag = ""
        if arguments.wandb_project:
            configuration = {
                "enabled": True,
                "project": arguments.wandb_project,
                "entity": arguments.wandb_entity,
                "mode": "online",
                "directory": str(arguments.run_root / "wandb"),
                "upload_checkpoints": False,
            }
            wandb_flag = " --observability.wandb " + shlex.quote(json.dumps(configuration))
        environment = {
            "AGENTCL_RUN_ROOT": str(arguments.run_root),
            "AGENTCL_MODEL_PATH": arguments.model_path,
            "AGENTCL_TEACHER_CHECKPOINT": arguments.teacher_checkpoint,
            "AGENTCL_STEPS": str(arguments.steps),
            "AGENTCL_WARMUP": str(min(10, arguments.steps - 1)),
            "AGENTCL_BATCH_SIZE": str(arguments.attempts),
            "REEF_PORT": "28902",
            "AGENTCL_ROUTER_PORT": "23002",
        }
        assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in environment.items())
        print(f"{assignments} reef serve -c {shlex.quote(str(HERE / 'serve.yaml'))}{wandb_flag}")
        return
    if arguments.command == "qualify-inputs":
        result = qualify_inputs(arguments.run_root)
        write_object(arguments.run_root / "native-input-checks.json", result)
        print(json.dumps(result, indent=2))
        return
    if arguments.command == "verify" and arguments.references:
        if os.environ.get("AGENTCL_EXECUTION_APPROVED") != "1":
            raise SystemExit("Reference container execution requires approval; then set AGENTCL_EXECUTION_APPROVED=1")
        if __package__:
            from .taskexport import verify_references
        else:
            from taskexport import verify_references
        print(json.dumps(await verify_references(arguments.data_root, arguments.run_root / "reference-lab"), indent=2))
        return
    if arguments.command in ("verify", "render-traces"):
        if __package__:
            from .render_traces import render_traces
            from .verification import verify_run
        else:
            from render_traces import render_traces
            from verification import verify_run
        if arguments.command == "verify":
            result = verify_run(arguments.run_root)
            write_object(arguments.run_root / "verification.json", result)
            print(json.dumps(result, indent=2))
            if result["complete"] is not True:
                raise SystemExit(1)
        else:
            print(render_traces(arguments.run_root))
        return
    if arguments.wandb_project and os.environ.get("AGENTCL_UPLOADS_APPROVED") != "1":
        raise SystemExit("Online W&B uploads require separate approval; then set AGENTCL_UPLOADS_APPROVED=1")
    if os.environ.get("AGENTCL_EXECUTION_APPROVED") != "1":
        raise SystemExit("Inference/training must be explicitly approved; only then set AGENTCL_EXECUTION_APPROVED=1")
    if manifest is None:
        raise SystemExit("export the pinned dataset first")
    api = HttpReefApi(arguments.service_url, arguments.scenario, os.environ.get("REEF_TOKEN"))
    api.ensure_scenario()
    campaign = Campaign(arguments, api, HarborBackend(arguments), manifest)
    if arguments.command == "baseline":
        await campaign.execute_phase("baseline")
        await campaign.execute_phase("baseline-independent")
    else:
        await campaign.execute_phase("train" if arguments.command == "train" else arguments.phase)
    records = [read_object(path) for path in sorted((arguments.run_root / "episodes").glob("*.json"))]
    records.sort(
        key=lambda record: (
            str(record["phase"]),
            int(cast(int, record["campaign_position"])),
            int(cast(int, record["attempt"])),
        )
    )
    summary = build_summary(records)
    write_object(arguments.run_root / "summary.json", summary)
    if arguments.wandb_project:
        import wandb

        with wandb.init(
            project=arguments.wandb_project,
            entity=arguments.wandb_entity,
            id=stable_id(arguments.run_id, "evaluation", "all", "all", 0, "wandb"),
            resume="allow",
            name=arguments.run_id + "-evaluation",
            dir=str(arguments.run_root / "wandb"),
            config={"method": METHOD, "profile": arguments.profile, "score_protocol": summary["score_protocol"]},
        ) as evaluation_run:
            for phase, phase_summary in cast(JsonObject, summary["phases"]).items():
                primary = cast(JsonObject, cast(JsonObject, phase_summary)["one_attempt"])
                evaluation_run.log(
                    {phase + "/" + key: value for key, value in primary.items() if isinstance(value, (int, float))}
                )


if __name__ == "__main__":
    asyncio.run(main())
