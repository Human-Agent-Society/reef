"""Run one AgentCL coding stream with native OPD, SDPO or SDFT."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import shlex
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

if __package__:
    from .report import JsonObject, read_object, stable_id, terminal_report, write_object
    from .taskexport import REVISION, export_tasks, reference_key, validate_task_files, verify_references
else:
    from report import JsonObject, read_object, stable_id, terminal_report, write_object
    from taskexport import REVISION, export_tasks, reference_key, validate_task_files, verify_references

HERE = Path(__file__).resolve().parent


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


def validate_reference_verification(root: Path, manifest: JsonObject, method: str) -> None:
    path = root / "reference-verification.json"
    if not path.exists():
        raise RuntimeError("training requires all 216 isolated reference-solution checks first")
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
        verification.get("revision") != REVISION
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


def load_manifest(root: Path) -> JsonObject:
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
    if manifest["revision"] != REVISION:
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


class Campaign:
    """One sequential stream; interrupted phases are terminal and never replayed."""

    def __init__(self, arguments: argparse.Namespace, api: ReefApi, backend: EpisodeBackend, manifest: JsonObject):
        self.arguments = arguments
        self.api = api
        self.backend = backend
        self.manifest = manifest
        self.root = arguments.run_root
        self.root.mkdir(parents=True, exist_ok=True)
        specification = {
            "method": arguments.method,
            "run_id": arguments.run_id,
            "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
            "model": arguments.model_path,
            "teacher_checkpoint": arguments.teacher_checkpoint if arguments.method == "opd" else None,
            "profile": arguments.profile,
            "steps": arguments.steps,
            "attempts": arguments.attempts,
            "seed": arguments.seed,
            "max_turns": arguments.max_turns,
            "scenario": arguments.scenario,
            "service_url": arguments.service_url,
            "wandb_project": arguments.wandb_project,
            "wandb_entity": arguments.wandb_entity,
        }
        path = self.root / "run-manifest.json"
        if path.exists():
            if read_object(path) != specification:
                raise RuntimeError("run settings changed; use a new run directory and scenario")
            self.state = read_object(self.root / "cursor.json")
        else:
            if api.commits([]):
                raise RuntimeError("a new campaign requires an unused training scenario")
            release = api.current_release()
            self.state = {"initial_release": release, "training_release": release, "training_step": 0, "phases": {}}
            write_object(path, specification)
            self.save()

    def save(self) -> None:
        write_object(self.root / "cursor.json", self.state)

    def tasks(self, phase: str) -> list[JsonObject]:
        role = "independent" if phase in ("independent", "baseline-independent") else "training"
        rows = cast(list[JsonObject], self.manifest[role])
        if self.arguments.profile == "smoke":
            if role == "training":
                return [rows[0], next(row for row in rows[48:] if row["pair_id"] == rows[0]["pair_id"])]
            return rows[:2]
        return rows

    def check_release(self, release: JsonObject) -> None:
        if self.api.current_release() != release:
            raise RuntimeError("the served release changed; another writer may own this scenario")

    async def episode(self, task: JsonObject, phase: str, attempt: int, release: JsonObject) -> JsonObject:
        identifier = stable_id(
            self.arguments.run_id, phase, str(task["category"]), str(task["id"]), attempt, "episode"
        )
        intent = self.root / "started-episodes" / (identifier + ".json")
        intent.parent.mkdir(exist_ok=True)
        with intent.open("x") as handle:
            json.dump({"episode_id": identifier, "phase": phase, "release_id": release["release_id"]}, handle)
        sampling = stable_id("sampling", "all", str(task["category"]), str(task["id"]), attempt, "seed")
        seed = (self.arguments.seed + int(sampling.replace("-", "")[:8], 16)) % (2**31)
        result = await self.backend.run({**task, "sampling_seed": seed}, identifier, phase, release)
        turns = cast(list[JsonObject], result["turns"])
        references = cast(list[str], result["references"])
        if result["episode_id"] != identifier or result["release_id"] != release["release_id"]:
            raise RuntimeError("episode identity or sampled release differs")
        if (
            not references
            or references != [turn["receipt"] for turn in turns]
            or len(set(references)) != len(references)
        ):
            raise RuntimeError("the episode must retain every ordered unique inference receipt")
        if any(turn["release_id"] != release["release_id"] for turn in turns):
            raise RuntimeError("episode turns were sampled from different releases")
        runtime_ids = {turn.get("runtime_load_id") for turn in turns if "runtime_load_id" in turn}
        if len(runtime_ids) > 1 or (
            runtime_ids
            and (
                not all(runtime_ids)
                or any(turn.get("runtime_load_id") != result.get("runtime_load_id") for turn in turns)
            )
        ):
            raise RuntimeError("episode turns have mixed or missing runtime load IDs")
        result.update(
            phase=phase,
            category=task["category"],
            task_id=task["id"],
            position=task["position"],
            attempt=attempt,
            report_id=stable_id(
                self.arguments.run_id, phase, str(task["category"]), str(task["id"]), attempt, "report"
            ),
        )
        write_object(self.root / "episodes" / (identifier + ".json"), result)
        return result

    async def commit_task(self, task: JsonObject, episodes: list[JsonObject], release: JsonObject) -> None:
        step = int(cast(int, self.state["training_step"])) + 1
        report_ids = [str(episode["report_id"]) for episode in episodes]
        references = [str(receipt) for episode in episodes for receipt in cast(list, episode["references"])]
        for episode in episodes:
            if self.arguments.method == "sdft":
                reference = self.arguments.data_root / str(task["reference_path"])
                if hashlib.sha256(reference.read_bytes()).hexdigest() != task["reference_sha256"]:
                    raise RuntimeError("verified demonstration bytes changed")
                context = reference.read_text()
            elif self.arguments.method == "sdpo":
                context = (
                    "The submitted solution passed the verifier."
                    if episode["score"] == 1
                    else "The submitted solution did not pass the verifier."
                )
            else:
                context = ""
            payload = terminal_report(self.arguments.method, episode, context, step)
            write_object(self.root / "reports" / (str(episode["report_id"]) + ".json"), payload)
            self.api.report(payload)
        parent = str(release["release_id"] if release["checkpoint"] else release["parent_release_id"])
        deadline = time.monotonic() + self.arguments.commit_timeout_seconds
        while True:
            commit = matched_commit(self.api, report_ids, references, str(release["release_id"]), step, parent)
            if commit is not None:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("unknown training outcome; preserve records and do not replay this phase")
            await asyncio.sleep(self.arguments.poll_seconds)
        write_object(self.root / "commits" / f"{step:03d}.json", commit)
        self.state.update(training_step=step, training_release=self.api.current_release())
        self.save()

    async def execute_phase(self, phase: str) -> JsonObject:
        phases = cast(JsonObject, self.state["phases"])
        if phase in phases:
            raise RuntimeError("this phase already started; interrupted work must never be replayed")
        if phase == "train":
            validate_reference_verification(self.arguments.data_root, self.manifest, self.arguments.method)
        elif phase in ("frozen-repeat", "independent") and self.state["training_step"] != self.arguments.steps:
            raise RuntimeError("final evaluation requires every configured training commit")
        release = cast(
            JsonObject,
            self.state["initial_release"] if phase.startswith("baseline") else self.state["training_release"],
        )
        self.check_release(release)
        phase_marker = self.root / "started-phases" / phase
        phase_marker.parent.mkdir(exist_ok=True)
        phase_marker.mkdir()
        phases[phase] = {"status": "started", "release_id": release["release_id"]}
        self.save()
        results = []
        for task in self.tasks(phase):
            self.check_release(release)
            # Complete all siblings before reporting any; no next task begins before its commit.
            episodes = [
                await self.episode(task, phase, attempt, release)
                for attempt in range(self.arguments.attempts if phase == "train" else 1)
            ]
            self.check_release(release)
            for episode in episodes:
                score = episode.get("score")
                if isinstance(score, bool) or not isinstance(score, (int, float)) or score not in (0, 1):
                    raise RuntimeError("the trusted verifier must return a binary score")
                scored = episode["outcome"] == "completed"
                truncated = phase != "train" and episode["outcome"] == "truncated" and score == 0
                if not scored and not truncated:
                    raise RuntimeError("episode fault or incomplete training; preserve the original outcome")
            if phase == "train":
                await self.commit_task(task, episodes, release)
                release = cast(JsonObject, self.state["training_release"])
            results.extend(episodes)
        summary: JsonObject = {
            "episodes": len(results),
            "accuracy": sum(float(row["score"]) for row in results) / len(results),
            "completion_tokens": sum(int(cast(int, row["completion_tokens"])) for row in results),
            "release_id": release["release_id"],
        }
        phases[phase] = {"status": "complete", **summary}
        self.save()
        return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "verify-references", "baseline", "train", "evaluate"))
    parser.add_argument("--method", choices=("opd", "sdft", "sdpo"), default="sdft")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--phase", choices=("frozen-repeat", "independent"), default="frozen-repeat")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--run-root", type=Path, default=HERE / "work")
    parser.add_argument("--data-root", type=Path, default=HERE / "data")
    parser.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    parser.add_argument(
        "--model-path", default=os.environ.get("AGENTCL_MODEL_PATH", "/root/models/Qwen2.5-7B-Instruct")
    )
    parser.add_argument("--teacher-checkpoint", default=os.environ.get("AGENTCL_TEACHER_CHECKPOINT", ""))
    parser.add_argument("--service-url", default=os.environ.get("REEF_SERVICE_URL", "http://127.0.0.1:28902"))
    parser.add_argument("--scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--commit-timeout-seconds", type=float, default=7200)
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    arguments = parser.parse_args()
    arguments.steps = 2 if arguments.profile == "smoke" else 96
    arguments.attempts = (2 if arguments.profile == "smoke" else 4) if arguments.method == "sdpo" else 1
    arguments.run_id = arguments.run_id or "agentcl-" + arguments.method
    arguments.scenario = arguments.scenario or arguments.run_id
    arguments.run_root = arguments.run_root.expanduser().resolve()
    arguments.data_root = arguments.data_root.expanduser().resolve()
    arguments.cache_dir = arguments.cache_dir.expanduser().resolve()
    if arguments.max_turns < 2 or any(
        not math.isfinite(value) or value <= 0 for value in (arguments.commit_timeout_seconds, arguments.poll_seconds)
    ):
        parser.error("positive timing and at least two allowed turns are required")
    if (
        arguments.method == "opd"
        and arguments.command in ("train", "baseline", "evaluate")
        and not arguments.teacher_checkpoint.strip()
    ):
        parser.error("OPD requires a compatible immutable teacher checkpoint")
    return arguments


async def main() -> None:
    arguments = parse_arguments()
    if arguments.dry_run:
        configuration = HERE / ("serve-" + arguments.method + ".yaml")
        environment = {
            "AGENTCL_RUN_ROOT": str(arguments.run_root),
            "AGENTCL_MODEL_PATH": arguments.model_path,
            "AGENTCL_STEPS": str(arguments.steps),
            "AGENTCL_WARMUP": str(min(10, arguments.steps - 1)),
            "AGENTCL_BATCH_SIZE": str(arguments.attempts),
            "REEF_PORT": "28902",
            "AGENTCL_ROUTER_PORT": "23002",
        }
        if arguments.method == "opd":
            environment["AGENTCL_TEACHER_CHECKPOINT"] = arguments.teacher_checkpoint
        override = ""
        if arguments.wandb_project:
            override = " --observability.wandb " + shlex.quote(
                json.dumps(
                    {
                        "enabled": True,
                        "project": arguments.wandb_project,
                        "entity": arguments.wandb_entity,
                        "mode": "online",
                        "upload_checkpoints": False,
                    }
                )
            )
        print(
            " ".join(key + "=" + shlex.quote(value) for key, value in environment.items())
            + " reef serve -c "
            + str(configuration)
            + override
        )
        print(
            json.dumps(
                {"method": arguments.method, "steps": arguments.steps, "attempts": arguments.attempts, "submits": 0}
            )
        )
        return
    if os.environ.get("AGENTCL_EXECUTION_APPROVED") != "1" and arguments.command != "export":
        raise SystemExit("Model and container execution require explicit approval")
    if arguments.wandb_project and os.environ.get("AGENTCL_UPLOADS_APPROVED") != "1":
        raise SystemExit("Online W&B uploads require explicit approval")
    if arguments.command == "export":
        export_tasks(arguments.data_root, arguments.cache_dir)
        return
    manifest = load_manifest(arguments.data_root)
    if arguments.command == "verify-references":
        result = await verify_references(arguments.data_root, arguments.run_root / "reference-lab")
        print(json.dumps(result))
        if result["status"] != "passed":
            raise SystemExit(1)
        return
    api = HttpReefApi(arguments.service_url, arguments.scenario, os.environ.get("REEF_TOKEN"))
    api.ensure_scenario()
    campaign = Campaign(arguments, api, HarborBackend(arguments), manifest)
    phases = (
        ("baseline", "baseline-independent")
        if arguments.command == "baseline"
        else ("train" if arguments.command == "train" else arguments.phase,)
    )
    for phase in phases:
        summary = await campaign.execute_phase(phase)
        print(json.dumps({"phase": phase, **summary}))
        if arguments.wandb_project:
            import wandb

            with wandb.init(
                project=arguments.wandb_project,
                entity=arguments.wandb_entity,
                id=stable_id(arguments.run_id, "evaluation", "all", "all", 0, "wandb"),
                resume="allow",
                dir=str(arguments.run_root),
                name=arguments.run_id + "-evaluation",
            ) as evaluation:
                evaluation.log(
                    {phase + "/" + key: value for key, value in summary.items() if isinstance(value, (int, float))}
                )


if __name__ == "__main__":
    asyncio.run(main())
