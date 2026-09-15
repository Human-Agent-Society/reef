"""Rounds: Designer rounds, then a Reasoning Agent round, the other side fixed, every report naming its opponent.

A round pulls the Designer's current prompt and both versions, runs one generation whose plays are held
back, waits for the Designer's deployment to commit on the regret reports, and, every ``agent_round_every``
rounds, reports the held plays of those rounds to the Reasoning Agent's deployment and waits for its commit.
A role whose scenario evolves nothing (no harness surface, no training runtime) is fixed: it is measured
against and never waited on. Each round is one line of ``.spade/rounds.jsonl`` under the tasks root.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from reef_client.client import ReefClient, ReefClientError

from recipes.beta.spade.designer import DesignerPrompt, PlayRecord
from recipes.beta.spade.generation import (
    REPORT_DIRECTORY,
    Checks,
    Designer,
    Generation,
    GenerationError,
    GenerationRequest,
    RealChecks,
    ReasoningAgent,
    TaskMeasure,
    add_generation_arguments,
    reef_designer,
    reef_reasoning_agent,
    request_from_arguments,
    roles_from_arguments,
)
from recipes.beta.spade.roles import Role, RoleVersion

ROUNDS_FILE = "rounds.jsonl"
TREE_FILE = "native/tree.json"
HARNESS_PATH = "/reef/harness"
STATUS_PATH = "/reef/status"
SCENARIOS_PATH = "/reef/scenarios"
DEFAULT_WAIT_TIMEOUT_S = 3600.0
DEFAULT_POLL_S = 5.0
DEFAULT_AGENT_ROUND_EVERY = 1
CLIENT_TIMEOUT_S = 60.0
KIND_HARNESS = "harness"
KIND_WEIGHTS = "weights"
KIND_FIXED = "fixed"
SYNCHRONIZED = "synchronized"
ARM_PLAIN = "plain"
ARM_HINT = "hint"


class RoundError(RuntimeError):
    """A round could not complete: a deployment refused, errored or never committed."""


class Timer(ABC):
    """The clock a wait reads and the sleep it takes between polls."""

    @abstractmethod
    def now(self) -> float: ...

    @abstractmethod
    def sleep(self, seconds: float) -> None: ...


class MonotonicTimer(Timer):
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class DeploymentStatus:
    """One scenario's block of ``GET /reef/status``, the fields a round waits on."""

    step: int | None
    step_metrics: Mapping[str, object]
    scenario_step: int
    runtime_load_id: str | None
    head_state: str | None
    release_id: str | None
    batch_ready: bool
    processor: Mapping[str, object]
    error: str | None


def parsed_status(document: Mapping[str, object], scenario: str) -> DeploymentStatus:
    """The scenario's status block; a scenario the service does not know is an error."""
    scenarios = document.get("scenarios")
    block = scenarios.get(scenario) if isinstance(scenarios, Mapping) else None
    if not isinstance(block, Mapping):
        raise RoundError(f"the service reports no scenario {scenario!r}")
    committed = block.get("last_committed_step")
    step = committed.get("step") if isinstance(committed, Mapping) else None
    metrics = committed.get("metrics") if isinstance(committed, Mapping) else None
    head = block.get("artifact_head_sync")
    head_state = head.get("state") if isinstance(head, Mapping) else None
    release_id = head.get("release_id") if isinstance(head, Mapping) else None
    processor = block.get("processor")
    error = document.get("error")
    return DeploymentStatus(
        step=step if isinstance(step, int) and not isinstance(step, bool) else None,
        step_metrics=dict(metrics) if isinstance(metrics, Mapping) else {},
        scenario_step=int(block.get("scenario_step") or 0),
        runtime_load_id=str(block["current_runtime_load_id"]) if block.get("current_runtime_load_id") else None,
        head_state=str(head_state) if head_state is not None else None,
        release_id=str(release_id) if release_id is not None else None,
        batch_ready=bool(block.get("batch_ready")),
        processor=dict(processor) if isinstance(processor, Mapping) else {},
        error=str(error) if error else None,
    )


class Deployment:
    """One role's Reef service: its scenario, what it evolves, its version, and the wait for its next commit."""

    def __init__(
        self,
        role: Role,
        *,
        client: ReefClient | None = None,
        timer: Timer | None = None,
        poll_s: float = DEFAULT_POLL_S,
        wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    ) -> None:
        if poll_s <= 0 or wait_timeout_s <= 0:
            raise RoundError("poll_s and wait_timeout_s must be positive")
        self.role = role
        self.client = (
            client if client is not None else ReefClient(role.reef_url, token=role.token, timeout_s=CLIENT_TIMEOUT_S)
        )
        self.timer = timer if timer is not None else MonotonicTimer()
        self.poll_s = poll_s
        self.wait_timeout_s = wait_timeout_s
        self.kind_cache: str | None = None

    def ensure_scenario(self) -> None:
        """Create the scenario when it does not exist; the harness read never creates one."""
        try:
            self.client.post(SCENARIOS_PATH, self.role.scenario, {"name": self.role.scenario})
        except ReefClientError as exc:
            raise RoundError(
                f"{self.role.reef_url} refused scenario {self.role.scenario!r} ({exc.status}): {exc.body[:300]}"
            ) from exc
        except OSError as exc:
            raise RoundError(f"{self.role.reef_url} is not reachable: {exc}") from exc

    def manifest(self) -> Mapping[str, object] | None:
        """The served harness tree, or None for a deployment that serves no files."""
        try:
            return self.client.get(HARNESS_PATH, extra_headers={"x-reef-scenario": self.role.scenario})
        except ReefClientError as exc:
            if exc.status == 404:
                return None
            raise RoundError(
                f"{self.role.reef_url} refused the harness read ({exc.status}): {exc.body[:300]}"
            ) from exc
        except OSError as exc:
            raise RoundError(f"{self.role.reef_url} is not reachable: {exc}") from exc

    def status(self) -> DeploymentStatus:
        try:
            document = self.client.get(STATUS_PATH)
        except ReefClientError as exc:
            raise RoundError(f"{self.role.reef_url} refused the status read ({exc.status}): {exc.body[:300]}") from exc
        except OSError as exc:
            raise RoundError(f"{self.role.reef_url} is not reachable: {exc}") from exc
        return parsed_status(document, self.role.scenario)

    def kind(self) -> str:
        """What the deployment evolves: a harness tree, weights, or nothing (fixed)."""
        if self.kind_cache is None:
            if self.manifest() is not None:
                self.kind_cache = KIND_HARNESS
            elif self.status().runtime_load_id is not None:
                self.kind_cache = KIND_WEIGHTS
            else:
                self.kind_cache = KIND_FIXED
        return self.kind_cache

    def version(self) -> RoleVersion:
        kind = self.kind()
        if kind == KIND_HARNESS:
            manifest = self.manifest()
            release_id = manifest.get("release_id") if manifest is not None else None
            if not isinstance(release_id, str) or not release_id:
                raise RoundError(f"the harness of {self.role.scenario!r} names no release")
            return RoleVersion("release", release_id)
        if kind == KIND_WEIGHTS:
            load_id = self.status().runtime_load_id
            if load_id is None:
                raise RoundError(f"the weights of {self.role.scenario!r} name no runtime load")
            return RoleVersion("runtime", load_id)
        return RoleVersion("model", self.role.model)

    def prompt(self, base: DesignerPrompt) -> DesignerPrompt:
        """The Designer prompt of the served release, or the role's own when the deployment serves no tree."""
        manifest = self.manifest()
        if manifest is None:
            return base
        files = manifest.get("files")
        tree = files.get(TREE_FILE) if isinstance(files, Mapping) else None
        if not isinstance(tree, str):
            raise RoundError(f"the harness of {self.role.scenario!r} carries no {TREE_FILE}")
        try:
            entries = json.loads(tree)
        except ValueError as exc:
            raise RoundError(f"{TREE_FILE} of {self.role.scenario!r} is not JSON: {exc}") from exc
        if not isinstance(entries, list):
            raise RoundError(f"{TREE_FILE} of {self.role.scenario!r} must hold a list of entries")
        try:
            return base.with_entries(entries)
        except ValueError as exc:
            raise RoundError(f"the served Designer prompt of {self.role.scenario!r} is unusable: {exc}") from exc

    def wait_for_step(self, after: int | None, *, expected_traces: int | None = None) -> DeploymentStatus:
        """Poll until a step later than ``after`` is committed; a harness must also have synchronized its head.

        ``expected_traces`` is the number of reports the step must have consumed: a harness deployment batches
        by count, so a step over fewer traces than one generation sent means the batch size and the
        generation disagree, and the released tree mixes generations.
        """
        deadline = self.timer.now() + self.wait_timeout_s
        needs_head = self.kind() == KIND_HARNESS
        while True:
            status = self.status()
            if status.error:
                raise RoundError(f"{self.role.scenario!r} stopped training: {status.error}")
            is_later = status.step is not None and (after is None or status.step > after)
            if is_later and (not needs_head or status.head_state == SYNCHRONIZED):
                traces = status.step_metrics.get("traces")
                if expected_traces is not None and isinstance(traces, int) and traces != expected_traces:
                    raise RoundError(
                        f"{self.role.scenario!r} step {status.step} consumed {traces} reports, not the {expected_traces} "
                        "of one generation: the deployment's batch size must equal the generation's count"
                    )
                return status
            if self.timer.now() >= deadline:
                raise RoundError(
                    f"{self.role.scenario!r} committed no step after {after} within {self.wait_timeout_s:g} s "
                    f"(batch_ready={status.batch_ready}, processor={json.dumps(status.processor, sort_keys=True)})"
                )
            self.timer.sleep(self.poll_s)


@dataclass(frozen=True)
class RoundResult:
    """What one round did: both versions before and after, the steps committed, and the generation's measures."""

    generation: int
    designer_kind: str
    agent_kind: str
    designer_version_before: RoleVersion
    designer_version_after: RoleVersion
    agent_version_before: RoleVersion
    agent_version_after: RoleVersion
    designer_step: int | None
    agent_step: int | None
    measured: int
    refused: int
    reported_plays: int
    mean_regret: float | None
    manifest_path: str | None
    report_path: str
    experience: tuple[PlayRecord, ...]

    def as_line(self) -> dict[str, object]:
        """The JSON line of ``rounds.jsonl``: everything but the experience, which the report on disk holds."""
        line = asdict(self)
        del line["experience"]
        return line

    def summary(self) -> dict[str, object]:
        """What the next round's Designer reports carry as ``previous``: the trend the proposer needs."""
        return {
            "generation": self.generation,
            "mean_regret": self.mean_regret,
            "measured": self.measured,
            "refused": self.refused,
            "designer_version": version_metadata(self.designer_version_before),
            "agent_version": version_metadata(self.agent_version_before),
        }


def version_metadata(version: RoleVersion | None) -> dict[str, str] | None:
    return None if version is None else {"kind": version.kind, "id": version.id}


class SpadeRun:
    """Rounds over two roles; the Designer and the Reasoning Agent implementations default to Reef's."""

    def __init__(
        self,
        designer: Role,
        agent: Role,
        *,
        checks: Checks,
        tasks_root: Path,
        work_dir: Path,
        agent_round_every: int = DEFAULT_AGENT_ROUND_EVERY,
        designer_impl: Designer | None = None,
        agent_impl: ReasoningAgent | None = None,
        designer_deployment: Deployment | None = None,
        agent_deployment: Deployment | None = None,
        poll_s: float = DEFAULT_POLL_S,
        wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    ) -> None:
        if not isinstance(designer.harness, DesignerPrompt):
            raise RoundError("the Designer role must carry a DesignerPrompt harness")
        if agent_round_every < 1:
            raise RoundError("agent_round_every must be at least 1")
        self.designer = designer
        self.agent = agent
        self.checks = checks
        self.tasks_root = Path(tasks_root)
        self.work_dir = Path(work_dir)
        self.agent_round_every = agent_round_every
        self.designer_impl = designer_impl if designer_impl is not None else reef_designer(designer)
        self.agent_impl = agent_impl if agent_impl is not None else reef_reasoning_agent(agent, work_dir=self.work_dir)
        self.designer_deployment = (
            designer_deployment
            if designer_deployment is not None
            else Deployment(designer, poll_s=poll_s, wait_timeout_s=wait_timeout_s)
        )
        self.agent_deployment = (
            agent_deployment
            if agent_deployment is not None
            else Deployment(agent, poll_s=poll_s, wait_timeout_s=wait_timeout_s)
        )
        # Held plays wait for the agent round; every generation of a block is on policy for the same version.
        self.held: list[tuple[int, RoleVersion, tuple[TaskMeasure, ...]]] = []
        self.rounds_in_block = 0

    def round(self, request: GenerationRequest) -> RoundResult:
        designer_dep, agent_dep = self.designer_deployment, self.agent_deployment
        designer_dep.ensure_scenario()
        agent_dep.ensure_scenario()
        designer_kind, agent_kind = designer_dep.kind(), agent_dep.kind()
        designer_before = designer_dep.status().step if designer_kind != KIND_FIXED else None
        agent_before = agent_dep.status().step if agent_kind != KIND_FIXED else None
        prompt = designer_dep.prompt(self.designer.harness)
        designer_version, agent_version = designer_dep.version(), agent_dep.version()

        # The plays are held: the Reasoning Agent must not step while the Designer is measured against it.
        generation = Generation(
            designer=self.designer_impl,
            reasoning_agent=self.agent_impl,
            checks=self.checks,
            tasks_root=self.tasks_root,
            is_reporting_designer=designer_kind != KIND_FIXED,
            is_reporting_agent=False,
            designer_reward=self.designer.reward,
            agent_reward=self.agent.reward,
        )
        result = generation.run(
            replace(request, prompt=prompt, designer_version=designer_version, agent_version=agent_version)
        )

        designer_step: int | None = None
        if designer_kind != KIND_FIXED and result.proposals:
            expected = len(result.proposals) if designer_kind == KIND_HARNESS else None
            designer_step = designer_dep.wait_for_step(designer_before, expected_traces=expected).step
        designer_after = designer_dep.version()

        self.held.append((request.generation, designer_version, result.measures))
        self.rounds_in_block += 1
        reported_plays = 0
        agent_step: int | None = None
        if agent_kind != KIND_FIXED and self.rounds_in_block >= self.agent_round_every:
            reported_plays = self.reported_held()
            if reported_plays:
                agent_step = agent_dep.wait_for_step(agent_before).step
        agent_after = agent_dep.version()

        regrets = [measure.regret for measure in result.measures]
        round_result = RoundResult(
            generation=request.generation,
            designer_kind=designer_kind,
            agent_kind=agent_kind,
            designer_version_before=designer_version,
            designer_version_after=designer_after,
            agent_version_before=agent_version,
            agent_version_after=agent_after,
            designer_step=designer_step,
            agent_step=agent_step,
            measured=len(result.measures),
            refused=sum(1 for proposal in result.proposals if not proposal.is_written),
            reported_plays=reported_plays,
            mean_regret=statistics.fmean(regrets) if regrets else None,
            manifest_path=str(result.manifest_path) if result.manifest_path is not None else None,
            report_path=str(result.report_path),
            experience=result.experience,
        )
        self.written_round(round_result)
        return round_result

    def reported_held(self) -> int:
        """Report every held play of the block, both arms, to the Reasoning Agent; the number of plays reported."""
        reported = 0
        # The round is one training unit: every plain play of the block, stamped with the block's total.
        label = f"generation-{self.held[0][0]:05d}"
        total = sum(
            1
            for _, _, measures in self.held
            for measure in measures
            for play in measure.plain_plays
            if play.reward is not None and play.receipts
        )
        for generation, designer_version, measures in self.held:
            base = {"generation": generation, "designer_version": version_metadata(designer_version)}
            for measure in measures:
                plain = {**base, "arm": ARM_PLAIN, "round": label, "round_plays": total}
                reported += len(
                    self.agent_impl.report_plays(measure.plain_plays, reward=self.agent.reward, metadata=plain)
                )
                self.agent_impl.report_plays(
                    measure.hint_plays, reward=self.agent.reward, metadata={**base, "arm": ARM_HINT}
                )
        self.held.clear()
        self.rounds_in_block = 0
        return reported

    def next_request(self, request: GenerationRequest, result: RoundResult) -> GenerationRequest:
        """The next round's request: the next generation, this round's experience and its summary as previous."""
        return replace(
            request, generation=request.generation + 1, experience=result.experience, previous=result.summary()
        )

    def run(self, first: GenerationRequest, rounds: int) -> tuple[RoundResult, ...]:
        """``rounds`` rounds in a row; each generation's experience and summary feed the next."""
        if rounds < 1:
            raise RoundError("rounds must be at least 1")
        results: list[RoundResult] = []
        request = first
        for _ in range(rounds):
            result = self.round(request)
            results.append(result)
            request = self.next_request(request, result)
        return tuple(results)

    def written_round(self, result: RoundResult) -> Path:
        directory = self.tasks_root / REPORT_DIRECTORY
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ROUNDS_FILE
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.as_line(), sort_keys=True) + "\n")
        return path


def main(
    argv: Sequence[str] | None = None,
    *,
    checks: Checks | None = None,
    designer_impl: Designer | None = None,
    agent_impl: ReasoningAgent | None = None,
    designer_deployment: Deployment | None = None,
    agent_deployment: Deployment | None = None,
) -> int:
    """Run rounds from the command line and print one JSON line per round."""
    parser = argparse.ArgumentParser(
        prog="python -m recipes.beta.spade.rounds",
        description="SPADE rounds: Designer rounds, then a Reasoning Agent round, the other side fixed.",
    )
    add_generation_arguments(parser)
    parser.add_argument("--rounds", type=int, default=1, help="how many rounds to run")
    parser.add_argument(
        "--agent-round-every",
        type=int,
        default=DEFAULT_AGENT_ROUND_EVERY,
        help="the Reasoning Agent trains on the held plays after this many Designer rounds",
    )
    parser.add_argument(
        "--wait-timeout-s", type=float, default=DEFAULT_WAIT_TIMEOUT_S, help="how long to wait for a commit"
    )
    parser.add_argument(
        "--poll-s", type=float, default=DEFAULT_POLL_S, help="how often to read the status while waiting"
    )
    arguments = parser.parse_args(argv)
    try:
        designer, agent = roles_from_arguments(arguments)
        request = request_from_arguments(arguments)
        if arguments.rounds < 1:
            raise RoundError("--rounds must be at least 1")
        run = SpadeRun(
            designer,
            agent,
            checks=checks if checks is not None else RealChecks(harbor=arguments.harbor),
            tasks_root=arguments.tasks_root,
            work_dir=arguments.work_dir,
            agent_round_every=arguments.agent_round_every,
            designer_impl=designer_impl,
            agent_impl=agent_impl,
            designer_deployment=designer_deployment,
            agent_deployment=agent_deployment,
            poll_s=arguments.poll_s,
            wait_timeout_s=arguments.wait_timeout_s,
        )
        current = request
        for _ in range(arguments.rounds):
            result = run.round(current)
            print(json.dumps(result.as_line(), sort_keys=True), flush=True)
            current = run.next_request(current, result)
    except (GenerationError, RoundError, ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
