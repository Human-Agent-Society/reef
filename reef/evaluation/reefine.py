"""Independent Reefine request, protected-task and model review checks.

This plugin alone selects a candidate. Proposer design notes and trial scores
are not evaluation inputs. Each model call starts with a fresh message list.
The plan reads the harness description, so requested behavior the harness
cannot deliver is reported as a limit instead of being run and failed; a
candidate that passes with limits waits for the person's review.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from reef.core.evaluation import (
    CandidateEvaluator,
    EvaluationResult,
    SelectionDecision,
    ServedComposition,
    UpdateCandidate,
)
from reef.core.plugins import ConfiguredPlugin, PluginConfigError
from reef.core.prompts import untrusted_text
from reef.core.training_request import TrainingRequest
from reef.evaluation.evaluators import SelectionPlugin
from reef.evaluation.harness import HarnessCandidate, verifier_reward
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.client.check_display import check_label
from reef.harness.episodes.model_binding import ModelBinding, ModelBindingError, ModelBindings
from reef.harness.episodes.run import EpisodeError, EpisodeResult, TrajectoryKeepError, run_episode
from reef.harness.episodes.trajectory import TrajectoryError, final_assistant_text, reader_for
from reef.harness.step_result import evaluation_report
from reef.sandbox.process import EpisodeExecutor, EpisodeTimeout


@dataclass(frozen=True)
class EvaluationContext:
    request: TrainingRequest | None
    current: ServedComposition


@dataclass(frozen=True, kw_only=True)
class ReefineCandidate(HarnessCandidate):
    context: EvaluationContext
    requires: tuple[Mapping[str, object], ...] = ()
    #: The change's How to use note, which tells a person how to type its commands. The plan may take a command form
    #: from it for a step the request does not name; requirements come from the request alone.
    usage: str = ""


class ReefineEvaluationBackend(CandidateEvaluator):
    """Episode inputs and progress reporting for independent Reefine evaluation."""

    episode_timeout_seconds: float

    @property
    @abstractmethod
    def evaluation_models(self) -> ModelBindings: ...

    @property
    @abstractmethod
    def episode_descriptor(self) -> AdapterDescriptor: ...

    @property
    @abstractmethod
    def episode_binary(self) -> str: ...

    @property
    @abstractmethod
    def episode_executor(self) -> EpisodeExecutor: ...

    @property
    @abstractmethod
    def harness_description(self) -> Mapping[str, object]:
        """What a change to the evaluated harness may contain and what the harness offers, for the plan."""

    @abstractmethod
    def episode_files(self, entries: tuple[Mapping[str, object], ...]) -> dict[str, str]: ...

    @abstractmethod
    def show_evaluation_checks(self, checks: tuple[Mapping[str, object], ...]) -> None: ...


#: How much of the first failed check's reason a rejection repeats; the page shows every reason in full.
REASON_CHARACTERS = 300
#: ``concern`` is a change review that found a problem it does not call unsafe: the release waits for the person.
CheckStatus = Literal["pending", "running", "pass", "fail", "invalid", "not_run", "unsupported", "untested", "concern"]


@dataclass(frozen=True)
class CheckResult:
    id: str
    group: str
    target: str
    status: CheckStatus = "pending"
    expected: str = ""
    observed: str = ""
    reason: str = ""
    score: float | None = None
    current_score: float | None = None
    seconds: float = 0.0
    record: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool) or not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("request verification requires a boolean passed and a nonempty reason")


class RequestVerifier(ConfiguredPlugin, ABC):
    """Optional operator-owned validation, outside the candidate's writable workspace."""

    @abstractmethod
    def verify(self, result: EpisodeResult, workspace: Path) -> VerificationResult: ...


@dataclass(frozen=True)
class RequestContext:
    prompt: str = ""
    files: Mapping[str, str] | None = None
    initialize_git: bool = False
    verifier: RequestVerifier | None = None
    version: str = "1"


@dataclass(frozen=True)
class EvaluationInstructions:
    plan: str
    behavior_review: str
    change_review: str


@dataclass(frozen=True)
class ReefineEvaluationSettings:
    reviewer_model: str = "served"
    request_context: RequestContext = RequestContext()
    protected_tasks: tuple[str, ...] = ()
    timeout_seconds: float = 900.0
    max_output_tokens: int = 8192
    scorer_version: str = "1"
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    #: Extra request fields for the evaluation's own model calls, in the reviewer endpoint's dialect, such as a
    #: reasoning effort.
    reviewer_params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("evaluation.timeout_s must be positive and finite")
        if self.max_output_tokens < 256:
            raise ValueError("evaluation.max_output_tokens must be at least 256")


class ReefineEvaluation(SelectionPlugin):
    def __init__(
        self,
        backend: ReefineEvaluationBackend,
        settings: ReefineEvaluationSettings,
        instructions: EvaluationInstructions,
    ):
        self.backend = backend
        self.settings = settings
        self.instructions = instructions

    @classmethod
    def from_config(cls, config: Mapping[str, object], *, candidate_backend: CandidateEvaluator) -> ReefineEvaluation:
        if not isinstance(candidate_backend, ReefineEvaluationBackend):
            raise PluginConfigError("Reefine evaluation requires a ReefineEvaluationBackend")
        settings = config.get("settings")
        instructions = config.get("instructions")
        if (
            set(config) != {"settings", "instructions"}
            or not isinstance(settings, ReefineEvaluationSettings)
            or not isinstance(instructions, EvaluationInstructions)
        ):
            raise PluginConfigError("Reefine evaluation requires parsed settings and instructions")
        return cls(candidate_backend, settings, instructions)

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        if not isinstance(candidate, ReefineCandidate):
            raise TypeError("Reefine evaluation requires a ReefineCandidate")
        started = time.monotonic()
        deadline = started + self.settings.timeout_seconds
        models = self.backend.evaluation_models
        reviewer = models[self.settings.reviewer_model]
        checks = [
            CheckResult("request-plan", "request", "plan", expected="tasks that show the requested behavior"),
            CheckResult("request-behavior", "request", "candidate", expected="all requested behavior"),
        ]
        for index, task in enumerate(self.settings.protected_tasks):
            checks.append(CheckResult(f"regression-{index}", "regression", "candidate/current", expected=task))
        if not self.settings.protected_tasks:
            checks.append(
                CheckResult(
                    "regression-missing",
                    "regression",
                    "candidate/current",
                    "not_run",
                    reason="no protected tasks configured",
                )
            )
        checks.append(
            CheckResult(
                "change-review", "review", "candidate", expected="request implemented; no unsafe or unrelated change"
            )
        )
        self.update_progress(checks)
        root = candidate.record_dir
        with TemporaryDirectory(prefix="reefine-evaluation-") as temporary:
            records = Path(temporary) if root is None else root / "reefine-evaluation"
            records.mkdir(parents=True, exist_ok=True)
            # Plan from the original instruction, operator context and the harness description, before viewing any
            # candidate content.
            request = candidate.context.request
            plan_index = 0
            plan = None
            tasks: list[dict[str, list[str]]] = []
            limits: list[dict[str, str]] = []
            untested: list[dict[str, str]] = []
            if request is None:
                checks[plan_index] = replace(
                    checks[plan_index], status="invalid", reason="no original training request"
                )
            else:
                self.begin(checks, plan_index)
                plan_started = time.monotonic()
                try:
                    context = self.settings.request_context
                    harness = dict(self.backend.harness_description)
                    payload: dict[str, object] = {
                        "request": request.text,
                        "prompt": context.prompt,
                        "files": dict(context.files or {}),
                        "harness": harness,
                        "how_to_use": candidate.usage,
                    }
                    replies = []
                    # A plan that breaks its limits is asked for once more, with the problem, before it counts.
                    for attempt in range(2):
                        reply = self.ask_json(reviewer, self.instructions.plan, payload, deadline)
                        replies.append(reply)
                        (records / "plan-replies.json").write_text(
                            json.dumps(replies, indent=2) + "\n", encoding="utf-8"
                        )
                        try:
                            tasks, limits, untested = validate_plan(reply)
                            break
                        except ValueError as exc:
                            if attempt == 1:
                                raise
                            payload = {**payload, "previous_plan_problem": str(exc)}
                    if context.prompt and tasks:
                        # The operator's prompt replaces the generated messages and keeps every generated check.
                        tasks = [
                            {
                                "messages": [context.prompt],
                                "checks": [item for task in tasks for item in task["checks"]],
                            }
                        ]
                    plan = {"tasks": tasks, "limits": limits, "untested": untested}
                    # The plan row says what was planned; only the task rows below observe behavior.
                    checks[plan_index] = replace(
                        checks[plan_index],
                        status="pass",
                        observed=(
                            f"{len(tasks)} {'task' if len(tasks) == 1 else 'tasks'} planned, {len(limits)} not "
                            f"possible here, {len(untested)} not tested here; this row observes no behavior"
                        ),
                        seconds=time.monotonic() - plan_started,
                    )
                    # Behavior the harness cannot deliver, or that no task can show here, is reported and
                    # never run; each task runs in its own fresh session.
                    checks[plan_index + 1 : plan_index + 2] = [
                        *(
                            CheckResult(
                                f"limit-{index}",
                                "limit",
                                "harness",
                                "unsupported",
                                expected=item["requirement"],
                                reason=item["reason"],
                            )
                            for index, item in enumerate(limits)
                        ),
                        *(
                            CheckResult(
                                f"untested-{index}",
                                "untested",
                                "evaluation",
                                "untested",
                                expected=item["requirement"],
                                reason=item["reason"],
                            )
                            for index, item in enumerate(untested)
                        ),
                        *(
                            CheckResult(
                                f"request-behavior-{index}",
                                "request",
                                "candidate",
                                expected="; ".join(task["checks"]),
                            )
                            for index, task in enumerate(tasks)
                        ),
                    ] or [checks[plan_index + 1]]
                except (ValueError, ModelBindingError) as exc:
                    checks[plan_index] = replace(
                        checks[plan_index], status="invalid", reason=str(exc), seconds=time.monotonic() - plan_started
                    )
            self.update_progress(checks)
            if plan is None or request is None:
                index = next(i for i, check in enumerate(checks) if check.id == "request-behavior")
                checks[index] = replace(checks[index], status="not_run", reason="request plan unavailable")
            elif not tasks:
                checks.insert(
                    next(i for i, check in enumerate(checks) if check.group == "regression"),
                    CheckResult(
                        "request-behavior",
                        "request",
                        "candidate",
                        "not_run",
                        expected="all requested behavior",
                        reason="no requested behavior can be tested here; see the limits and untested items",
                    ),
                )
            else:
                context = self.settings.request_context
                descriptor = self.backend.episode_descriptor
                for task_index, planned in enumerate(tasks):
                    check_index = next(
                        i for i, check in enumerate(checks) if check.id == f"request-behavior-{task_index}"
                    )
                    if len(planned["messages"]) > 1 and not (
                        descriptor.follow_up_messages or descriptor.continue_argv
                    ):
                        # The adapter cannot continue a session, so the task is shown as not run and waits for review.
                        checks[check_index] = replace(
                            checks[check_index],
                            status="untested",
                            reason=f"this task sends {len(planned['messages'])} messages in one session, and the "
                            f"{descriptor.name} adapter sends one message per session",
                        )
                        self.update_progress(checks)
                        continue
                    self.begin(checks, check_index)
                    before = time.monotonic()
                    record_dir = records / f"request-{task_index}"
                    try:
                        result = self.episode(
                            candidate,
                            "candidate",
                            planned["messages"],
                            context.files or {},
                            context.initialize_git,
                            record_dir,
                            deadline,
                        )
                        passed, reason = self.judge_episode(
                            reviewer,
                            request.text,
                            {**planned, "limits": limits, "untested": untested},
                            result,
                            candidate.candidate_files,
                            candidate.requires,
                            deadline,
                            git_repository=context.initialize_git,
                        )
                        if context.verifier is not None:
                            verified = context.verifier.verify(result, record_dir / "workspace")
                            passed = passed and verified.passed
                            reason = reason + "; " + verified.reason
                        checks[check_index] = replace(
                            checks[check_index],
                            status="pass" if passed else "fail",
                            score=float(passed),
                            observed=reason,
                            reason="" if passed else reason,
                            seconds=time.monotonic() - before,
                            record=str(record_dir) if root else None,
                        )
                    except (
                        EpisodeError,
                        TrajectoryError,
                        TrajectoryKeepError,
                        ValueError,
                        ModelBindingError,
                        OSError,
                    ) as exc:
                        if isinstance(exc.__cause__, EpisodeTimeout):
                            # The candidate's session did not finish: a failure of the change. The review reads the
                            # record kept so far, so the reason can say why, such as agents starting each other.
                            seconds = time.monotonic() - before
                            reason = f"The task did not finish: the session was still running after {seconds:.0f} s."
                            try:
                                kept = reader_for(descriptor.trajectory_format)(record_dir)
                                if kept:
                                    reason = self.judge_episode(
                                        reviewer,
                                        request.text,
                                        {**planned, "limits": limits, "untested": untested},
                                        EpisodeResult(0, "", "", kept, ()),
                                        candidate.candidate_files,
                                        candidate.requires,
                                        deadline,
                                        git_repository=context.initialize_git,
                                        unfinished_after_seconds=seconds,
                                    )[1]
                            except (TrajectoryError, ValueError, ModelBindingError, OSError):
                                pass
                            checks[check_index] = replace(
                                checks[check_index],
                                status="fail",
                                score=0.0,
                                reason=reason,
                                seconds=time.monotonic() - before,
                            )
                        else:
                            checks[check_index] = replace(
                                checks[check_index],
                                status="invalid",
                                reason=str(exc),
                                seconds=time.monotonic() - before,
                            )
                    self.update_progress(checks)
            self.update_progress(checks)
            for index, task in enumerate(self.settings.protected_tasks):
                check_index = next(i for i, check in enumerate(checks) if check.id == f"regression-{index}")
                self.begin(checks, check_index)
                before = time.monotonic()
                scores = {}
                try:
                    # Matched prompt, workspace, model and budget; fresh sessions on both sides.
                    for side in ("current", "candidate"):
                        checks[check_index] = replace(checks[check_index], observed=f"running {side} episode")
                        self.update_progress(checks)
                        result = self.episode(
                            candidate,
                            side,
                            [task],
                            self.settings.request_context.files or {},
                            False,
                            records / f"regression-{index}-{side}",
                            deadline,
                        )
                        passed, reason = self.judge_episode(
                            reviewer,
                            task,
                            {"messages": [task], "checks": [task]},
                            result,
                            candidate.current_files if side == "current" else candidate.candidate_files,
                            (),
                            deadline,
                            git_repository=False,
                        )
                        scores[side] = float(passed)
                    passed = scores["candidate"] >= scores["current"] and scores["candidate"] == 1.0
                    checks[check_index] = replace(
                        checks[check_index],
                        status="pass" if passed else "fail",
                        score=scores["candidate"],
                        current_score=scores["current"],
                        observed=f"current {scores['current']:g}; candidate {scores['candidate']:g}",
                        reason="" if passed else "protected capability failed or regressed",
                        seconds=time.monotonic() - before,
                    )
                except (EpisodeError, TrajectoryError, TrajectoryKeepError, ValueError, ModelBindingError) as exc:
                    checks[check_index] = replace(
                        checks[check_index], status="invalid", reason=str(exc), seconds=time.monotonic() - before
                    )
                self.update_progress(checks)
            review_index = len(checks) - 1
            self.begin(checks, review_index)
            before = time.monotonic()
            try:
                change_review = self.ask_json(
                    reviewer,
                    self.instructions.change_review,
                    {
                        "request": "" if request is None else request.text,
                        "serving_release_files": candidate.current_files,
                        "candidate_files": candidate.candidate_files,
                        "how_to_use": candidate.usage,
                        "harness": dict(self.backend.harness_description),
                        "requires": list(candidate.requires),
                        "checks": [
                            {"label": check_label(check.to_dict()), **check.to_dict()} for check in checks[:-1]
                        ],
                    },
                    deadline,
                )
                passed, reason = validate_review(change_review)
                unsafe = change_review.get("unsafe")
                if not passed and not isinstance(unsafe, bool):
                    raise ValueError("a failed change review requires a boolean unsafe")
                # A problem the review does not call unsafe holds a release whose behavior passed for the person.
                checks[review_index] = replace(
                    checks[review_index],
                    status="pass" if passed else "fail" if unsafe else "concern",
                    score=float(passed),
                    observed=reason,
                    reason="" if passed else reason,
                    seconds=time.monotonic() - before,
                )
            except (ValueError, ModelBindingError) as exc:
                checks[review_index] = replace(
                    checks[review_index], status="invalid", reason=str(exc), seconds=time.monotonic() - before
                )
            self.update_progress(checks)
            same_model = reviewer.model == models.served.model
            report = {
                "version": "1",
                "current_release_id": candidate.context.current.release_id,
                "request_id": None if request is None else request.id,
                "model": models.served.model,
                "reviewer_model": reviewer.model,
                "reviewer_binding": self.settings.reviewer_model,
                "reviewer_context": "fresh",
                "same_model": same_model,
                "adapter": self.backend.episode_descriptor.name,
                "requires": list(candidate.requires),
                "task_version": self.settings.request_context.version,
                "scorer_version": self.settings.scorer_version,
                "protected_tasks_checksum": hashlib.sha256(
                    json.dumps(self.settings.protected_tasks).encode()
                ).hexdigest(),
                "plan": plan,
                "checks": [check.to_dict() for check in checks],
                "seconds": round(time.monotonic() - started, 3),
            }
            if root is not None:
                (records / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return EvaluationResult("reefine", "1", {"evaluation_report": report, "evaluation_sides": ["candidate"]})

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        report = evaluation_report(evaluation.metrics)
        checks = report.get("checks") if isinstance(report, Mapping) else None
        failures = []
        limits = []
        untested = []
        # Rows reported but never run carry their own status; every other row must pass.
        reported_status = {"limit": "unsupported", "untested": "untested"}
        skipped = []
        concerns = []
        if not isinstance(checks, list) or not checks:
            failures.append("evaluation returned no checks")
        else:
            groups = {check.get("group") for check in checks if isinstance(check, Mapping)}
            if not {"request", "regression", "review"} <= groups or groups - {
                "request",
                "regression",
                "review",
                *reported_status,
            }:
                failures.append("evaluation is missing a required check group")
            limits = [check for check in checks if isinstance(check, Mapping) and check.get("group") == "limit"]
            untested = [check for check in checks if isinstance(check, Mapping) and check.get("group") == "untested"]
            # A change review that found a problem it does not call unsafe holds the release for review.
            concerns = [
                check
                for check in checks
                if isinstance(check, Mapping) and check.get("group") == "review" and check.get("status") == "concern"
            ]
            # A request task the adapter cannot run is reported as untested and holds the release for review.
            skipped = [
                check
                for check in checks
                if isinstance(check, Mapping) and check.get("group") == "request" and check.get("status") == "untested"
            ]
            if isinstance(candidate, ReefineCandidate):
                plan = report.get("plan") if isinstance(report, Mapping) else None
                tasks = plan.get("tasks") if isinstance(plan, Mapping) else None
                expected_ids = {"request-plan", "change-review"}
                if isinstance(tasks, list) and tasks:
                    expected_ids.update(f"request-behavior-{index}" for index in range(len(tasks)))
                else:
                    expected_ids.add("request-behavior")
                expected_ids.update(f"regression-{index}" for index in range(len(self.settings.protected_tasks)))
                expected_ids.update(f"limit-{index}" for index in range(len(limits)))
                expected_ids.update(f"untested-{index}" for index in range(len(untested)))
                ids = [check.get("id") for check in checks if isinstance(check, Mapping)]
                if set(ids) != expected_ids or len(ids) != len(expected_ids):
                    failures.append("evaluation is missing required checks or contains duplicate checks")
                request_id = None if candidate.context.request is None else candidate.context.request.id
                if isinstance(report, Mapping) and (
                    report.get("request_id") != request_id
                    or report.get("current_release_id") != candidate.context.current.release_id
                ):
                    failures.append("evaluation does not match this request and serving release")
            if not all(isinstance(check, Mapping) for check in checks):
                failures.append("evaluation contains a malformed check")
            failed = [
                check
                for check in checks
                if isinstance(check, Mapping)
                and check.get("status") != reported_status.get(str(check.get("group")), "pass")
                and check not in skipped
                and check not in concerns
            ]
            if failed:
                # The page shows every check's full reason; the decision names the failed checks and the first cause.
                first = str(failed[0].get("reason") or failed[0].get("status")).strip()
                if len(first) > REASON_CHARACTERS:
                    cut = first[:REASON_CHARACTERS]
                    if cut.rfind(". ") > REASON_CHARACTERS // 3:
                        first = cut[: cut.rfind(". ") + 1]
                    else:
                        first = cut.rsplit(" ", 1)[0] + "... (the page has the full reason)"
                failures.append(
                    f"Failed: {', '.join(check_label(check) for check in failed)}. "
                    f"{check_label(failed[0])}: {first.rstrip('.')}"
                )
        selected = not failures
        if not selected:
            reason = ". ".join(failures) + "."
        else:
            reason = "All checks that ran passed." if skipped or concerns else "All required checks passed."
            if limits:
                reason += " Part of the request is not possible on this harness: " + "; ".join(
                    str(check.get("expected")).strip().rstrip(".") for check in limits
                )
                reason += "."
            for check in skipped:
                reason += f" {check_label(check)} was not run: {str(check.get('reason')).strip().rstrip('.')}."
            for check in concerns:
                reason += f" The change review asks for your judgment: {str(check.get('reason')).strip().rstrip('.')}."
            if limits or skipped or concerns:
                reason += " Review the release before you install it."
        if selected and untested:
            reason += (
                " Not tested here: "
                + "; ".join(str(check.get("expected")).strip().rstrip(".") for check in untested)
                + "."
            )
        feedback = []
        if isinstance(checks, list):
            feedback = [
                {
                    "id": check.get("id"),
                    "status": check.get("status"),
                    "reason": check.get("reason") or check.get("observed"),
                }
                for check in checks
                if isinstance(check, Mapping)
                and check.get("group") in ("request", "limit")
                and check.get("status") != "pass"
            ]
        return SelectionDecision(
            selected=selected,
            policy="reefine",
            reason=reason,
            evaluation=evaluation,
            metrics={"request_feedback": feedback},
            requires_review=selected and bool(limits or skipped or concerns),
        )

    def begin(self, checks: list[CheckResult], index: int) -> None:
        checks[index] = replace(checks[index], status="running")
        self.update_progress(checks)

    def update_progress(self, checks: list[CheckResult]) -> None:
        self.backend.show_evaluation_checks(tuple(check.to_dict() for check in checks))

    def ask_json(
        self, model: ModelBinding, instructions: str, payload: Mapping[str, object], deadline: float
    ) -> dict[str, object]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("evaluation time budget exhausted")
        # No proposer messages, design, review notes or trial results are carried into this conversation.
        messages = [
            {
                "role": "user",
                "content": instructions + "\n" + untrusted_text(json.dumps(payload), "evaluation inputs"),
            }
        ]
        # A transient transport failure, or a reply that is not valid JSON, is asked once more in a fresh conversation
        # within the original deadline; a well-formed answer is never asked again.
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("evaluation time budget exhausted")
            try:
                reply = model.chat(
                    messages,
                    timeout_s=remaining,
                    max_tokens=self.settings.max_output_tokens,
                    **self.settings.reviewer_params,
                )
            except ModelBindingError as exc:
                if attempt == 1 or (exc.status is not None and exc.status not in {408, 429, 500, 502, 503, 504}):
                    raise
                continue
            if reply.strip().startswith("```"):
                reply = "\n".join(reply.strip().splitlines()[1:-1])
            try:
                parsed = json.loads(reply)
            except json.JSONDecodeError:
                if attempt == 1:
                    raise
                continue
            if not isinstance(parsed, dict):
                raise ValueError("reviewer must return a JSON object")
            return parsed
        raise ValueError("the reviewer gave no usable reply")

    def episode(
        self,
        candidate: ReefineCandidate,
        side: str,
        messages: list[str],
        files: Mapping[str, str],
        initialize_git: bool,
        record_dir: Path,
        deadline: float,
    ) -> EpisodeResult:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("evaluation time budget exhausted")
        entries = candidate.candidate_entries if side == "candidate" else candidate.current_entries
        # Real tasks run tools that cache under the episode's home; the root is removed after the run, so the review
        # reads what was written outside the workspace instead of the run being refused for it.
        prompt, *follow_ups = messages
        return run_episode(
            self.backend.episode_descriptor,
            self.backend.episode_files(entries),
            prompt,
            binary=self.backend.episode_binary,
            timeout=min(remaining, self.backend.episode_timeout_seconds),
            executor=self.backend.episode_executor,
            keep_dir=record_dir,
            workspace_files=files,
            initialize_git=initialize_git,
            keep_workspace=True,
            online=not prompt.startswith("[health]"),
            task_environment=self.settings.environment,
            follow_ups=tuple(follow_ups),
        )

    def judge_episode(
        self,
        model: ModelBinding,
        request: str,
        plan: Mapping[str, object],
        result: EpisodeResult,
        harness_files: Mapping[str, str],
        requires: tuple[Mapping[str, object], ...],
        deadline: float,
        *,
        git_repository: bool,
        unfinished_after_seconds: float | None = None,
    ) -> tuple[bool, str]:
        if result.exit_code != 0:
            return False, f"harness exited {result.exit_code}: {result.stderr[-500:]}"
        if not result.trajectory:
            raise ValueError("episode produced no trajectory")
        review = self.ask_json(
            model,
            self.instructions.behavior_review,
            {
                "request": request,
                "plan": plan,
                "harness": harness_files,
                "requires": list(requires),
                "trajectory": shortened_trajectory(result.trajectory),
                "stderr": result.stderr[-2000:],
                "files_outside_workspace": list(result.residue[:50]),
                "workspace_is_git_repository": git_repository,
                **(
                    {}
                    if unfinished_after_seconds is None
                    else {"unfinished_after_seconds": round(unfinished_after_seconds)}
                ),
            },
            deadline,
        )
        return validate_review(review)


HEALTH_TASK_DIRECTORY = str(Path(__file__).parents[1] / "recipe" / "reefine" / "health")


def evaluate_health(task: str, result: EpisodeResult) -> float:
    """Score the bundled health prompt or its Harbor task directory."""
    if task == HEALTH_TASK_DIRECTORY:
        return verifier_reward(task, result)
    return grade_health_text(task, final_assistant_text(result.trajectory))


def grade_health_text(task: str, text: str | None) -> float:
    """Match reef-ok on the final nonempty line of a health-task response."""
    if not task.startswith("[health]") or text is None:
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return 1.0 if lines and lines[-1] == "reef-ok" else 0.0


def validate_plan(
    value: Mapping[str, object],
) -> tuple[list[dict[str, list[str]]], list[dict[str, str]], list[dict[str, str]]]:
    """The plan's tasks (each the messages of one fresh session and the checks they show), limits and untested items."""
    tasks = value.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("request plan needs a list of tasks")
    if len(tasks) > 5:
        raise ValueError(f"request plan has {len(tasks)} tasks; at most 5 are allowed")
    parsed_tasks = []
    for task in tasks:
        messages = task.get("messages") if isinstance(task, Mapping) else None
        checks = task.get("checks") if isinstance(task, Mapping) else None
        if (
            not isinstance(messages, list)
            or not messages
            or any(not isinstance(message, str) or not message.strip() or len(message) > 4000 for message in messages)
        ):
            raise ValueError("each request task needs at least one nonempty message of at most 4000 characters")
        if (
            not isinstance(checks, list)
            or not 1 <= len(checks) <= 20
            or any(not isinstance(check, str) or not check.strip() for check in checks)
        ):
            raise ValueError("each request task needs 1 to 20 observable checks")
        parsed_tasks.append({"messages": list(messages), "checks": list(checks)})
    reported = []
    for key in ("limits", "untested"):
        items = value.get(key, [])
        if (
            not isinstance(items, list)
            or len(items) > 10
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"requirement", "reason"}
                or any(not isinstance(text, str) or not text.strip() for text in item.values())
                for item in items
            )
        ):
            raise ValueError(f"request plan {key} must be at most 10 objects with a requirement and a reason")
        reported.append([{"requirement": item["requirement"], "reason": item["reason"]} for item in items])
    if not parsed_tasks and not any(reported):
        raise ValueError("request plan needs at least one task, limit or untested item")
    return parsed_tasks, reported[0], reported[1]


#: Characters of trajectory text a reviewer reads at most.
TRAJECTORY_KEPT_CHARACTERS = 400_000
#: Base64-style data without spaces that mixes digits and both letter cases, such as an encoded page image.
BINARY_TEXT = re.compile(r"(?=[^\s]*\d)(?=[^\s]*[a-z])(?=[^\s]*[A-Z])[A-Za-z0-9+/=]{2000,}")


def text_leaves(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in text_leaves(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in text_leaves(item)]
    return []


def shortened_value(value: object, cap: int | None) -> object:
    """Binary data such as a page image becomes a note a reviewer can read; with a ``cap``, a longer text keeps
    its start and end."""
    if isinstance(value, str):
        if BINARY_TEXT.fullmatch(value):
            return (
                f"[binary content, such as a page image, that the tool returned to the agent: {len(value)} characters]"
            )
        if cap is None or len(value) <= cap:
            return value
        head, tail = value[: cap * 3 // 4], value[-(cap // 4) :]
        return f"{head}\n[... {len(value) - len(head) - len(tail)} characters omitted ...]\n{tail}"
    if isinstance(value, Mapping):
        return {key: shortened_value(item, cap) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [shortened_value(item, cap) for item in value]
    return value


def shortened_trajectory(trajectory: Sequence[Mapping[str, object]]) -> list[object]:
    """The trajectory a reviewer reads. Binary data is always replaced by a note. Only when the rest is longer than
    a reviewer should read are texts cut, the longest first, to the largest length that fits; short texts such as
    the agent's own answer stay whole. Middle events are dropped only when cutting is not enough."""
    events = [shortened_value(event, None) for event in trajectory]
    lengths = sorted((len(text) for event in events for text in text_leaves(event)), reverse=True)
    if sum(lengths) > TRAJECTORY_KEPT_CHARACTERS:
        low, high = 1000, lengths[0]
        while low < high:
            middle = (low + high + 1) // 2
            if sum(min(length, middle) for length in lengths) <= TRAJECTORY_KEPT_CHARACTERS:
                low = middle
            else:
                high = middle - 1
        events = [shortened_value(event, low) for event in events]
    sizes = [len(json.dumps(event)) for event in events]
    if sum(sizes) <= TRAJECTORY_KEPT_CHARACTERS * 1.2:
        return events
    head: list[object] = []
    tail: list[object] = []
    first, last, budget, from_head = 0, len(events) - 1, TRAJECTORY_KEPT_CHARACTERS, True
    while first <= last:
        index = first if from_head else last
        if sizes[index] > budget:
            break
        budget -= sizes[index]
        if from_head:
            head.append(events[first])
            first += 1
        else:
            tail.insert(0, events[last])
            last -= 1
        from_head = not from_head
    dropped = last - first + 1
    return [*head, {"omitted_events": dropped}, *tail] if dropped > 0 else [*head, *tail]


def validate_review(value: Mapping[str, object]) -> tuple[bool, str]:
    passed = value.get("passed")
    reason = value.get("reason")
    if not isinstance(passed, bool) or not isinstance(reason, str) or not reason.strip():
        raise ValueError("review requires a boolean passed and a nonempty reason")
    return passed, reason[:4000]
