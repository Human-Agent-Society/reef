"""Run the Harbor Python process in a disposable E2B sandbox as well as its task.

The snapshot carries a pinned runtime, with a checked manifest. Only rendered
episode inputs and explicitly named environment variables cross the boundary.
Candidate code runs as an unprivileged user; it never executes on the host.
Provider credentials needed by Harbor remain accessible to the candidate.
"""

import hashlib
import json
import math
import re
import shlex
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from reef.harness.executor import EpisodeLaunchError, EpisodeTimeout, ProcessOutcome, SandboxUnavailable

from .e2b_completion import RemoteCompletion
from .e2b_transport import run_attached, transport_failure
from .sandbox_files import collect_outputs, episode_archive

REMOTE_ROOT = "/episode"
REMOTE_PYTHON = "/opt/reef/recipes/meta_harness/examples/terminal_bench/.venv/bin/python"
MANIFEST = "/opt/reef-runtime.json"
MAX_EVIDENCE_BYTES = 512 * 1024 * 1024


def collect_evidence(sandbox, root, writable_paths, remaining, diagnostics):
    """Download a bounded archive; retry reads without repeating execution."""
    diagnostics["phase"] = "evidence_archive"
    sandbox.commands.run("tar -cf /tmp/reef-output.tar -C /episode .", user="root", timeout=min(60, remaining()))
    diagnostics["phase"] = "evidence_download"
    for attempt in range(3):
        try:
            with tempfile.TemporaryFile() as evidence:
                count = 0
                with sandbox.files.read(
                    "/tmp/reef-output.tar", format="stream", user="root", request_timeout=min(60, remaining())
                ) as stream:
                    for block in stream:
                        count += len(block)
                        if count > MAX_EVIDENCE_BYTES:
                            raise EpisodeLaunchError("E2B evidence archive exceeds the transport limit")
                        evidence.write(block)
                evidence.seek(0)
                diagnostics["phase"] = "evidence_validate"
                collect_outputs(evidence, root, writable_paths, max_bytes=MAX_EVIDENCE_BYTES)
            break
        except Exception as exc:
            if diagnostics["phase"] != "evidence_download" or not transport_failure(exc) or attempt == 2:
                raise
            diagnostics["evidence_download_retries"] = attempt + 1
            time.sleep(min(attempt + 1, remaining()))
    diagnostics.update(evidence_collected=True, phase="complete")


def remember_final_cost(root, diagnostics):
    """A finished raw trial can retain its bill even when transport failed."""
    paths = list(Path(root, "terminus/trials").rglob("result.json"))
    if len(paths) != 1:
        return
    from reef.harness.terminus.trial import trial_outcome

    trial = json.loads(paths[0].read_text())
    if trial.get("finished_at") and (cost := trial_outcome(trial)["cost_usd"]) is not None:
        diagnostics["observed_cost_usd"] = cost


def retain_completion(root, writable_paths, evidence):
    sessions = Path(root, "terminus/sessions")
    if sessions.resolve() not in {Path(path).resolve() for path in writable_paths}:
        return
    for path in sessions.glob("*.json"):
        value = json.loads(path.read_text())
        value["execution_completion"] = evidence
        path.write_text(json.dumps(value))


def retain_failure_trajectory(root, env, retained_evidence):
    """Keep collected diagnostics when an exception skips the normal reader."""
    from reef.harness.trajectory import read_terminus_atif

    raw = json.dumps({"trajectory": read_terminus_atif(Path(root, "terminus/sessions"))})
    for key in ("OPENAI_API_KEY", "E2B_API_KEY", "ANTHROPIC_API_KEY"):
        if value := env.get(key):
            raw = raw.replace(str(value), "[REDACTED]")
    retained_evidence.update(json.loads(raw))


def preserve_missing_summary(argv, root, writable_paths, env, diagnostics, exit_code):
    """Keep collected Harbor evidence when the runner died before its summary.

    This does not make the trial valid or infer a bill from incomplete usage.
    It emits a diagnostic record through the normal adapter reader so Reef's
    next scenario commit can retain the partial trajectory and available bill.
    """
    if len(argv) != 3 or argv[1] != "--task":
        return
    sessions, trials = Path(root, "terminus/sessions"), Path(root, "terminus/trials")
    if list(sessions.glob("*.json")):
        return  # Never replace a runner's own summary or conceal malformed evidence.
    writable = {Path(path).resolve() for path in writable_paths}
    if sessions.resolve() not in writable or trials.resolve() not in writable:
        return
    from reef.harness.terminus.runner import trial_record, write_trial

    record = trial_record(argv[2], None, trials, "isolated runner exited without a trial summary")
    record["outcome"].update(
        valid=False,
        phase="missing_runner_summary",
        reward=None,
        error="isolated runner exited without a trial summary",
    )
    record.update(
        rewards={},
        reward=None,
        failed=True,
        runner_exit_code=exit_code,
        transport_diagnostics={**diagnostics, "summary_reconstructed": True},
    )
    # ATIF steps can contain command text; redact inherited provider secrets
    # before the diagnostic record becomes durable or proposer-visible.
    raw = json.dumps(record)
    for key in ("OPENAI_API_KEY", "E2B_API_KEY", "ANTHROPIC_API_KEY"):
        if value := env.get(key):
            raw = raw.replace(str(value), "[REDACTED]")
    write_trial(json.loads(raw), sessions)


@dataclass(frozen=True)
class E2BEpisodeExecutor:
    snapshot_id: str
    manifest_sha256: str
    verifier_compat: str | None = None

    def preflight(self):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+(?::[a-zA-Z0-9_-]+)?", self.snapshot_id):
            raise SandboxUnavailable("E2B requires a snapshot identifier from the prepared runtime")
        if not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256):
            raise SandboxUnavailable("E2B requires the SHA-256 of the prepared runtime manifest")
        if self.verifier_compat is not None:
            from .verifier_compat import PROTOCOL

            if self.verifier_compat != PROTOCOL:
                raise SandboxUnavailable("unknown verifier compatibility protocol")

    def launch(self, argv, *, root, workspace, env, timeout, writable_paths=(), readonly_paths=()):
        from e2b import Sandbox
        from e2b.exceptions import TimeoutException

        self.preflight()
        if not math.isfinite(timeout) or timeout <= 0:
            raise EpisodeLaunchError("E2B episode timeout must be finite and positive")
        if argv[0] != "reef-terminus-e2b":
            raise EpisodeLaunchError("the E2B research executor only launches the isolated Terminus runner")
        payload = episode_archive(root, writable_paths, readonly_paths)
        sandbox = None
        started = time.monotonic()
        diagnostics = {
            "protocol": "e2b-command-reconnect-v1",
            "phase": "sandbox_create",
            "sandbox_id": None,
            "evidence_collected": False,
        }
        retained_evidence = {}
        command_attempted = False

        def remaining():
            left = timeout - (time.monotonic() - started)
            if left <= 0:
                raise EpisodeTimeout("E2B harness exceeded its configured timeout")
            return left

        try:
            sandbox = Sandbox.create(
                self.snapshot_id,
                timeout=math.ceil(timeout) + 120,
                metadata={"reef_role": "meta-harness-runner"},
                secure=True,
                lifecycle={"on_timeout": "kill"},
                request_timeout=min(60, remaining()),
            )
            diagnostics.update(sandbox_id=sandbox.sandbox_id, phase="runtime_verify")
            manifest = sandbox.files.read(MANIFEST, format="bytes", user="root", request_timeout=min(60, remaining()))
            if hashlib.sha256(manifest).hexdigest() != self.manifest_sha256:
                raise EpisodeLaunchError("E2B runtime manifest differs from the frozen plan")
            sandbox.commands.run(
                shlex.join(
                    [REMOTE_PYTHON, "-m", "recipes.meta_harness.examples.terminal_bench.e2b_runtime", "--verify"]
                ),
                user="root",
                cwd="/opt/reef",
                timeout=min(60, remaining()),
            )
            sandbox.files.write("/tmp/reef-input.tar", payload, user="root", request_timeout=min(60, remaining()))
            diagnostics["phase"] = "input_prepare"
            sandbox.commands.run(
                "mkdir /episode && tar --same-owner -xf /tmp/reef-input.tar -C /episode && rm /tmp/reef-input.tar",
                user="root",
                timeout=min(60, remaining()),
            )
            remote_env = {key: str(value).replace(str(root), REMOTE_ROOT) for key, value in env.items()}
            remote_env.pop("REEF_TERMINUS_VERIFIER_COMPAT", None)
            if self.verifier_compat:
                remote_env["REEF_TERMINUS_VERIFIER_COMPAT"] = self.verifier_compat
            remote_env.update(
                HOME="/episode/workspace",
                TMPDIR="/tmp",
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH="/opt/reef",
                PATH=str(Path(REMOTE_PYTHON).parent) + ":/usr/local/bin:/usr/bin:/bin",
            )
            command_argv = [
                REMOTE_PYTHON,
                "-m",
                "recipes.meta_harness.examples.terminal_bench.isolated_runner",
                *argv[1:],
            ]
            completion = RemoteCompletion(
                sandbox,
                REMOTE_PYTHON,
                command_argv,
                str(workspace).replace(str(root), REMOTE_ROOT),
                hashlib.sha256(payload).hexdigest(),
                remaining,
            )
            command_attempted = True
            result = run_attached(
                sandbox.commands,
                completion.command(),
                user="root",
                cwd=REMOTE_ROOT,
                envs=remote_env,
                remaining=remaining,
                diagnostics=diagnostics,
                recover_finished=completion.read_finished,
            )
            completed = completion.read_finished(diagnostics["command_pid"])
            if completed is None or completed.exit_code != result.exit_code:
                raise EpisodeLaunchError("runner end event differs from its protected completion receipt")
            result = completed
            collect_evidence(sandbox, root, writable_paths, remaining, diagnostics)
            preserve_missing_summary(argv, root, writable_paths, env, diagnostics, result.exit_code)
            retain_completion(
                root, writable_paths, completion.evidence(end_received=diagnostics["command_end_received"])
            )
            from .terminal_evidence import preserve_terminal_evidence

            preserve_terminal_evidence(root, writable_paths, env)
            remember_final_cost(root, diagnostics)
            return ProcessOutcome(result.exit_code, result.stdout, result.stderr)
        except (TimeoutException, TimeoutError) as exc:
            failure = EpisodeTimeout("E2B harness timed out; usage may be unknown")
            diagnostics["error_type"] = type(exc).__name__
            failure.e2b_diagnostics = diagnostics
            failure.e2b_retained_evidence = retained_evidence
            raise failure from exc
        except (EpisodeTimeout, EpisodeLaunchError) as exc:
            diagnostics["error_type"] = type(exc).__name__
            exc.e2b_diagnostics = diagnostics
            exc.e2b_retained_evidence = retained_evidence
            raise
        except Exception as exc:
            # SDK exception strings can include HTTP request details. Retain
            # the exception type here, never provider credentials or bodies.
            failure = EpisodeLaunchError(f"E2B harness launch or evidence collection failed ({type(exc).__name__})")
            diagnostics["error_type"] = type(exc).__name__
            failure.e2b_diagnostics = diagnostics
            failure.e2b_retained_evidence = retained_evidence
            raise failure from exc
        finally:
            if sandbox is not None:
                if (
                    command_attempted
                    and not diagnostics["evidence_collected"]
                    and diagnostics["phase"]
                    in ("command_start", "command_wait", "command_reconnect", "command_complete")
                ):
                    # Retain what exists before destroying the owned runner.
                    # This snapshot is invalid diagnostic evidence, never a
                    # fabricated completion or an estimate of partial usage.
                    capture = {}
                    capture_deadline = time.monotonic() + 30

                    def capture_remaining():
                        left = min(remaining(), capture_deadline - time.monotonic())
                        if left <= 0:
                            raise EpisodeTimeout("failure evidence capture deadline")
                        return left

                    try:
                        collect_evidence(sandbox, root, writable_paths, capture_remaining, capture)
                        diagnostics["evidence_collected"] = True
                        remember_final_cost(root, diagnostics)
                        preserve_missing_summary(argv, root, writable_paths, env, diagnostics, None)
                        retain_failure_trajectory(root, env, retained_evidence)
                    except Exception as capture_error:
                        capture["error_type"] = type(capture_error).__name__
                    diagnostics["failure_evidence_capture"] = capture
                try:
                    sandbox.kill(request_timeout=30)
                    diagnostics["cleanup_confirmed"] = True
                except Exception as exc:
                    # Do not silently consider an episode complete if cleanup
                    # is unconfirmed. Its provider lifetime remains bounded.
                    failure = EpisodeLaunchError(
                        f"E2B runner cleanup is unconfirmed ({type(exc).__name__}); sandbox {sandbox.sandbox_id}"
                    )
                    diagnostics.update(cleanup_confirmed=False, cleanup_error_type=type(exc).__name__)
                    if diagnostics["evidence_collected"] and not retained_evidence:
                        try:
                            retain_failure_trajectory(root, env, retained_evidence)
                        except Exception as evidence_error:
                            diagnostics["retained_trajectory_error_type"] = type(evidence_error).__name__
                    failure.e2b_diagnostics = diagnostics
                    failure.e2b_retained_evidence = retained_evidence
                    raise failure from exc
