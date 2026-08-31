"""Qwen3-8B TTT-Discover LoRA experiment runner for a single GPU node."""

from __future__ import annotations

import json
import logging
import math
import os
import secrets
import subprocess
import sys
import time
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-8B"


def _training_scenario_name(run_name: str) -> str:
    """Return the stable Reef training lane for an experiment run."""
    return f"erdos-{run_name}-session-0"


def project_root() -> Path:
    """Resolve the repository root in a checkout or the Docker image."""
    source_path = Path(__file__).resolve()
    for candidate in source_path.parents:
        if (candidate / "docker" / "Dockerfile.reef").is_file():
            return candidate
    return Path("/workspace/Reef")


def run_experiment(
    *,
    gpu_count: int,
    tensor_parallel_size: int = 2,
    state_root: Path,
    model_path: Path | None = None,
    groups: int = 8,
    rollouts: int = 64,
    phase1_max_tokens: int = 26_000,
    sequence_length: int = 30_000,
    program_budget_s: float = 30,
    eval_timeout_s: float | None = None,
    eval_cpus_per_task: int = 1,
    program_executor: Callable[..., Any] | None = None,
    lora_rank: int = 32,
    steps: int = 1,
    training_horizon_steps: int | None = None,
    require_nonconstant_step: bool = False,
    run_name: str = "",
    resume: bool = False,
) -> dict[str, Any]:
    """Run consecutive generation → optimizer → weight-sync TTTD steps."""
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    if gpu_count % tensor_parallel_size != 0:
        raise ValueError("gpu_count must be divisible by tensor_parallel_size")
    if groups < 1:
        raise ValueError("groups must be positive")
    if rollouts < 2:
        raise ValueError("rollouts must be at least two")
    if phase1_max_tokens < 1 or sequence_length <= phase1_max_tokens:
        raise ValueError("sequence_length must be larger than phase1_max_tokens")
    if program_budget_s <= 0:
        raise ValueError("program_budget_s must be positive")
    resolved_eval_timeout_s = program_budget_s + 60 if eval_timeout_s is None else eval_timeout_s
    if resolved_eval_timeout_s <= program_budget_s:
        raise ValueError("eval_timeout_s must be larger than program_budget_s")
    if eval_cpus_per_task < 1:
        raise ValueError("eval_cpus_per_task must be positive")
    if lora_rank <= 0:
        raise ValueError("lora_rank must be positive")
    if steps < 1:
        raise ValueError("steps must be positive")
    resolved_training_horizon_steps = steps if training_horizon_steps is None else training_horizon_steps
    if (
        not isinstance(resolved_training_horizon_steps, int)
        or isinstance(resolved_training_horizon_steps, bool)
        or resolved_training_horizon_steps < steps
    ):
        raise ValueError("training_horizon_steps must be at least steps")
    if resume and not run_name.strip():
        raise ValueError("resume requires an explicit run_name")

    import torch

    visible_gpus = torch.cuda.device_count()
    if visible_gpus != gpu_count:
        raise RuntimeError(f"PyTorch sees {visible_gpus} GPUs, expected {gpu_count}")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(visible_gpus)]

    resolved_run_name = run_name.strip() or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    resolved_model_path = model_path or state_root / "hf" / MODEL_ID.replace("/", "--")
    state_dir = state_root / "runs" / resolved_run_name
    if resume:
        if not state_dir.is_dir():
            raise ValueError(f"cannot resume missing run directory: {state_dir}")
    else:
        state_dir.mkdir(parents=True, exist_ok=False)
    stack_log = state_dir / "stack.log"
    result_path = state_dir / ("smoke-result.json" if steps == 1 else "experiment-result.json")
    progress_path = state_dir / "progress.json"
    resume_path = state_dir / "resume-state.json"
    checkpoint_root = state_dir / "checkpoints" / "megatron"
    resume_state = _read_resume_state(resume_path) if resume else None
    start_step = int(resume_state["next_step"]) if resume_state is not None else 0
    if resume_state is not None:
        _validate_resume_settings(
            resume_state,
            run_name=resolved_run_name,
            model=MODEL_ID,
            groups=groups,
            rollouts=rollouts,
            phase1_max_tokens=phase1_max_tokens,
            sequence_length=sequence_length,
            lora_rank=lora_rank,
            program_budget_s=program_budget_s,
            eval_timeout_s=resolved_eval_timeout_s,
            eval_cpus_per_task=eval_cpus_per_task,
            tensor_parallel_size=tensor_parallel_size,
            training_horizon_steps=resolved_training_horizon_steps,
        )
        if not (checkpoint_root / "latest_checkpointed_iteration.txt").is_file():
            raise ValueError(f"resumable Megatron checkpoint is missing under {checkpoint_root}")
    if start_step >= steps:
        raise ValueError(f"run already completed {start_step} steps; requested total is {steps}")

    _download_model(resolved_model_path)

    # Slime binds its SGLang router to the node address returned by
    # ``get_host_info()``, not necessarily loopback. Reef must use that same
    # address on the direct single-node deployment.
    from reef.train.slime.utils.http_utils import _wrap_ipv6, get_host_info

    inference_host = _wrap_ipv6(get_host_info()[1])
    no_proxy_host = inference_host.strip("[]")
    token = secrets.token_hex(16)
    global_batch_size = groups * rollouts
    environment = {
        **os.environ,
        "NO_PROXY": ",".join(
            value
            for value in (
                os.environ.get("NO_PROXY", ""),
                no_proxy_host,
            )
            if value
        ),
        "REEF_TOKEN": token,
        "TTTD_MODEL_PATH": str(resolved_model_path),
        "TTTD_STATE_DIR": str(state_dir),
        "TTTD_INFERENCE_HOST": inference_host,
        "TTTD_GROUPS_PER_STEP": str(groups),
        "TTTD_ROLLOUTS_PER_GROUP": str(rollouts),
        "TTTD_GLOBAL_BATCH_SIZE": str(global_batch_size),
        # Reef's artifact checkpoint path exports a complete HF model. Long
        # experiments instead use the resumable Megatron checkpoints requested
        # explicitly below, avoiding a full HF export every few steps.
        "TTTD_CHECKPOINT_INTERVAL": str(steps + 1),
        # Megatron derives optimizer-scheduler bounds from num_rollout. Keep
        # the final horizon stable when a run resumes from a checkpoint.
        "TTTD_NUM_STEPS": str(resolved_training_horizon_steps),
        "TTTD_LOAD_PATH": str(checkpoint_root),
        "TTTD_PHASE1_MAX_TOKENS": str(phase1_max_tokens),
        "TTTD_ROLLOUT_TEMPERATURE": "1.0",
        "TTTD_MAX_RESPONSE_LENGTH": str(sequence_length),
        "TTTD_SEQ_LENGTH": str(sequence_length),
        "TTTD_MAX_TOKENS_PER_GPU": str(sequence_length),
        "TTTD_NUM_GPUS": str(gpu_count),
        "TTTD_CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(gpu_count)),
        "TTTD_TENSOR_PARALLEL_SIZE": str(tensor_parallel_size),
        "TTTD_RAY_NUM_CPUS": str(min(60, os.cpu_count() or 1)),
        "TTTD_LORA_RANK": str(lora_rank),
        "TTTD_LORA_ALPHA": str(lora_rank),
        "TTTD_CUDA_GRAPH_MAX_BS": str(min(global_batch_size, 128)),
        "RAY_ADDRESS": "127.0.0.1:6379",
    }
    root = project_root()
    config_path = root / "examples" / "tttd" / "deployments" / "qwen3_8b_lora" / "config.yaml"
    stack: subprocess.Popen[bytes] | None = None
    ray_initialized = False
    log_handle = stack_log.open("ab" if resume else "wb")
    try:
        if resume:
            log_handle.write(f"\n===== resume from step {start_step} =====\n".encode())
            log_handle.flush()
        stack = subprocess.Popen(
            [sys.executable, "-m", "reef", "serve", "-c", str(config_path)],
            cwd=root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        _wait_for_health(stack, stack_log, timeout_s=3_600)

        import ray

        ray.init(address="127.0.0.1:6379", namespace="reef", logging_level=logging.ERROR)
        ray_initialized = True
        bridge = ray.get_actor("reef-train-bridge", namespace="reef")
        bridge_start_step = int(ray.get(bridge.start_rollout_id.remote()))
        if bridge_start_step != start_step:
            raise RuntimeError(
                f"checkpoint starts at rollout {bridge_start_step}, "
                f"but archive resume state starts at step {start_step}"
            )
        initial_weight_version = str(ray.get(bridge.current_weight_version.remote()))

        from examples.reef_client import ReefClient
        from examples.tttd import ReefTTTDiscoverHarness, TTTDChatRequestBuilder
        from examples.tttd.erdos_min_overlap import ErdosMinOverlapProblem

        client = ReefClient("http://127.0.0.1:8900", token=token, timeout_s=8 * 60 * 60)
        problem = ErdosMinOverlapProblem(
            eval_timeout_s=resolved_eval_timeout_s,
            num_cpus_per_task=eval_cpus_per_task,
            work_dir=state_dir / "programs",
            program_budget_s=program_budget_s,
            program_executor=program_executor,
        )
        harness = ReefTTTDiscoverHarness(
            client,
            problem,
            # Reef restores the durable scenario registration before accepting
            # new requests and permits one training scenario per process. Keep
            # the lane stable across restarts so its commit-log high-water mark
            # and the paired Megatron/PUCT checkpoint resume together.
            scenario=_training_scenario_name(resolved_run_name),
            model=str(resolved_model_path),
            inference_path="/v1/chat/completions",
            groups_per_step=groups,
            rollouts_per_group=rollouts,
            max_workers=global_batch_size,
            request_builder=TTTDChatRequestBuilder(
                max_new_tokens=phase1_max_tokens,
                temperature=1.0,
                enable_thinking=True,
                two_phase_context_window=sequence_length,
                lora_path="reef_lora",
            ),
        )
        if resume_state is not None:
            harness.archive.load_state_dict(
                resume_state["archive"],
                deserialize_state=_deserialize_erdos_state,
            )

        print(
            json.dumps(
                {
                    "event": "experiment_started",
                    "run_name": resolved_run_name,
                    "start_step": start_step,
                    "steps_requested": steps,
                    "training_horizon_steps": resolved_training_horizon_steps,
                    "groups": groups,
                    "rollouts": rollouts,
                    "tensor_parallel_size": tensor_parallel_size,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        started_at = time.monotonic()
        prior_progress = _read_json(progress_path) if resume and progress_path.is_file() else {}
        prior_elapsed_s = float(prior_progress.get("elapsed_s", 0.0))
        step_summaries = [
            summary
            for summary in prior_progress.get("step_summaries", [])
            if isinstance(summary, dict) and int(summary.get("step", -1)) < start_step
        ]
        final_weight_version = initial_weight_version
        for step in range(start_step, steps):
            step_started_at = time.monotonic()
            version_before = str(ray.get(bridge.current_weight_version.remote()))
            results = harness.run_step(step)
            bridge_health = _wait_for_training_step(
                bridge,
                expected_completed_steps=step - start_step + 1,
                expected_rollout_id=step,
                stack=stack,
                stack_log=stack_log,
            )
            step_elapsed_s = time.monotonic() - step_started_at
            version_after = str(ray.get(bridge.current_weight_version.remote()))

            grouped_rewards = [
                [result.reward for result in results[offset : offset + rollouts]]
                for offset in range(0, len(results), rollouts)
            ]
            nonconstant_groups = sum(
                any(reward != rewards[0] for reward in rewards[1:]) for rewards in grouped_rewards
            )
            retained_trajectories = (nonconstant_groups or 1) * rollouts
            valid_rollouts = sum(result.state is not None for result in results)
            metrics = dict(bridge_health.get("last_train_metrics") or {})
            grad_norm = metrics.get("train/grad_norm")
            local_completed_steps = step - start_step + 1

            grad_norm_value = _require_step_success(
                expected_rollouts=global_batch_size,
                actual_rollouts=len(results),
                valid_rollouts=valid_rollouts,
                nonconstant_groups=nonconstant_groups,
                retained_trajectories=retained_trajectories,
                version_before=version_before,
                version_after=version_after,
                bridge_health=bridge_health,
                expected_completed_train_steps=local_completed_steps,
                expected_rollout_id=step,
                grad_norm=grad_norm,
                lora_rank=lora_rank,
                require_nonconstant_step=require_nonconstant_step,
            )

            best = harness.archive.best()
            step_summary = {
                "step": step,
                "elapsed_s": step_elapsed_s,
                "sampled_rollouts": len(results),
                "valid_rollouts": valid_rollouts,
                "nonconstant_groups": nonconstant_groups,
                "degenerate_step": valid_rollouts == 0 or nonconstant_groups == 0,
                "retained_trajectories": retained_trajectories,
                "weight_version_before": version_before,
                "weight_version_after": version_after,
                "grad_norm": grad_norm_value,
                "train_metrics": metrics,
                "best_c5_bound": getattr(best.state, "c5_bound", None),
                "best_reward": best.reward,
            }
            step_summaries.append(step_summary)
            final_weight_version = version_after
            print(
                json.dumps(
                    {
                        "event": "step_completed",
                        **{
                            key: step_summary[key]
                            for key in (
                                "step",
                                "elapsed_s",
                                "valid_rollouts",
                                "nonconstant_groups",
                                "retained_trajectories",
                                "weight_version_after",
                                "grad_norm",
                                "best_c5_bound",
                            )
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            progress = {
                "ok": True,
                "status": "running",
                "run_name": resolved_run_name,
                "steps_requested": steps,
                "completed_steps": step + 1,
                "next_step": step + 1,
                "last_checkpoint_step": (int(resume_state["next_step"]) - 1 if resume_state is not None else None),
                "elapsed_s": prior_elapsed_s + time.monotonic() - started_at,
                "step_summaries": step_summaries,
                "state_dir": str(state_dir),
            }
            _write_json(progress_path, progress)

            # The durable bridge transaction has already saved the paired
            # Megatron/HF checkpoint before publishing this weight version.
            # Persist the matching search archive after every completed step
            # so model and PUCT state always resume at the same rollout id.
            if not (checkpoint_root / "latest_checkpointed_iteration.txt").is_file():
                raise RuntimeError(f"durable Megatron checkpoint is missing under {checkpoint_root}")
            resume_state = {
                "format_version": 2,
                "run_name": resolved_run_name,
                "next_step": step + 1,
                "steps_requested": steps,
                "training_horizon_steps": resolved_training_horizon_steps,
                "model": MODEL_ID,
                "groups_per_step": groups,
                "rollouts_per_group": rollouts,
                "phase1_max_tokens": phase1_max_tokens,
                "sequence_length": sequence_length,
                "lora_rank": lora_rank,
                "tensor_parallel_size": tensor_parallel_size,
                "program_budget_s": program_budget_s,
                "eval_timeout_s": resolved_eval_timeout_s,
                "eval_cpus_per_task": eval_cpus_per_task,
                "archive": harness.archive.state_dict(
                    serialize_state=_serialize_erdos_state,
                ),
            }
            _write_json(resume_path, resume_state)
            progress["last_checkpoint_step"] = step
            _write_json(progress_path, progress)
            print(
                json.dumps(
                    {
                        "event": "checkpoint_completed",
                        "step": step,
                        "path": str(checkpoint_root),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        elapsed_s = prior_elapsed_s + time.monotonic() - started_at
        bridge_health = dict(ray.get(bridge.health.remote()))
        best = harness.archive.best()
        summary = {
            "ok": True,
            "status": "completed",
            "run_name": resolved_run_name,
            "model": MODEL_ID,
            "thinking": True,
            "gpu": gpu_names[0] if len(set(gpu_names)) == 1 else "mixed",
            "gpu_count": gpu_count,
            "gpu_names": gpu_names,
            "tensor_parallel_size": tensor_parallel_size,
            "lora_rank": lora_rank,
            "steps_requested": steps,
            "training_horizon_steps": resolved_training_horizon_steps,
            "completed_steps": steps,
            "start_step": start_step,
            "groups_per_step": groups,
            "rollouts_per_group": rollouts,
            "sampled_rollouts": sum(item["sampled_rollouts"] for item in step_summaries),
            "valid_rollouts": sum(item["valid_rollouts"] for item in step_summaries),
            "nonconstant_groups": sum(item["nonconstant_groups"] for item in step_summaries),
            "degenerate_steps": sum(int(_step_summary_is_degenerate(item)) for item in step_summaries),
            "retained_trajectories": sum(item["retained_trajectories"] for item in step_summaries),
            "program_budget_s": program_budget_s,
            "eval_timeout_s": resolved_eval_timeout_s,
            "eval_cpus_per_task": eval_cpus_per_task,
            "phase1_max_tokens": phase1_max_tokens,
            "sequence_length": sequence_length,
            "elapsed_s": elapsed_s,
            "weight_version_before": initial_weight_version,
            "weight_version_after": final_weight_version,
            "grad_norm": step_summaries[-1]["grad_norm"],
            "train_metrics": step_summaries[-1]["train_metrics"],
            "bridge_health": bridge_health,
            "best_c5_bound": getattr(best.state, "c5_bound", None),
            "best_reward": best.reward,
            "step_summaries": step_summaries,
            "state_dir": str(state_dir),
        }
        _write_json(result_path, summary)
        _write_json(progress_path, summary)
        print(
            json.dumps(
                {
                    "event": "experiment_completed",
                    "run_name": resolved_run_name,
                    "completed_steps": steps,
                    "best_c5_bound": summary["best_c5_bound"],
                    "elapsed_s": elapsed_s,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return summary
    except Exception as exc:
        progress = _read_json(progress_path) if progress_path.is_file() else {}
        completed_steps = progress.get("completed_steps", start_step)
        if not isinstance(completed_steps, int) or isinstance(completed_steps, bool):
            completed_steps = start_step
        failure = {
            "ok": False,
            "status": "failed",
            "run_name": resolved_run_name,
            "error": f"{type(exc).__name__}: {exc}",
            "steps_requested": steps,
            "completed_steps": completed_steps,
            "stack_log_tail": _tail(stack_log),
            "state_dir": str(state_dir),
        }
        _write_json(result_path, failure)
        raise RuntimeError(json.dumps(failure, ensure_ascii=False)) from exc
    finally:
        if ray_initialized:
            import ray

            ray.shutdown()
        if stack is not None and stack.poll() is None:
            stack.terminate()
            try:
                stack.wait(timeout=180)
            except subprocess.TimeoutExpired:
                stack.kill()
                stack.wait()
        log_handle.close()


def _download_model(model_path: Path) -> None:
    if (model_path / "config.json").is_file() and any(model_path.glob("*.safetensors")):
        return
    from huggingface_hub import snapshot_download

    model_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=model_path,
        token=os.environ.get("HF_TOKEN"),
    )


def _wait_for_health(stack: subprocess.Popen[bytes], stack_log: Path, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    health_url = "http://127.0.0.1:8900/healthz"
    while time.monotonic() < deadline:
        return_code = stack.poll()
        if return_code is not None:
            raise RuntimeError(f"Reef stack exited with code {return_code}: {_tail(stack_log)}")
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"Reef stack was not ready after {timeout_s:g}s: {_tail(stack_log)}")


def _wait_for_training_step(
    bridge: Any,
    *,
    expected_completed_steps: int,
    expected_rollout_id: int,
    stack: subprocess.Popen[bytes],
    stack_log: Path,
    timeout_s: float = 14_400,
) -> dict[str, Any]:
    """Wait for Reef's asynchronous durable train/checkpoint/publish job."""
    import ray

    deadline = time.monotonic() + timeout_s
    last_health: dict[str, Any] = {}
    while time.monotonic() < deadline:
        return_code = stack.poll()
        if return_code is not None:
            raise RuntimeError(f"Reef stack exited with code {return_code}: {_tail(stack_log)}")
        last_health = dict(ray.get(bridge.health.remote(), timeout=30))
        if last_health.get("ok") is not True:
            raise RuntimeError(f"durable training bridge failed: {last_health}; {_tail(stack_log)}")
        if last_health.get("completed_train_steps") == expected_completed_steps:
            if last_health.get("last_train_rollout_id") != expected_rollout_id:
                raise RuntimeError(f"durable training completed the wrong rollout: {last_health}")
            if last_health.get("phase") != "serving":
                raise RuntimeError(f"durable training did not return to serving: {last_health}")
            return last_health
        time.sleep(2)
    raise TimeoutError(
        f"training rollout {expected_rollout_id} did not complete after {timeout_s:g}s: "
        f"{last_health}; {_tail(stack_log)}"
    )


def _require_step_success(
    *,
    expected_rollouts: int,
    actual_rollouts: int,
    valid_rollouts: int,
    nonconstant_groups: int,
    retained_trajectories: int,
    version_before: str,
    version_after: str,
    bridge_health: dict[str, Any],
    expected_completed_train_steps: int,
    expected_rollout_id: int,
    grad_norm: Any,
    lora_rank: int,
    require_nonconstant_step: bool = False,
) -> float:
    if actual_rollouts != expected_rollouts:
        raise RuntimeError(f"sampled {actual_rollouts} rollouts, expected {expected_rollouts}")
    if require_nonconstant_step and valid_rollouts == 0:
        raise RuntimeError("every generated Erdős program failed verification")
    if require_nonconstant_step and nonconstant_groups == 0:
        raise RuntimeError("every reward group was constant")
    if bridge_health.get("ok") is not True or bridge_health.get("phase") != "serving":
        raise RuntimeError(f"bridge did not return to serving: {bridge_health}")
    if bridge_health.get("completed_train_steps") != expected_completed_train_steps:
        raise RuntimeError(f"expected {expected_completed_train_steps} completed train steps: {bridge_health}")
    if bridge_health.get("last_train_rollout_id") != expected_rollout_id:
        raise RuntimeError(f"expected last rollout id {expected_rollout_id}: {bridge_health}")
    if version_before == version_after:
        raise RuntimeError(f"serving weight version did not change: {version_before}")
    if bridge_health.get("last_weight_version") != version_after:
        raise RuntimeError(f"bridge/actor version mismatch: {bridge_health} vs {version_after}")
    if not isinstance(grad_norm, (int, float)) or not math.isfinite(float(grad_norm)) or grad_norm <= 0:
        raise RuntimeError(f"optimizer did not report a positive finite grad norm: {grad_norm!r}")
    reported_batch = (bridge_health.get("last_train_metrics") or {}).get("train/global_batch_size")
    if reported_batch != retained_trajectories:
        raise RuntimeError(f"trainer consumed global batch {reported_batch!r}, expected {retained_trajectories}")
    metrics = bridge_health.get("last_train_metrics") or {}
    trainable = metrics.get("train/lora_trainable_parameters")
    base_trainable = metrics.get("train/lora_base_trainable_parameters")
    lora_b_nonzero = metrics.get("train/lora_b_nonzero")
    lora_b_l1 = metrics.get("train/lora_b_l1")
    if not isinstance(trainable, (int, float)) or trainable <= 0:
        raise RuntimeError(f"rank-{lora_rank} LoRA exposed no trainable adapter parameters: {metrics}")
    if base_trainable != 0:
        raise RuntimeError(f"LoRA step left {base_trainable!r} base parameters trainable")
    if not isinstance(lora_b_nonzero, (int, float)) or lora_b_nonzero <= 0:
        raise RuntimeError(f"LoRA-B remained zero after the optimizer step: {metrics}")
    if not isinstance(lora_b_l1, (int, float)) or not math.isfinite(lora_b_l1) or lora_b_l1 <= 0:
        raise RuntimeError(f"LoRA-B did not receive a finite nonzero update: {metrics}")
    return float(grad_norm)


def _tail(path: Path, characters: int = 12_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-characters:]
    except OSError:
        return ""


def _serialize_erdos_state(state: Any) -> dict[str, Any]:
    from examples.tttd.erdos_min_overlap import ErdosConstruction

    if not isinstance(state, ErdosConstruction):
        raise TypeError(f"expected ErdosConstruction, got {type(state).__name__}")
    return asdict(state)


def _deserialize_erdos_state(payload: Any):
    from examples.tttd.erdos_min_overlap import ErdosConstruction

    if not isinstance(payload, dict):
        raise ValueError("Erdős archive state must be an object")
    try:
        return ErdosConstruction(
            h_values=tuple(float(value) for value in payload["h_values"]),
            c5_bound=float(payload["c5_bound"]),
            n_points=int(payload["n_points"]),
            timestep=int(payload.get("timestep", -1)),
            code=str(payload.get("code", "")),
            parent_values=tuple(float(value) for value in payload.get("parent_values", ())),
            parent_ids=tuple(str(value) for value in payload.get("parent_ids", ())),
            observation=str(payload.get("observation", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid Erdős archive state") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _read_resume_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"resume state does not exist: {path}")
    payload = _read_json(path)
    if payload.get("format_version") != 2:
        raise ValueError(f"unsupported resume-state format in {path}")
    next_step = payload.get("next_step")
    archive = payload.get("archive")
    if not isinstance(next_step, int) or isinstance(next_step, bool) or next_step < 1 or not isinstance(archive, dict):
        raise ValueError(f"invalid resume state in {path}")
    return payload


def _validate_resume_settings(
    payload: dict[str, Any],
    *,
    run_name: str,
    model: str,
    groups: int,
    rollouts: int,
    phase1_max_tokens: int,
    sequence_length: int,
    lora_rank: int,
    program_budget_s: float,
    eval_timeout_s: float,
    eval_cpus_per_task: int,
    tensor_parallel_size: int,
    training_horizon_steps: int,
) -> None:
    expected = {
        "run_name": run_name,
        "model": model,
        "groups_per_step": groups,
        "rollouts_per_group": rollouts,
        "phase1_max_tokens": phase1_max_tokens,
        "sequence_length": sequence_length,
        "lora_rank": lora_rank,
        "program_budget_s": program_budget_s,
        "eval_timeout_s": eval_timeout_s,
        "eval_cpus_per_task": eval_cpus_per_task,
    }
    if "tensor_parallel_size" in payload:
        expected["tensor_parallel_size"] = tensor_parallel_size
    if "training_horizon_steps" in payload:
        expected["training_horizon_steps"] = training_horizon_steps
    mismatches = {
        key: {"checkpoint": payload.get(key), "requested": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume settings do not match the checkpoint: {mismatches}")


def _step_summary_is_degenerate(summary: dict[str, Any]) -> bool:
    """Read new summaries while remaining compatible with older runs."""
    if "degenerate_step" in summary:
        return bool(summary["degenerate_step"])
    return int(summary.get("valid_rollouts", 0)) == 0 or int(summary.get("nonconstant_groups", 0)) == 0
