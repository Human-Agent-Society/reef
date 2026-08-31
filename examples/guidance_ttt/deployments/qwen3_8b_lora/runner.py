"""Run summary-only Guidance-TTT against Reef's Qwen3-8B LoRA stack."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from examples.guidance_ttt import (
    OpenAICompatibleExecutionClient,
    ReefGuidanceTTTHarness,
    create_verified_baseline_seed,
    get_task_spec,
    gpt_oss_120b_backend,
    openrouter_glm_5_2_backend,
    prepare_library,
)
from examples.guidance_ttt.verifier.config import sanitize_verifier_config
from examples.tttd.deployments.qwen3_8b_lora.runner import (
    _download_model,
    _require_step_success,
    _tail,
    _wait_for_health,
    _wait_for_training_step,
    project_root,
)

MODEL_ID = "Qwen/Qwen3-8B"


def _training_scenario_name(run_name: str) -> str:
    return f"guidance-{run_name}-session-0"


def run_experiment(
    *,
    gpu_count: int,
    state_root: Path,
    executor: str,
    tensor_parallel_size: int = 2,
    executor_base_url: str = "http://127.0.0.1:8000/v1",
    gpt_oss_reasoning_effort: str = "high",
    executor_temperature: float | None = None,
    executor_max_tokens: int | None = None,
    model_path: Path | None = None,
    groups: int = 2,
    rollouts: int = 4,
    guidance_max_tokens: int = 8_192,
    guidance_temperature: float = 1.0,
    guidance_top_p: float = 1.0,
    guidance_top_k: int = -1,
    sampling_seed: int | None = None,
    sequence_length: int = 12_288,
    lora_rank: int = 32,
    steps: int = 1,
    training_horizon_steps: int | None = None,
    max_workers: int | None = None,
    verifier_timeout_s: int | None = None,
    verifier_concurrency: int | None = None,
    verifier_config: Mapping[str, Any] | None = None,
    frontiercs_base_dir: Path | None = None,
    judge_url: str = "http://127.0.0.1:8081",
    seed_library_path: Path | None = None,
    run_name: str = "",
    resume: bool = False,
    require_nonconstant_step: bool = False,
    task_id: str = "polyomino_packing",
    adaptation_enabled: bool = True,
) -> dict[str, Any]:
    """Run Guidance-TTT with either adaptive or matched frozen policy weights."""
    task = get_task_spec(task_id)
    resolved_verifier_timeout_s = (
        _default_verifier_timeout_s(task.task_id)
        if verifier_timeout_s is None
        else int(verifier_timeout_s)
    )
    _validate_settings(
        gpu_count=gpu_count,
        tensor_parallel_size=tensor_parallel_size,
        groups=groups,
        rollouts=rollouts,
        guidance_max_tokens=guidance_max_tokens,
        sequence_length=sequence_length,
        lora_rank=lora_rank,
        steps=steps,
        verifier_timeout_s=resolved_verifier_timeout_s,
    )
    _validate_sampling_settings(
        temperature=guidance_temperature,
        top_p=guidance_top_p,
        top_k=guidance_top_k,
        sampling_seed=sampling_seed,
    )
    training_horizon = steps if training_horizon_steps is None else training_horizon_steps
    if (
        not isinstance(training_horizon, int)
        or isinstance(training_horizon, bool)
        or training_horizon < steps
    ):
        raise ValueError("training_horizon_steps must be at least steps")
    if resume and not run_name.strip():
        raise ValueError("resume requires an explicit run_name")

    if executor == "gpt_oss_120b":
        backend = gpt_oss_120b_backend(
            base_url=executor_base_url,
            concurrency=max_workers or 16,
            reasoning_effort=gpt_oss_reasoning_effort,
        )
    elif executor == "openrouter_glm_5_2":
        backend = openrouter_glm_5_2_backend(concurrency=max_workers or 16)
    else:
        raise ValueError("executor must be 'gpt_oss_120b' or 'openrouter_glm_5_2'")
    if executor_temperature is not None:
        if not math.isfinite(float(executor_temperature)) or float(executor_temperature) < 0:
            raise ValueError("executor_temperature must be finite and non-negative")
        backend = replace(backend, temperature=float(executor_temperature))
    if executor_max_tokens is not None:
        if int(executor_max_tokens) < 1:
            raise ValueError("executor_max_tokens must be positive")
        backend = replace(backend, max_tokens=int(executor_max_tokens))
    execution_client = OpenAICompatibleExecutionClient(backend)

    import torch

    visible_gpus = torch.cuda.device_count()
    if visible_gpus != gpu_count:
        raise RuntimeError(f"PyTorch sees {visible_gpus} actor GPUs, expected {gpu_count}")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(visible_gpus)]

    root = project_root()
    resolved_run_name = run_name.strip() or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    state_dir = state_root / "guidance-runs" / resolved_run_name
    if resume:
        if not state_dir.is_dir():
            raise ValueError(f"cannot resume missing run directory: {state_dir}")
    else:
        state_dir.mkdir(parents=True, exist_ok=False)
    resolved_model_path = model_path or state_root / "hf" / MODEL_ID.replace("/", "--")
    resolved_frontiercs = frontiercs_base_dir or root / "reference" / "Frontier-CS"
    stack_log = state_dir / "stack.log"
    result_path = state_dir / "result.json"
    resume_path = state_dir / "resume-state.json"
    working_library_path = state_dir / "library.json"
    committed_library_path = state_dir / "committed-library.json"
    checkpoint_root = state_dir / "checkpoints" / "megatron"
    resolved_verifier_config = _resolve_verifier_config(
        task_id=task.task_id,
        frontiercs_base_dir=resolved_frontiercs,
        judge_url=judge_url,
        overrides=verifier_config,
    )
    safe_verifier_config = _safe_verifier_config(resolved_verifier_config)
    resolved_verifier_concurrency = verifier_concurrency
    if resolved_verifier_concurrency is None and "concurrency" in resolved_verifier_config:
        resolved_verifier_concurrency = int(resolved_verifier_config["concurrency"])
    if resolved_verifier_concurrency is not None and resolved_verifier_concurrency < 1:
        raise ValueError("verifier_concurrency must be positive")
    resolved_seed = seed_library_path or _default_seed_path(root, state_dir, task.task_id)
    if not resume and seed_library_path is None and task.bootstrap_solution:
        create_verified_baseline_seed(
            resolved_seed,
            task=task,
            verifier_timeout_s=resolved_verifier_timeout_s,
            verifier_config=resolved_verifier_config,
        )
    seed_sha256 = _sha256_file(resolved_seed)

    resume_state = _read_json(resume_path) if resume else None
    start_step = int(resume_state["next_step"]) if resume_state else 0
    if start_step >= steps:
        raise ValueError(f"run already completed {start_step} steps; requested total is {steps}")
    if resume:
        _validate_resume_state(
            resume_state,
            executor=executor,
            executor_backend=backend.safe_dict(),
            gpt_oss_reasoning_effort=gpt_oss_reasoning_effort,
            groups=groups,
            rollouts=rollouts,
            guidance_max_tokens=guidance_max_tokens,
            guidance_temperature=float(guidance_temperature),
            guidance_top_p=float(guidance_top_p),
            guidance_top_k=int(guidance_top_k),
            sampling_seed=sampling_seed,
            sequence_length=sequence_length,
            lora_rank=lora_rank,
            tensor_parallel_size=tensor_parallel_size,
            training_horizon_steps=training_horizon,
            task_id=task.task_id,
            verifier_config=safe_verifier_config,
            verifier_concurrency=resolved_verifier_concurrency,
            verifier_timeout_s=resolved_verifier_timeout_s,
            seed_sha256=seed_sha256,
            adaptation_enabled=adaptation_enabled,
        )
        if not committed_library_path.is_file():
            raise ValueError(f"committed Guidance archive is missing: {committed_library_path}")
        shutil.copy2(committed_library_path, working_library_path)
        if adaptation_enabled and not (checkpoint_root / "latest_checkpointed_iteration.txt").is_file():
            raise ValueError(f"resumable Megatron checkpoint is missing under {checkpoint_root}")

    library = prepare_library(
        seed_path=resolved_seed,
        run_path=working_library_path,
        groups_per_step=groups,
        rollouts_per_group=rollouts,
        task=task,
    )
    if not committed_library_path.exists():
        _atomic_copy(working_library_path, committed_library_path)
    _download_model(resolved_model_path)

    from reef.train.slime.utils.http_utils import _wrap_ipv6, get_host_info

    inference_host = _wrap_ipv6(get_host_info()[1])
    no_proxy_host = inference_host.strip("[]")
    global_batch_size = groups * rollouts
    environment = {
        **os.environ,
        "NO_PROXY": ",".join(filter(None, (os.environ.get("NO_PROXY", ""), no_proxy_host))),
        "REEF_TOKEN": secrets.token_hex(16),
        "TTTD_MODEL_PATH": str(resolved_model_path),
        "TTTD_STATE_DIR": str(state_dir),
        "TTTD_INFERENCE_HOST": inference_host,
        "TTTD_GROUPS_PER_STEP": str(groups),
        "TTTD_ROLLOUTS_PER_GROUP": str(rollouts),
        "TTTD_GLOBAL_BATCH_SIZE": str(global_batch_size),
        "TTTD_CHECKPOINT_INTERVAL": str(steps + 1),
        "TTTD_NUM_STEPS": str(training_horizon),
        "TTTD_LOAD_PATH": str(checkpoint_root),
        "TTTD_PHASE1_MAX_TOKENS": str(guidance_max_tokens),
        "TTTD_ROLLOUT_TEMPERATURE": str(float(guidance_temperature)),
        "TTTD_MAX_RESPONSE_LENGTH": str(guidance_max_tokens),
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
    config_path = root / "examples" / "tttd" / "deployments" / "qwen3_8b_lora" / "config.yaml"
    stack: subprocess.Popen[bytes] | None = None
    ray_initialized = False
    log_handle = stack_log.open("ab" if resume else "wb")
    try:
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
        if adaptation_enabled and bridge_start_step != start_step:
            raise RuntimeError(
                f"checkpoint starts at rollout {bridge_start_step}, but Guidance archive starts at {start_step}"
            )
        if not adaptation_enabled and bridge_start_step != 0:
            raise RuntimeError(
                "frozen-policy control must start from the unmodified base policy, "
                f"but the training bridge starts at rollout {bridge_start_step}"
            )
        initial_weight_version = str(ray.get(bridge.current_weight_version.remote()))

        from examples.reef_client import ReefClient

        client = ReefClient(
            "http://127.0.0.1:8900",
            token=environment["REEF_TOKEN"],
            timeout_s=8 * 60 * 60,
        )
        harness = ReefGuidanceTTTHarness(
            client,
            execution_client,
            library,
            scenario=_training_scenario_name(resolved_run_name),
            model=str(resolved_model_path),
            task=task,
            groups_per_step=groups,
            rollouts_per_group=rollouts,
            guidance_max_tokens=guidance_max_tokens,
            guidance_temperature=guidance_temperature,
            guidance_top_p=guidance_top_p,
            guidance_top_k=guidance_top_k,
            sampling_seed=sampling_seed,
            max_workers=max_workers or min(global_batch_size, backend.concurrency),
            verifier_timeout_s=resolved_verifier_timeout_s,
            verifier_concurrency=resolved_verifier_concurrency,
            verifier_config=resolved_verifier_config,
            report_training=adaptation_enabled,
        )
        step_summaries = list((resume_state or {}).get("step_summaries", []))
        final_weight_version = initial_weight_version
        started_at = time.monotonic()
        for step in range(start_step, steps):
            step_started_at = time.monotonic()
            version_before = str(ray.get(bridge.current_weight_version.remote()))
            results = harness.run_step(step)
            if adaptation_enabled:
                bridge_health = _wait_for_training_step(
                    bridge,
                    expected_completed_steps=step - start_step + 1,
                    expected_rollout_id=step,
                    stack=stack,
                    stack_log=stack_log,
                )
            else:
                bridge_health = dict(ray.get(bridge.health.remote(), timeout=30))
            version_after = str(ray.get(bridge.current_weight_version.remote()))
            grouped_rewards = [
                [result.reward for result in results[offset : offset + rollouts]]
                for offset in range(0, len(results), rollouts)
            ]
            nonconstant_groups = sum(
                any(reward != rewards[0] for reward in rewards[1:]) for rewards in grouped_rewards
            )
            valid_rollouts = sum(result.verification.valid for result in results)
            # The TTTD processor keeps one group on an otherwise fully
            # constant step so the frozen-base KL transaction remains
            # well-defined. Mirror that cardinality in the qualification
            # checks instead of claiming that the trainer consumed no data.
            retained_trajectories = (nonconstant_groups or 1) * rollouts if adaptation_enabled else 0
            metrics = dict(bridge_health.get("last_train_metrics") or {})
            if require_nonconstant_step and valid_rollouts == 0:
                raise RuntimeError("every execution-model candidate failed verification")
            if require_nonconstant_step and nonconstant_groups == 0:
                raise RuntimeError("every Guidance-TTT reward group was constant")
            grad_norm: float | None
            if adaptation_enabled:
                grad_norm = _require_step_success(
                    expected_rollouts=global_batch_size,
                    actual_rollouts=len(results),
                    valid_rollouts=valid_rollouts,
                    nonconstant_groups=nonconstant_groups,
                    retained_trajectories=retained_trajectories,
                    version_before=version_before,
                    version_after=version_after,
                    bridge_health=bridge_health,
                    expected_completed_train_steps=step - start_step + 1,
                    expected_rollout_id=step,
                    grad_norm=metrics.get("train/grad_norm"),
                    lora_rank=lora_rank,
                    require_nonconstant_step=False,
                )
                if not math.isfinite(grad_norm) or grad_norm <= 0:
                    raise RuntimeError(f"invalid optimizer grad norm: {grad_norm!r}")
            else:
                _require_frozen_step_success(
                    expected_rollouts=global_batch_size,
                    actual_rollouts=len(results),
                    initial_weight_version=initial_weight_version,
                    version_before=version_before,
                    version_after=version_after,
                    bridge_health=bridge_health,
                )
                grad_norm = None
            final_weight_version = version_after
            snapshot = library.snapshot()
            best_node = snapshot["nodes"].get(snapshot.get("best_node_id")) or {}
            step_summary = {
                "step": step,
                "elapsed_s": time.monotonic() - step_started_at,
                "sampled_rollouts": len(results),
                "valid_rollouts": valid_rollouts,
                "well_formed_guidance": sum(result.guidance_format_ok for result in results),
                "nonconstant_groups": nonconstant_groups,
                "degenerate_step": valid_rollouts == 0 or nonconstant_groups == 0,
                "retained_trajectories": retained_trajectories,
                "adaptation_enabled": adaptation_enabled,
                "training_reported": adaptation_enabled,
                "policy_sampling_seed_start": (
                    None
                    if sampling_seed is None
                    else sampling_seed + step * global_batch_size
                ),
                "policy_sampling_seed_end": (
                    None
                    if sampling_seed is None
                    else sampling_seed + (step + 1) * global_batch_size - 1
                ),
                "weight_version_before": version_before,
                "weight_version_after": version_after,
                "grad_norm": grad_norm,
                "best_raw_score": best_node.get("raw_score"),
                "raw_score_label": task.raw_score_label,
                "score_direction": task.score_direction,
                "train_metrics": metrics,
            }
            step_summaries.append(step_summary)

            if adaptation_enabled and not (checkpoint_root / "latest_checkpointed_iteration.txt").is_file():
                raise RuntimeError(f"durable Megatron checkpoint is missing under {checkpoint_root}")
            _atomic_copy(working_library_path, committed_library_path)
            resume_state = {
                "format_version": 1,
                "run_name": resolved_run_name,
                "next_step": step + 1,
                "model": MODEL_ID,
                "executor": executor,
                "gpt_oss_reasoning_effort": gpt_oss_reasoning_effort,
                "executor_backend": backend.safe_dict(),
                "task_id": task.task_id,
                "score_direction": task.score_direction,
                "verifier_config": safe_verifier_config,
                "verifier_concurrency": resolved_verifier_concurrency,
                "verifier_timeout_s": resolved_verifier_timeout_s,
                "seed_sha256": seed_sha256,
                "adaptation_enabled": adaptation_enabled,
                "groups_per_step": groups,
                "rollouts_per_group": rollouts,
                "guidance_max_tokens": guidance_max_tokens,
                "guidance_temperature": float(guidance_temperature),
                "guidance_top_p": float(guidance_top_p),
                "guidance_top_k": int(guidance_top_k),
                "sampling_seed": sampling_seed,
                "sequence_length": sequence_length,
                "lora_rank": lora_rank,
                "tensor_parallel_size": tensor_parallel_size,
                "training_horizon_steps": training_horizon,
                "step_summaries": step_summaries,
            }
            _write_json(resume_path, resume_state)
            print(json.dumps({"event": "guidance_step_completed", **step_summary}, sort_keys=True), flush=True)

        result = {
            "ok": True,
            "run_name": resolved_run_name,
            "model": MODEL_ID,
            "executor": executor,
            "executor_backend": backend.safe_dict(),
            "task_id": task.task_id,
            "score_direction": task.score_direction,
            "raw_score_label": task.raw_score_label,
            "verifier_config": safe_verifier_config,
            "verifier_concurrency": resolved_verifier_concurrency,
            "verifier_timeout_s": resolved_verifier_timeout_s,
            "seed_sha256": seed_sha256,
            "adaptation_enabled": adaptation_enabled,
            "training_reported": adaptation_enabled,
            "prompt_mode": "summary_only",
            "guidance_generation_attempts": 1,
            "require_nonconstant_step": require_nonconstant_step,
            "gpu_count": gpu_count,
            "gpu_names": gpu_names,
            "groups_per_step": groups,
            "rollouts_per_group": rollouts,
            "guidance_max_tokens": guidance_max_tokens,
            "guidance_temperature": float(guidance_temperature),
            "guidance_top_p": float(guidance_top_p),
            "guidance_top_k": int(guidance_top_k),
            "sampling_seed": sampling_seed,
            "sequence_length": sequence_length,
            "lora_rank": lora_rank,
            "tensor_parallel_size": tensor_parallel_size,
            "training_horizon_steps": training_horizon,
            "completed_steps": steps,
            "elapsed_s": time.monotonic() - started_at,
            "weight_version_before": initial_weight_version,
            "weight_version_after": final_weight_version,
            "step_summaries": step_summaries,
            "state_dir": str(state_dir),
        }
        _write_json(result_path, result)
        return result
    except Exception as exc:
        failure = {
            "ok": False,
            "run_name": resolved_run_name,
            "task_id": task.task_id,
            "adaptation_enabled": adaptation_enabled,
            "error": f"{type(exc).__name__}: {exc}",
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


def _validate_settings(**settings: int) -> None:
    for key in ("gpu_count", "tensor_parallel_size", "groups", "guidance_max_tokens", "lora_rank", "steps"):
        if settings[key] < 1:
            raise ValueError(f"{key} must be positive")
    if settings["rollouts"] < 2:
        raise ValueError("rollouts must be at least two")
    if settings["gpu_count"] % settings["tensor_parallel_size"]:
        raise ValueError("gpu_count must be divisible by tensor_parallel_size")
    if settings["sequence_length"] <= settings["guidance_max_tokens"]:
        raise ValueError("sequence_length must be larger than guidance_max_tokens")
    if settings["verifier_timeout_s"] < 1:
        raise ValueError("verifier_timeout_s must be positive")


def _validate_sampling_settings(
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    sampling_seed: int | None,
) -> None:
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("guidance_temperature must be finite and positive")
    if not math.isfinite(float(top_p)) or not 0 < float(top_p) <= 1:
        raise ValueError("guidance_top_p must be in (0, 1]")
    if int(top_k) != -1 and int(top_k) < 1:
        raise ValueError("guidance_top_k must be -1 or positive")
    if sampling_seed is not None and int(sampling_seed) < 0:
        raise ValueError("sampling_seed must be non-negative")


def _require_frozen_step_success(
    *,
    expected_rollouts: int,
    actual_rollouts: int,
    initial_weight_version: str,
    version_before: str,
    version_after: str,
    bridge_health: Mapping[str, Any],
) -> None:
    """Prove that a frozen-policy control sampled normally without updating weights."""
    if actual_rollouts != expected_rollouts:
        raise RuntimeError(f"sampled {actual_rollouts} rollouts, expected {expected_rollouts}")
    if bridge_health.get("ok") is not True or bridge_health.get("phase") != "serving":
        raise RuntimeError(f"frozen-policy bridge is not serving: {dict(bridge_health)}")
    if bridge_health.get("completed_train_steps") != 0:
        raise RuntimeError(f"frozen-policy control unexpectedly trained: {dict(bridge_health)}")
    if not (initial_weight_version == version_before == version_after):
        raise RuntimeError(
            "frozen-policy serving weight changed: "
            f"initial={initial_weight_version}, before={version_before}, after={version_after}"
        )


def _validate_resume_state(state: dict[str, Any], **expected: Any) -> None:
    names = {
        "groups": "groups_per_step",
        "rollouts": "rollouts_per_group",
        **{key: key for key in expected if key not in {"groups", "rollouts"}},
    }
    mismatches = [
        f"{persisted}={state.get(persisted)!r}, requested={expected[requested]!r}"
        for requested, persisted in names.items()
        if state.get(persisted) != expected[requested]
    ]
    if mismatches:
        raise ValueError("resume settings mismatch: " + "; ".join(mismatches))


def _default_seed_path(root: Path, state_dir: Path, task_id: str) -> Path:
    seed_root = root / "examples" / "guidance_ttt" / "seeds"
    if task_id == "polyomino_packing":
        return seed_root / "polyomino_packing" / "gpt_oss_120b_bootstrap_library.json"
    if task_id == "trimul":
        return seed_root / "trimul" / "glm52_scratch_bootstrap_library.json"
    if task_id == "vliw_kernel_optimization":
        return state_dir / "verified-vliw-baseline-seed.json"
    raise KeyError(f"No default Guidance-TTT seed for task {task_id!r}")


def _default_verifier_timeout_s(task_id: str) -> int:
    if task_id == "polyomino_packing":
        return 340
    if task_id == "vliw_kernel_optimization":
        return 120
    if task_id == "trimul":
        return 1_160
    raise KeyError(f"No default verifier timeout for task {task_id!r}")


def _resolve_verifier_config(
    *,
    task_id: str,
    frontiercs_base_dir: Path,
    judge_url: str,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if task_id == "polyomino_packing":
        config: dict[str, Any] = {
            "base_dir": str(frontiercs_base_dir),
            "judge_url": judge_url,
            "problem_id": "0",
        }
    elif task_id == "vliw_kernel_optimization":
        config = {
            "provider": "docker",
            "concurrency": 16,
            "runner_timeout_s": 60,
            "request_timeout_s": 90,
            "max_retries": 3,
            "retry_backoff_s": 2.0,
            "reward_mode": "inverse_cycles",
            "reward_scale": 1_000_000.0,
        }
    elif task_id == "trimul":
        config = {
            "provider": "http",
            "endpoint_env": "TRIMUL_JUDGE_URL",
            "api_key_env": "TRIMUL_JUDGE_TOKEN",
            "concurrency": 1,
            "runner_timeout_s": 520,
            "request_timeout_s": 1_150,
            "max_retries": 2,
            "retry_backoff_s": 2.0,
            "reward_scale": 1_500.0,
        }
    else:
        raise KeyError(f"No default verifier configuration for task {task_id!r}")
    config.update(dict(overrides or {}))
    return config


def _safe_verifier_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return sanitize_verifier_config(config)


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Guidance-TTT seed library is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload
