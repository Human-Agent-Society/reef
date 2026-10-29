"""GPU acceptance: native generation, SAO update, file publication, then repeat."""

import asyncio
import json
import logging
import math
import os
import time
from pathlib import Path

import requests
import yaml
from tests.support._trajectories import policy_trajectory
from transformers import AutoTokenizer

from reef.artifact.artifact import Artifact
from reef.core.job_data import JobData
from reef.runtime.deployment import close_runtimes
from reef.runtime.interfaces import TrainingMethod
from reef.runtime.jobs import TrainingJob
from reef.service.assembly import connect_training_component
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.service_config import service_config_from_mapping


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run = Path(os.environ["SMOKE_RUN"])
    run.mkdir(parents=True, exist_ok=True)
    model = os.environ["MODEL_DIR"]
    gpus = int(os.environ.get("SMOKE_GPUS", "2"))
    raw = {
        "recipe": {"implementation": "recipes.sao.recipe:SAORecipe"},
        "inference": {
            "model-path": model,
            "num-gpus": gpus,
            "tensor-parallel-size": gpus,
            "timeout-s": 600,
            "options": {
                "mem-fraction-static": 0.3,
                "context-length": 2048,
                "max-running-requests": 2,
                "disable-cuda-graph": True,
                "attention-backend": "triton",
            },
        },
        "training": {
            "backend": "slime",
            "colocate": True,
            "ray-address": os.environ["RAY_ADDRESS"],
            "timeout-s": 600,
            "options": {
                "megatron-to-hf-mode": "bridge",
                "save": str(run / "megatron"),
                "load": str(run / "megatron"),
                "save-hf": str(run / "hf" / "{rollout_id}"),
                "save-interval": 1,
                "reef-checkpoint-min-free-space": "2GiB",
                "num-rollout": 100,
                "global-batch-size": 2,
                "rollout-batch-size": 2,
                "micro-batch-size": 1,
                "actor-num-nodes": 1,
                "actor-num-gpus-per-node": gpus,
                "num-gpus-per-node": gpus,
                "tensor-model-parallel-size": gpus,
                "pipeline-model-parallel-size": 1,
                "seq-length": 2048,
                "rollout-max-response-len": 64,
                "padded-vocab-size": 151936,
                "position-embedding-type": "rope",
                "swiglu": True,
                "disable-bias-linear": True,
                "qk-layernorm": True,
                "normalization": "RMSNorm",
                "rotary-base": 1000000,
                "attention-dropout": 0.0,
                "hidden-dropout": 0.0,
                "optimizer": "adam",
                "lr": 1e-6,
                "lr-decay-style": "constant",
                "use-rollout-logprobs": True,
                "use-critic": True,
                "critic-lr": 5e-6,
                "critic-steps-per-actor": 2,
                "num-critic-only-steps": 0,
                "sao-length-adaptive-lambda": True,
                "sao-lambda-alpha": 1.5,
                "gamma": 1.0,
                "lambd": 1.0,
                "eps-clip": 0.3,
                "eps-clip-high": 5.0,
                "n-samples-per-prompt": 1,
                "entropy-coef": 0.0,
            },
        },
    }
    (run / "config.yaml").write_text(yaml.safe_dump(raw))
    config, _ = resolve_deployment_config(raw, None, run / "config.yaml")
    settings = service_config_from_mapping(config)
    component, serving = connect_training_component(settings, model_path=model, loss_family="sao")
    try:
        import ray

        assert int(ray.cluster_resources()["GPU"]) == gpus
        assert component.resources.inference_placement == component.resources.placement_groups["actor"]
        tokenizer = AutoTokenizer.from_pretrained(model)
        start = component.base_start
        for step in range(2):
            version = serving.current_runtime_load_id()
            samples = []
            for index in range(2):
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"What is {index+2} plus 3?"}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
                handle = asyncio.run(serving.acquire_inference())
                try:
                    response = requests.post(
                        serving.base_url + "/generate",
                        json={
                            "input_ids": prompt,
                            "sampling_params": {"max_new_tokens": 32, "temperature": 0},
                            "return_logprob": True,
                        },
                        timeout=120,
                    )
                    response.raise_for_status()
                    result = response.json()
                finally:
                    handle.release()
                pairs = result["meta_info"]["output_token_logprobs"]
                assert pairs
                samples.append(
                    policy_trajectory(
                        f"{step}:{index}",
                        prompt + [pair[1] for pair in pairs],
                        [1] * len(pairs),
                        [pair[0] for pair in pairs],
                        1.0 if index == 0 else 0.0,
                        version,
                        action_mask=[1] * len(pairs),
                    )
                )
            job = TrainingJob(
                f"shared-smoke-{step}",
                component.name,
                start,
                JobData(f"smoke-{step}", tuple(samples)),
                {"method": TrainingMethod("sao").to_dict()},
            )
            component.submit(job)
            deadline = time.monotonic() + 600
            while True:
                status = component.status(job.job_id)
                if status.state != "running":
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("training job did not finish")
                time.sleep(1)
            assert status.state == "finished", status.error
            assert status.checkpoints
            assert serving.phase == "serving"
            assert serving.current_runtime_load_id() == version
            checkpoint = status.checkpoints[-1]
            assert checkpoint.metrics["sao/actor_trained"] == 1
            assert checkpoint.metrics["sao/critic_updates"] == 2
            # Version metadata alone cannot prove that offload preserved the
            # incumbent weights. Compare a real deterministic request before
            # publication with the same request sampled before training.
            handle = asyncio.run(serving.acquire_inference())
            try:
                response = requests.post(
                    serving.base_url + "/generate",
                    json={
                        "input_ids": prompt,
                        "sampling_params": {"max_new_tokens": 32, "temperature": 0},
                        "return_logprob": True,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                restored = response.json()["meta_info"]["output_token_logprobs"]
            finally:
                handle.release()
            assert [pair[1] for pair in restored] == [pair[1] for pair in pairs]
            assert all(
                math.isclose(before[0], after[0], rel_tol=1e-4, abs_tol=1e-4)
                for before, after in zip(pairs, restored, strict=True)
            )
            published = serving.activate_checkpoint(Artifact.local(Path(checkpoint.release_path)))
            assert published != version
            start = checkpoint.path
            with (run / "results.jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "step": step,
                            "previous_version": version,
                            "published_version": published,
                            "checkpoint": checkpoint.release_path,
                            "metrics": dict(checkpoint.metrics),
                        }
                    )
                    + "\n"
                )
            print(f"COLOCATE_STEP_PASS step={step} version={published}", flush=True)
        response = requests.post(
            serving.base_url + "/generate",
            json={"text": "1+1=", "sampling_params": {"max_new_tokens": 8}},
            timeout=120,
        )
        response.raise_for_status()
    finally:
        close_runtimes(component, serving)
    print("COLOCATE_GPU_PASS", flush=True)


if __name__ == "__main__":
    main()
