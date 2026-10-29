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
from reef.surface.adapter import adapter_name


def generate(serving, url, prompt, scenario):
    version = serving.serving_adapter_runtime_load_id(scenario)
    payload = {
        "input_ids": prompt,
        "sampling_params": {"max_new_tokens": 32, "temperature": 0},
        "return_logprob": True,
    }
    if version is not None:
        payload["lora_path"] = adapter_name(scenario, version)
    handle = asyncio.run(serving.acquire_inference())
    try:
        response = requests.post(url + "/generate", json=payload, timeout=120)
        response.raise_for_status()
        pairs = response.json()["meta_info"]["output_token_logprobs"]
        assert pairs
        return pairs
    finally:
        handle.release()


def same_generation(before, after):
    assert [pair[1] for pair in before] == [pair[1] for pair in after]
    assert all(
        math.isclose(left[0], right[0], rel_tol=1e-4, abs_tol=1e-4) for left, right in zip(before, after, strict=True)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run = Path(os.environ["SMOKE_RUN"])
    run.mkdir(parents=True, exist_ok=True)
    model = os.environ["MODEL_DIR"]
    gpus = int(os.environ.get("SMOKE_GPUS", "2"))
    inference_tp = int(os.environ.get("SMOKE_INFERENCE_TP", str(gpus)))
    lora_rank = int(os.environ.get("SMOKE_LORA_RANK", "0"))
    scenario = "smoke"
    raw = {
        "recipe": {"implementation": "recipes.sao.recipe:SAORecipe"},
        "inference": {
            "model-path": model,
            "num-gpus": gpus,
            "tensor-parallel-size": inference_tp,
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
    if lora_rank:
        raw["training"]["options"].update(
            {"megatron-lora-rank": lora_rank, "megatron-lora-alpha": lora_rank, "lr": 1e-4}
        )
    (run / "config.yaml").write_text(yaml.safe_dump(raw))
    config, _ = resolve_deployment_config(raw, None, run / "config.yaml")
    settings = service_config_from_mapping(config)
    component, serving = connect_training_component(settings, model_path=model, loss_family="sao")
    try:
        import ray

        assert int(ray.cluster_resources()["GPU"]) == gpus
        assert component.resources.inference_placement == component.resources.placement_groups["actor"]
        tokenizer = AutoTokenizer.from_pretrained(model)
        workers = requests.get(serving.base_url + "/workers", timeout=30)
        workers.raise_for_status()
        urls = [worker["url"] for worker in workers.json()["workers"]]
        assert len(set(urls)) == gpus // inference_tp, urls
        print(f"COLOCATE_ENGINES urls={urls} lora_rank={lora_rank}", flush=True)
        start = component.base_start
        for step in range(2):
            version = serving.serving_adapter_runtime_load_id(scenario) or serving.current_runtime_load_id()
            samples = []
            for index in range(2):
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"What is {index+2} plus 3?"}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
                pairs = generate(serving, serving.base_url, prompt, scenario)
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
            before_engines = [generate(serving, url, prompt, scenario) for url in urls]
            for before in before_engines:
                same_generation(pairs, before)
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
            assert (serving.serving_adapter_runtime_load_id(scenario) or serving.current_runtime_load_id()) == version
            checkpoint = status.checkpoints[-1]
            assert checkpoint.metrics["sao/actor_trained"] == 1
            assert checkpoint.metrics["sao/critic_updates"] == 2
            # Version metadata alone cannot prove that offload preserved the
            # incumbent weights. Compare a real deterministic request before
            # publication with the same request sampled before training.
            for url, before in zip(urls, before_engines, strict=True):
                same_generation(before, generate(serving, url, prompt, scenario))
            same_generation(pairs, generate(serving, serving.base_url, prompt, scenario))
            published = serving.activate_checkpoint(Artifact.local(Path(checkpoint.release_path)), scenario=scenario)
            after_engines = [generate(serving, url, prompt, scenario) for url in urls]
            for after in after_engines[1:]:
                same_generation(after_engines[0], after)
            if lora_rank:
                from safetensors.torch import load_file

                weights = load_file(str(Path(checkpoint.release_path) / "adapter" / "adapter_model.safetensors"))
                assert any(tensor.count_nonzero().item() > 0 for name, tensor in weights.items() if "lora_B" in name)
            assert published != version
            start = checkpoint.path
            with (run / "results.jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "step": step,
                            "inference_tp": inference_tp,
                            "engines": len(urls),
                            "lora_rank": lora_rank,
                            "previous_version": version,
                            "published_version": published,
                            "checkpoint": checkpoint.release_path,
                            "metrics": dict(checkpoint.metrics),
                        }
                    )
                    + "\n"
                )
            print(f"COLOCATE_STEP_PASS step={step} version={published}", flush=True)
        generate(serving, serving.base_url, prompt, scenario)
    finally:
        close_runtimes(component, serving)
    print("COLOCATE_GPU_PASS", flush=True)


if __name__ == "__main__":
    main()
