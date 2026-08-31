## Role

You are a Python and reinforcement-learning systems engineer optimizing a CPU-only locomotion agent.

## Project

Read `task_prompt.md`, `README.md`, `baseline/train_baseline.py`, and `envs/locomotion_env.py`. The editable submission area is `submission/`.

## Objective

Train a BipedalWalker policy on CPU in this work container and save the final trained checkpoint as `submission/policy.pth`. The trusted judge does not run training; it only receives the submitted checkpoint and evaluates it on BipedalWalker-v3 and BipedalWalkerHardcore-v3.

## Constraints

- Keep your final implementation and checkpoint under `submission/`.
- `submission/policy.pth` is submitted and required. Keep intermediate checkpoints outside `submission/` or under `submission/checkpoints/`.
- You may provide `submission/policy.py` for custom PyTorch architectures; the judge supports `load_policy(path)`, `build_policy(checkpoint)`, or a `Policy` class.
- Do not download or use pretrained policies, external RL libraries, or hidden evaluator files.
- Use CPU only; CUDA is disabled during evaluation.
- Hidden evaluator code is not present in this work container.
- Trusted feedback is aggregate-only: reward summaries and scoring components are shown, but evaluator internals are not exposed.
- Do not start with a long training run. First run smoke tests and short training loops; keep early iterations under 10 minutes, submit, then scale up once checkpoint loading works.
- Submit whenever you have a candidate `submission/policy.pth` and want trusted feedback.

## Time-Budget Workflow

Use the available 2-hour agent window for active improvement. Do not stop after the first loadable checkpoint; submit within the first 30 minutes, then continue training, tuning, debugging, and submitting stronger checkpoints until you are close to the time limit. Do not idle or sleep just to consume wall-clock time.

## Deliverable

A final `submission/policy.pth`, plus any required `submission/policy.py`/`submission/train.py`, that the judge can load and evaluate.