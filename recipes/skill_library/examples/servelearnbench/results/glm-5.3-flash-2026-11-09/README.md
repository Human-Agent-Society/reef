# skill_library on ServeLearnBench, GLM-5.3-Flash (2026-11-09)

The recipe under `recipes/skill_library/` on two ServeLearnBench scenarios, `banking_l1` (336 serving tasks in 4
windows, 178 test tasks) and `retail_l1` (720 serving tasks in 6 windows, 384 test tasks), against the benchmark's
Blind, Oracle and RAG arms and against `recipes/reasoning_bank`. Three repeats per arm. This folder keeps the scores
of every run and the final skill library of every skill_library run. The full episode logs (about 400 MB) are kept
offline and are available on request.

## Setup

- Model: `zai-org/GLM-5.3-Flash` as the agent and as every method's extractor and judge, served by vLLM 0.33.0 with
  tensor parallel 8 on one 8xH100 node, `--tool-call-parser glm47 --reasoning-parser glm47` and MTP speculative
  decoding (3 tokens). The agent and the extractors run at reasoning effort high. GLM's chat template falls back to
  max when no effort is set.
- Benchmark: ServeLearnBench at `57aa919b`, its own agent loop and grader, at most 50 tool calls per episode.
- Steps: each block of 12 serving episodes becomes one Reef step on a real `Workspace` with `StepServices`, and the
  recipe's own `Method.step` runs on the block's scored traces. `evaluate` replays a serving task of the block through
  the benchmark's agent loop and grader with the given tree. The other arms only see the serving stream.
- Code: `skill_library_replay` is the recipe of this PR (with the `</think>` parser fix and the tied-ablation fix).
  `skill_library_replay_f97bff77` and `skill_library_no_replay` ran the recipe at f97bff77, before both fixes. A run
  without replay never ablates, so the tie fix cannot change it. `reasoning_bank` ran `recipes/reasoning_bank` at
  f97bff77 with the same extractor as its judge.

## Results

H is the mean reward x 100 on hidden-dependent test tasks, F on fully specified ones, mean ± sample sd over the three
repeats.

| arm | banking H | banking F | retail H | retail F |
| --- | --- | --- | --- | --- |
| Blind | 0.0 ± 0.0 | 94.7 ± 1.1 | 10.8 ± 3.1 | 100.0 ± 0.0 |
| Oracle | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.3 ± 0.5 | 100.0 ± 0.0 |
| RAG | 69.4 ± 5.0 | 92.2 ± 1.6 | 51.7 ± 2.6 | 95.6 ± 1.1 |
| ReasoningBank | 0.0 ± 0.0 | 88.7 ± 3.7 | 14.2 ± 4.4 | 98.4 ± 1.1 |
| skill_library, no replay | 0.0 ± 0.0 | 96.5 ± 3.3 | 1.4 ± 0.2 | 99.5 ± 0.4 |
| skill_library, replay | 21.0 ± 7.0 | 96.5 ± 4.4 | 23.1 ± 1.6 | 99.8 ± 0.4 |
| skill_library, replay, f97bff77 | 38.1 ± 17.6 | 90.8 ± 8.6 | | |

The PR description discusses these numbers. The replays are extra graded episodes beyond the serving stream (about
230 per banking run and 650 per retail run), which the benchmark's leaderboard rules do not allow.

## Files

- `scores.json`: for each scenario, arm and repeat, H and F overall and per window, the number of test tasks, and for
  the learning arms the steps, the skills or memory items at the end, the replay episodes, and the extractor calls and
  tokens. For the skill_library arms it also has the verified skills at the end, the retired skills and the ablation
  outcomes (a loss when the library did better without the skill, a win when it did worse, otherwise a tie).
- `libraries/<scenario>-<arm>-rep<k>.json`: the final `state/skills.json` of each skill_library run. Each skill has its
  slug, name, description, procedure, check and known bad result, whether a replay verified it, its wins and losses,
  and the step that wrote its current text. The recipe serves each one as `skills/sl-<slug>/SKILL.md`.

## Rerunning

Serve the model with vLLM as above, then run each arm through ServeLearnBench's `Runner` with block size 12. For a
Reef arm, a small `Method` adapter builds a `Workspace` per block from the block's episodes (instruction as the user
message, the episode record and its score line as the response), runs the recipe's `step`, and serves the new tree's
`skills/*/SKILL.md` in the agent's context for the next block. That adapter and the episode logs are not in the
repository. I can share both.
