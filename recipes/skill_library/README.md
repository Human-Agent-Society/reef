# skill_library

A Reef harness evolution method that keeps a library of procedural skills, `skills/sl-<slug>/SKILL.md`. Each step
induces at most one candidate skill from each scored run, and keeps a candidate only when a replay of the run's task
scores higher with the skill than without it. Every skill must carry a check that the agent runs before it trusts a
result, together with a known bad result on which the check fails. Skills that later replays show to hurt leave the
library.

The method combines three published ideas. The skills are induced from runs, as workflows are in
[Agent Workflow Memory](https://arxiv.org/abs/2409.07429). A skill joins only after it is run and succeeds, as in
[Voyager](https://arxiv.org/abs/2305.16291) and [SkillWeaver](https://arxiv.org/abs/2504.07079). A skill keeps a count
of the outcomes for and against it and leaves when the count turns against it, as an insight does in
[ExpeL](https://arxiv.org/abs/2308.10144). The survey that led to this design is in
[#1086](https://github.com/Human-Agent-Society/reef/issues/1086).

- Method: `recipes.skill_library:SkillLibrary` in [method.py](method.py), with its prompt in
  [prompts/induce.md](prompts/induce.md)
- Example deployment: [examples/serve.yaml](examples/serve.yaml), on pi with `policy: compare`
- Results on ServeLearnBench `banking_l1` and `retail_l1` with GLM-5.3-Flash:
  [examples/servelearnbench/results/glm-5.3-flash-2026-11-09/](examples/servelearnbench/results/glm-5.3-flash-2026-11-09/README.md)
- Tests: [tests/recipes/skill_library/](../../tests/recipes/skill_library/)

## What one step does

1. **Find the runs.** A run is a trace with a task and a score. The step handles the runs with the lowest score
   first, because a replay can show a gain only where the library fails. Unscored traces are skipped.
2. **Induce a candidate.** The extractor model reads the run and the library's names and descriptions. It returns at
   most one skill (`name`, `description`, `procedure`, `check`, `known_bad`) and the names of the library's skills
   that the run implicates in a failure. A skill without a check or without a known bad case is refused. A name that
   the library holds revises that skill. The prompt keeps the run's names and identifiers out of the skill, but it
   lets the skill state a value, such as a limit, when the run's score shows that the documented value is not the
   one in force. Without that, a skill can only restate the documents, and it cannot carry what the documents get
   wrong.
3. **Count against implicated skills.** When a failed run implicates a skill and its task can be replayed, the step
   runs the task without the skill and with the full library. A library that scores higher without the skill gives
   the skill a loss, and a library that scores higher with it gives a win. A tie gives neither, because a task that
   fails with and without the skill says nothing about it. At most `max_ablations_per_step` such replays run per
   step.
4. **Verify the candidate.** When the run failed and its task can be replayed, the step runs the task with the
   library plus the candidate and with the library alone. The candidate joins (or replaces its old text) only when its mean score is
   higher by more than `min_margin`, and it is marked verified with one more win.
5. **Without a replay.** A run that reached `success_score` is not replayed, because its task leaves no room to
   show a gain. Its candidate joins unverified if `keep_unverified` is set. A candidate from a failed run whose task
   cannot be replayed does not join.
6. **Retire and cap.** A skill whose losses pass its wins by `max_losses` leaves. Past `max_skills`, the weakest
   skills leave: unverified before verified, then fewer wins minus losses, then the oldest.
7. **Write the tree.** The step writes one skill per library entry and removes every other `skills/sl-*` skill. The
   other files of the tree do not change. When the new tree fails `services.check`, neither the tree nor the library
   changes.

A replay runs a task with `Services.evaluate`, so it needs a scorer and the task among the step's evaluation tasks: a
run's task text must equal an evaluation prompt, or the instruction of an evaluation task directory. Reef reuses an
earlier run of the same tree, task and repeat, so the library's own replays are often free. When the services refuse
a replay (the step's run budget is spent), no further replay runs in that step. The step's notes list every run with
what happened to it, the replays with their mean scores, and the retired and dropped skills.

The library is the method's state, `state/skills.json`: a list of skills `{slug, name, description, procedure, check,
known_bad, verified, wins, losses, step}`, oldest first.

## Skills in the tree

```markdown
---
name: sl-recompute-totals-from-rows
description: Use when a task asks for a total that is derived from line items.
---

# Recompute totals from rows

- Sum the line items yourself before you answer.
- Report the computed total, not the stated one.

## Check before you trust the result

Recompute the total from the rows and compare it with your answer.

The check must fail on this known bad result: A total that differs from the sum of its rows.
```

The description is the retrieval key, as in `reasoning_bank`: the harness lists every skill's name and description,
and the agent opens the skills that fit its task. Every Reef adapter renders skills, so the method runs on every
adapter.

## Configuration

The settings go under `evolution.method.config`:

| Setting | Default | What it sets |
| --- | --- | --- |
| `extract_model` | `served` | The model that writes a candidate from a run: `served` or a name under `evolution.models`. |
| `success_score` | `1.0` | A run with at least this score is a success. |
| `repeats` | `0` | Runs per tree in a replay. `0` takes `evaluation.repeats`. |
| `min_margin` | `0.0` | A replay keeps a candidate when its mean beats the library's by more than this. |
| `keep_unverified` | `true` | Keep a candidate from a successful run whose task cannot be replayed, marked unverified. |
| `max_runs_per_step` | `0` | The most runs that one step learns from. `0` sets no limit. |
| `max_ablations_per_step` | `2` | The most implicated skills that one step replays without. |
| `max_losses` | `2` | A skill leaves when its losses pass its wins by this many. |
| `max_skills` | `30` | The most skills the library keeps. `0` sets no limit. |

Each run costs one extractor call. A replay costs `2 × repeats` runs, less the runs Reef already has. Set
`evolution.max_runs_per_step` to bound the runs of one step.

## Deploy it

[examples/serve.yaml](examples/serve.yaml) runs the method on pi with the tutorial's tasks and scorer:

```bash
REEF_UPSTREAM_URL=... REEF_UPSTREAM_MODEL=... reef serve -c recipes/skill_library/examples/serve.yaml
```

Each scored report is one step. A failed report on one of the evaluation prompts is verified by a replay. A
successful report adds its skill unverified, and a failed report on any other prompt adds nothing. `policy: compare` then compares the whole new
library with the head on every task before the release, which catches skills that each help alone but conflict
together. Set `keep_unverified: false` to keep only verified skills.
