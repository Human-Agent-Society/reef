Write a harness method
======================

Implement ``Method.run(job)`` to edit an agent's source tree.
Use ``HarnessEvolutionRecipe`` and select ``recipe.config.evolution.method``.
Reef validates and evaluates the result; activation decides whether it serves.

An example: add a tool
----------------------

This method proposes a pi extension from failed traces:

.. code:: python

   from reef.core.prompts import untrusted_text
   from reef.harness.tree import TreePatch
   from reef.train.harness_backend import Method, MethodJob


   class AddTool(Method):
       def run(self, job: MethodJob) -> None:
           workspace = job.workspace
           failing = [trace for trace in workspace.traces() if trace.score is not None and trace.score < 1.0]
           if not failing:
               workspace.write_output("No failing traces.")
               return
           traces = "\n\n".join(f"Task: {trace.task}\nReply: {trace.reply}" for trace in failing)
           prompt = (
               "Write one pi extension that uses pi.registerTool to help with these failures. "
               "Return only the TypeScript source.\n\n" + untrusted_text(traces)
           )
           code = job.services.models.served.chat([{"role": "user", "content": prompt}])
           name = f"extensions/tool-{workspace.step}.ts"
           workspace.write_tree(workspace.tree().apply(TreePatch({name: code.strip() + "\n"})))
           workspace.write_output(f"Proposed {name}.")

Read ``workspace.host()["parts"]`` before generating adapter-specific files.
An unchanged or invalid tree skips the job. Exceptions fail it without
publishing tree changes or method state.

Workspace and services
----------------------

- ``workspace.tree()`` and ``write_tree`` read/write the candidate.
- ``traces()``, ``tasks()`` and ``host()`` read job inputs; trace fields can be ``None``.
- ``state/`` holds persistent UTF-8 files, at most 16 MiB. Failed jobs save no state.
- ``scratch/`` holds temporary files; ``write_output(summary, notes)`` records diagnostics.
- ``services.models.served.chat(messages)`` calls the model under test.
  ``models[name]`` selects a configured model binding.
- ``services.check(tree)`` validates; ``evaluate(tree, tasks, repeats=1)`` returns scored runs.
- ``services.run_agent`` runs the coding agent through Reef's budgeted gateway.

Model calls and evaluation obey ``max_model_calls_per_step`` and ``max_runs_per_step``.
Reused runs are free. Failed evaluation admission runs no tasks.
Constructors receive settings only; access models and credentials through ``job.services``.
Set ``runs_agent=True`` when an agent is required.
See the `method contract <../../reef/train/harness_backend/contract.py>`__.

Evaluation
----------

Configure tasks, a scorer and publication policy.
Runs are reused only with matching evaluation settings.
A ``TaskScorer`` returns finite scores; higher is better:

.. code:: python

   from reef.evaluation.harness import TaskScorer
   from reef.harness.runs.trajectory import final_assistant_text


   class ExactAnswer(TaskScorer):
       def score(self, task, result):
           text = final_assistant_text(result.trajectory) or ""
           return 1.0 if text.strip().endswith("2880067194370816120") else 0.0

This scorer matches the Fibonacci task below.
Model judges implement ``score_with_models`` as well as ``score``.
Declare GPU requirements and load models lazily in workers.

Collect traffic prompts
~~~~~~~~~~~~~~~~~~~~~~~

``collect_failures=true`` adds trace prompts to evaluation.
Optional ``TaskCollector.collect(samples)`` filters them; otherwise every trace prompt is kept.

Configure a deployment
----------------------

.. code:: yaml

   recipe:
     implementation: reef.recipe.harness:HarnessEvolutionRecipe
     config:
       evolution:
         adapter: pi
         seed: methods/seed
         method: methods.add_tool:AddTool
         evaluation:
           tasks:
             - "Compute fib(90) exactly. Reply with the integer alone on the last line."
           scorer: methods.scoring:ExactAnswer
           policy: compare
   activation:
     module: always
     config:
       review_kinds: [extension]
   inference:
     upstream-url: ${REEF_UPSTREAM_URL}
     upstream-model: qwen3-8b

Run ``reef serve -c CONFIG`` where ``methods`` is importable.
The seed must pass validation. This configuration holds extension changes for manual activation.
Invalid configuration raises ``RecipeConfigError`` or ``PluginConfigError``.

Optional method state
---------------------

``workspace.history()`` reads prior trees, runs and decisions.
``workspace.step`` counts ended jobs plus one, including failures.
Ignore entries at or beyond the current step; an unsettled retry reuses its step.

For prompt-only methods, use ``AgentMethod`` with a required ``START.md`` and
optional ``RESUME.md``. See its `agent protocol <../../reef/train/harness_backend/agent/protocol.md>`__.
Custom settings use ``from_config`` and ``PluginSettings``.

Untrusted input
---------------

Wrap trace text in ``untrusted_text``. Keep credentials out of tree files.
Require review for generated executable parts; native loops always require review.
Client tags identify users only when a trusted gateway supplies them.
