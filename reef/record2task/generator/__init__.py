"""The generator service: a served model writes Harbor tasks, and Reef checks and plays them.

The generated-task recipe (``reef.recipe.generated_tasks``) and the deployment's task generator
(:class:`reef.core.tasks.TaskGenerator`) run inside the Reef service, where nothing may block the trainer
and nothing should need Docker. The slow, container bound steps therefore run in the generator service, a
process ``reef serve`` starts beside the HTTP service when a deployment carries a ``generator`` section,
and the recipe and the generator drive it over HTTP:

- ``designer``: the task contract as a prompt, the served model asked for one task through Reef (every
  proposal an inference record with a receipt), the reply parsed into the instruction, the files and a hint.
- ``harbor``: the reply held to the authoring rules, the task written with ``reef.core.tasks``, tasks
  deduplicated by content, Harbor's oracle and nop agents run through the ``harbor`` command line, a split
  per generation.
- ``terminus``: Terminus 2 for the generator's plays, whose episode is still verified when the agent fills its
  context or a send-keys to its terminal fails (selected by ``import_path``; the only module here that imports
  harbor).
- ``service``: the HTTP API and its job runner; ``client``: the same steps as asynchronous calls; ``wire``:
  the JSON forms. The ``generator`` section that starts the service is parsed in
  ``reef.service.deploy.generator``, and ``reef.service.deploy.generator_process`` runs the process.

What is method policy, the prompt's experience section, the measurement arms a task asks for and the score a
proposal earns, stays with the method's task generator (``recipes/beta/spade`` is the first).
"""

from reef.record2task.generator.client import (
    DuplicateTask,
    Generator,
    GeneratorError,
    HttpGenerator,
    ProposedTask,
    TaskNameConflict,
    WrittenTask,
)
from reef.record2task.generator.designer import (
    Designer,
    DesignerAnswer,
    DesignerError,
    DesignerReplyError,
    DesignerRequest,
    HarborReply,
    ReefDesigner,
    designer_messages,
    designer_prompt,
    parse_harbor_reply,
)
from reef.record2task.generator.harbor import (
    GeneratedHarborTask,
    HarborRuns,
    OracleResult,
    OracleUnavailable,
    content_hash,
    harbor_task,
    oracle_check,
    reply_errors,
)
from reef.record2task.generator.service import (
    GeneratorService,
    HarborChecks,
    JobRunner,
    ReadinessProbe,
    ReefTaskPlays,
    TaskChecks,
    TaskPlays,
    readiness_probes,
)

__all__ = [
    "Designer",
    "DesignerAnswer",
    "DesignerError",
    "DesignerReplyError",
    "DesignerRequest",
    "DuplicateTask",
    "GeneratedHarborTask",
    "Generator",
    "GeneratorError",
    "GeneratorService",
    "HarborChecks",
    "HarborReply",
    "HarborRuns",
    "HttpGenerator",
    "JobRunner",
    "OracleResult",
    "OracleUnavailable",
    "ProposedTask",
    "ReadinessProbe",
    "ReefDesigner",
    "ReefTaskPlays",
    "TaskChecks",
    "TaskNameConflict",
    "TaskPlays",
    "WrittenTask",
    "content_hash",
    "designer_messages",
    "designer_prompt",
    "harbor_task",
    "oracle_check",
    "parse_harbor_reply",
    "readiness_probes",
    "reply_errors",
]
