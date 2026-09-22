"""Records to datasets: a served model writes Harbor tasks, Reef checks and plays them, a processor trains on the episodes.

The processor that generates tasks (:class:`reef.train.processors.TaskGenerationProcessor`) runs inside
the Reef service, where nothing may block the trainer and nothing should need Docker. The slow, container
bound steps therefore run in the generator service, a process ``reef serve`` starts beside the HTTP
service when a deployment carries a ``generator`` section, and the processor drives it over HTTP:

- ``designer``: the task contract as a prompt, the served model asked for one task through Reef (every
  proposal an inference record with a receipt), the reply parsed into the instruction, the files and a hint;
  a generation's turn at a Designer whose deployment evolves between generations.
- ``harbor``: the reply held to the authoring rules, the task written with ``reef.core.tasks``, tasks
  deduplicated by content, Harbor's oracle and nop agents run through the ``harbor`` command line, a split
  per generation.
- ``service``: the HTTP API and its job runner; ``client``: the same steps as asynchronous calls for a
  processor; ``wire``: the JSON forms. The ``generator`` section that starts the service is parsed in
  ``reef.service.deploy.generator``, beside the process it describes.

What is method policy, the prompt's experience section, the arms a task is played with and the score a
proposal earns, stays with the method's processor (``recipes/beta/spade`` is the first).
"""

from reef.record2dataset.client import (
    DuplicateTask,
    Generator,
    GeneratorError,
    HttpGenerator,
    ProposedTask,
    TaskNameConflict,
    WrittenTask,
)
from reef.record2dataset.designer import (
    Designer,
    DesignerAnswer,
    DesignerError,
    DesignerPrompt,
    DesignerReplyError,
    DesignerRequest,
    DesignerTurn,
    FixedPrompt,
    HarborReply,
    HarnessPrompt,
    PromptSource,
    ReefDesigner,
    designer_messages,
    designer_prompt,
    parse_harbor_reply,
)
from reef.record2dataset.harbor import (
    GeneratedHarborTask,
    HarborRuns,
    OracleResult,
    OracleUnavailable,
    content_hash,
    harbor_task,
    oracle_check,
    reply_errors,
    split_generation,
)
from reef.record2dataset.service import (
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
    "DesignerPrompt",
    "DesignerReplyError",
    "DesignerRequest",
    "DesignerTurn",
    "DuplicateTask",
    "FixedPrompt",
    "GeneratedHarborTask",
    "Generator",
    "GeneratorError",
    "GeneratorService",
    "HarborChecks",
    "HarborReply",
    "HarborRuns",
    "HarnessPrompt",
    "HttpGenerator",
    "JobRunner",
    "OracleResult",
    "OracleUnavailable",
    "PromptSource",
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
    "split_generation",
]
