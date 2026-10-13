"""Record to task: the stage that turns one record set into tasks.

Reef reuses records in a pipeline of three stages::

    record ─► record2record ─► record set ─► record2task ─► task ─► task2batch ─► batch

``reef.record2record`` decides which records Reef uses and forms record sets.
This package turns each set into tasks. A task is a ``TrainDataItem``: a
``TrajectoryItem`` that trains directly, or a ``TaskItem`` that the algorithm
rolls out. A recipe's ``Task2Batch`` (``reef.task2batch``) shapes the
tasks into batches, and ``reef.train.pipeline.TrainingPipeline`` drives the
three stages.

Entry point: ``Record2Task`` in ``base.py``. The recipe's ``pipeline_spec()``
names the class, and ``from_config`` builds it for each scenario.

Implementations:

* ``ReportTasks`` (``reports.py``): one valid report and its inference
  records into one task. A recipe writes ``make_sample``.
* ``JudgedTasks`` (``judged.py``): one judged record into one task. A recipe
  writes ``make_sample``.
* ``DistillTasks`` (``distill.py``): one report into one distillation task,
  the student's sample plus the teacher sequence. ``TeacherRenderer`` renders
  the teacher's request with the served model's chat template.
* ``GeneratedTaskTasks`` (in the built-in recipe ``reef.recipe.generated_tasks``):
  new Harbor tasks from a description and, optionally, from recorded sessions.
  Each round generates tasks, checks them, and plays them; the training plays
  come back as reports and become tasks.

Shared parts of ``make_sample``:

* ``SampleAssembly`` (``assembly.py``): a report and its inference records into
  one policy sample, or one multi-turn sample.
* ``samples.py``: readers of recorded inferences and the policy sample builders.

The subpackage ``generator``, which this module does not import, is the
generator service. It runs the slow steps that need containers (the designer
prompt, the authoring rules, the oracle check, and the task plays) in a
process beside the HTTP service.

The modules in this directory depend only on ``reef.core``,
``reef.record2record``, and ``reef.pipeline_context``. Thus the trainer can use
them without the generator service.
"""

from reef.record2task.base import NoTasks, Record2Task, TaskResult
from reef.record2task.judged import JudgedTasks
from reef.record2task.reports import ReportTasks, SkippedReport

__all__ = [
    "JudgedTasks",
    "NoTasks",
    "Record2Task",
    "ReportTasks",
    "SkippedReport",
    "TaskResult",
]
