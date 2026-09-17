"""Processors: records in, one typed training batch out.

Feedback either arrives as reports referencing inference records — where it is
whatever the report carries, scores, text, or structured objects, not only
numbers — or is mined from the traffic itself. ``TaskGenerationProcessor``
declares asynchronous generation and validation hooks for task-producing
processors; a method pairs them with an engine and a worker of its own, as
SPADE does with the reported engine and the generator service.
``TeacherSequenceProcessor`` is the reported engine of the distilling
recipes: the student's rollout plus the teacher's prompt, which a recipe
composes from the recorded request and the report's context.

The design — the four-method contract, the two engines and the one question
that picks between them, what a recipe writes on each tier, and a record's
path to a batch — is at https://reefinfra.ai/docs/developer-guide/processors/.

Everything numeric about a method — advantages, loss family — lives in its
backend training objective, not here: processors own the records and their
retention, objectives own the training signal.
"""

from reef.train.processors.base import DataProcessor, RetentionDecision
from reef.train.processors.computed import ComputedFeedbackProcessor
from reef.train.processors.reported import ReportedFeedbackProcessor
from reef.train.processors.task_generation import TaskGenerationProcessor
from reef.train.processors.teacher_sequence import TeacherSequenceProcessor

__all__ = [
    "ComputedFeedbackProcessor",
    "DataProcessor",
    "ReportedFeedbackProcessor",
    "RetentionDecision",
    "TaskGenerationProcessor",
    "TeacherSequenceProcessor",
]
