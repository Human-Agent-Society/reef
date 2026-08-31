"""Processors: records in, one typed training batch out.

Feedback either arrives as reports referencing inference records — where it is
whatever the report carries, scores, text, or structured objects, not only
numbers — or is mined from the traffic itself.

The design — the four-method contract, the two engines and the one question
that picks between them, what a recipe writes on each tier, and a record's
path to a batch — is at https://reefinfra.ai/docs/developer-guide/processors/.

Everything numeric about a method — advantages, loss family — lives in its
backend step preparer, not here: processors own the records and their
retention, preparers own the training signal.
"""

from reef.train.processors.base import DataProcessor, RetentionDecision
from reef.train.processors.computed import ComputedFeedbackProcessor
from reef.train.processors.reported import ReportedFeedbackProcessor

__all__ = [
    "ComputedFeedbackProcessor",
    "DataProcessor",
    "ReportedFeedbackProcessor",
    "RetentionDecision",
]
