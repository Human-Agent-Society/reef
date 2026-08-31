"""CORAL ``TaskGrader`` implementation backed by reef_eval."""

from pathlib import Path

from coral.grader import TaskGrader
from coral.types import Score, ScoreBundle

from reef_eval_coral_grader.judge import SubmissionBudgetExhausted, submit_file


class Grader(TaskGrader):
    def evaluate(self) -> ScoreBundle:
        solution = Path(self.codebase_path) / self.args.get(
            "solution_file", "solution.json"
        )
        try:
            result = submit_file(solution)
        except SubmissionBudgetExhausted as error:
            return ScoreBundle(
                scores={"score": Score(value=0.0, name="score")},
                aggregated=0.0,
                feedback=str(error),
            )
        score = float(result["score"])
        feedback = str(result.get("reason") or result)
        return ScoreBundle(
            scores={"score": Score(value=score, name="score")},
            aggregated=score,
            feedback=feedback,
            metadata={"remaining": result.get("remaining")},
        )
