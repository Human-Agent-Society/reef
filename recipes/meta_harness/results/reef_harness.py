from harbor.agents.terminus_2 import Terminus2


class Agent(Terminus2):
    """Terminus2 with a completion checkpoint based on observed results."""

    _REVIEW_GATE = """

MANDATORY FALSIFICATION CHECKPOINT
Do not merely repeat the completion flag. Act as an independent reviewer of the current result and use terminal commands now unless the displayed output already demonstrates that every requirement is met.

Before confirming completion:
1. Re-read the original task and form a concise acceptance matrix covering every explicit behavior, exact path/name/schema/interface, boundary condition, and negative constraint.
2. Inspect the final artifacts and workspace state. Check for stale generated files, accidental edits, or extra files that violate the requested deliverable.
3. Run the smallest high-information behavioral checks through the exact public interface the grader or user will invoke. Exercise representative edge cases and interactions between requirements. Checking that a file exists, compiling/importing it, or testing one happy path alone does not establish correctness.
4. If a dependency is absent, try an available package/environment tool or construct a faithful minimal probe from documented public APIs rather than assuming runtime compatibility. Do not inspect hidden tests, private verifier material, solution artifacts, or reward files.
5. Read command exit statuses and outputs. If any observation contradicts the acceptance matrix, fix the implementation and rerun the relevant checks. Do not rewrite a passing solution without a concrete failing observation.

Only repeat the completion flag after observed command results demonstrate that all rows of the acceptance matrix are satisfied. Otherwise issue the review or repair commands in this response.
"""

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        base_message = super()._get_completion_confirmation_message(terminal_output)
        return base_message + self._REVIEW_GATE
