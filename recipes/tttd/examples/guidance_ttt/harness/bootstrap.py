from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from .library import GuidanceLibrary
from .state import LibraryEntry
from .tasks import TaskSpec
from .verifier.config import sanitize_verifier_config


def create_verified_baseline_seed(
    output_path: str | Path,
    *,
    task: TaskSpec,
    verifier_timeout_s: int,
    verifier_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a task's pinned baseline and materialize a pristine Reef archive."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to replace existing Guidance-TTT seed: {output_path}")
    if not task.bootstrap_solution or not task.bootstrap_summary:
        raise ValueError(f"task {task.task_id!r} has no packaged baseline seed")

    language = "cpp" if task.solution_language.lower() in {"cpp", "c++", "cxx"} else "python"
    execution_text = (
        f"<solution>\n```{language}\n{task.bootstrap_solution.strip()}\n```\n</solution>\n\n"
        f"<summary>\n{task.bootstrap_summary.strip()}\n</summary>"
    )
    verification = task.verify_execution_text(
        execution_text,
        timeout_s=verifier_timeout_s,
        config=dict(verifier_config or {}),
    )
    if not verification.valid or verification.raw_score is None:
        raise RuntimeError(
            f"packaged baseline for {task.task_id!r} failed verification: "
            f"{verification.status}: {verification.message}"
        )

    root = task.create_root_node()
    root.metadata.update({"task": task.task_id, "score_direction": task.score_direction})
    library = GuidanceLibrary(output_path, initial_nodes=[root], score_direction=task.score_direction)
    entry = LibraryEntry(
        id=str(uuid4()),
        parent_id=root.id,
        problem_id=task.task_id,
        timestep=0,
        guidance="Bootstrap from the pinned official task baseline.",
        execution_thinking="",
        solution=task.bootstrap_solution,
        verifier_reward=verification.reward,
        verifier_raw_score=verification.raw_score,
        verifier_status=verification.status,
        verifier_message=verification.message,
        summary=task.bootstrap_summary,
        reusable_idea=task.bootstrap_summary,
        failure_mode=None,
        metadata={
            "bootstrap": True,
            "bootstrap_source": "task_baseline",
            "prompt_mode": "summary_only",
            "summary_semantics": "canonical_full_candidate",
            "raw_model_summary": task.bootstrap_summary,
            "execution_text": execution_text,
            "verification_artifacts": verification.artifacts,
            "task": {
                "id": task.task_id,
                "verifier": sanitize_verifier_config(verifier_config or {}),
            },
        },
    )
    library.attach_entry_to_root(root.id, entry)
    return library.snapshot()
