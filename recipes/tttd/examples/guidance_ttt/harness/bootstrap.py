"""Verify a task's initial program before it enters a new search archive."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from .config import RunConfig
from .library import GuidanceLibrary
from .scorer import JudgeScorer
from .state import LibraryEntry, make_root_node


def prepare_seed(config: RunConfig, scorer: JudgeScorer) -> Path:
    """Re-score the seed with this deployment's judge; never invent a seed score."""
    if config.task == "polyomino_packing" and "GUIDANCE_SEED" not in os.environ:
        payload = json.loads((config.task_dir / "solution/gpt_oss_120b_bootstrap_library.json").read_text())
        entry = next(iter(payload["entries"].values()))
        code, summary = entry["solution"], entry["summary"]
    else:
        suffix = ".cpp" if config.contract().solution_language == "cpp" else ".py"
        seed = Path(os.environ.get("GUIDANCE_SEED", config.state_dir / f"bootstrap{suffix}"))
        if not seed.is_file():
            raise FileNotFoundError(f"missing bootstrap: {seed}; run prepare.py or set GUIDANCE_SEED")
        code = seed.read_text()
        summary_path = seed.with_suffix(".md")
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing canonical bootstrap summary: {summary_path}")
        summary = summary_path.read_text().strip()
    if not code.strip() or not summary.strip():
        raise ValueError("bootstrap code and summary must be non-empty")
    result = scorer(code)
    if not result.valid or result.raw_score is None:
        raise RuntimeError(f"bootstrap failed the task judge: {result.status}: {result.message}")
    seed_path = config.state_dir / "verified-bootstrap-library.json"
    # A failed startup can leave an unused seed. Replace only this freshly verified seed.
    seed_path.unlink(missing_ok=True)
    root = make_root_node(problem_id=config.task, raw_score=result.raw_score, reward=result.reward)
    library = GuidanceLibrary(
        seed_path,
        initial_nodes=[root],
        rollout_n=config.rollouts,
        groups_per_batch=config.groups,
        discover_compat=True,
        puct_q_mode="best_child",
        topk_children=2,
        score_direction=config.contract().score_direction,
    )
    library.attach_entry_to_root(
        root.id,
        LibraryEntry(
            id=str(uuid4()),
            parent_id=root.id,
            problem_id=config.task,
            timestep=0,
            guidance="",
            execution_thinking="",
            solution=code,
            verifier_reward=result.reward,
            verifier_raw_score=result.raw_score,
            verifier_status="valid",
            verifier_message=result.message,
            summary=summary,
            reusable_idea=summary,
            failure_mode=None,
            metadata={"bootstrap": True, "verification_artifacts": result.artifacts},
        ),
    )
    return seed_path
