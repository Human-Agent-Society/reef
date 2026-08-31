"""Summary-only prompts and strict response parsing for Guidance-TTT."""

from __future__ import annotations

from dataclasses import dataclass

from .state import LibraryEntry, LibraryNode


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str


def _score_for_prompt(entry: LibraryEntry, *, raw_score_label: str) -> str:
    raw_score = "None" if entry.verifier_raw_score is None else repr(float(entry.verifier_raw_score))
    return "\n".join(
        [
            f"Verifier status: {entry.verifier_status}",
            f"{raw_score_label}: {raw_score}",
            f"Reward: {float(entry.verifier_reward)!r}",
            f"Verifier message: {entry.verifier_message}",
        ]
    )


def _model_summary(entry: LibraryEntry) -> str:
    raw_summary = (entry.metadata or {}).get("raw_model_summary")
    if isinstance(raw_summary, str) and raw_summary.strip():
        return raw_summary.strip()
    if entry.summary.strip():
        return entry.summary.strip()
    return "No previous summary is attached."


def build_guidance_prompt(
    *,
    problem_prompt: str,
    selected_node: LibraryNode,
    selected_entry: LibraryEntry,
    objective_text: str,
    mechanism_constraint: str,
    raw_score_label: str,
) -> Prompt:
    """Build the policy prompt; intentionally omit parent source code."""
    _ = selected_node
    selected_summary = "\n\n".join(
        (
            _model_summary(selected_entry),
            _score_for_prompt(selected_entry, raw_score_label=raw_score_label),
        )
    )
    user = f"""<problem>
{problem_prompt}
</problem>

The next section is the summary-only view of the selected candidate. It is the
only run-local implementation context available to the guidance model.

<selected_summary>
{selected_summary}
</selected_summary>

# Objective
{objective_text}

# Evolutionary Guidelines
1. Use `<selected_summary>` to identify what has already been tried, what worked,
   and which bottleneck the next attempt should address.
2. Stay at the algorithmic-strategy level. Propose high-level algorithmic
   directions or changes to important components. Do not write code, low-level
   implementation details, or parameter schedules.
3. Propose only executable mechanisms.
   {mechanism_constraint}
4. Return exactly one non-empty terminal `<guidance>...</guidance>` block.
   Do not return another custom XML block, commentary, code, or markdown after it.

Required final-answer format:

<guidance>
Provide one conceptual and actionable evolutionary direction for the next execution attempt.
</guidance>
"""
    return Prompt(
        system=(
            "You are the Guidance Model, acting as a strategic navigator for an open-ended scientific "
            "discovery process. Provide evolutionary guidance, not final code or low-level implementation details."
        ),
        user=user,
    )


def build_execution_prompt(
    *,
    problem_prompt: str,
    selected_entry: LibraryEntry,
    guidance: str,
    solution_language: str,
    solution_contract: str,
    score_direction: str,
    raw_score_label: str,
) -> Prompt:
    """Build the executor prompt; unlike the policy prompt, include full parent code."""
    if not selected_entry.solution.strip():
        raise ValueError("Execution prompt requires a selected entry with non-empty solution code")
    fenced_language = "cpp" if solution_language.lower() in {"cpp", "c++", "cxx"} else "python"
    language_name = "C++17" if fenced_language == "cpp" else "Python"
    placeholder = (
        "// complete executable solution required by the problem"
        if fenced_language == "cpp"
        else "# complete executable solution required by the problem"
    )
    parent = (
        f"{_score_for_prompt(selected_entry, raw_score_label=raw_score_label)}\n\n"
        f"<parent_code>\n```{fenced_language}\n{selected_entry.solution.strip()}\n```\n</parent_code>"
    )
    summary_contract = """Write a concise natural-language summary of the complete candidate. Explain:
1. the implemented algorithmic idea;
2. what changed from the parent in response to the guidance and what was preserved;
3. the search, refinement, or optimization mechanisms actually present.

Include enough information for a later model to understand the overall algorithm from the summary alone. If a
guidance component was simplified, approximated, or omitted, say so. Do not include source code, code fences,
copied constants, raw candidate parameters, benchmark profiles, or these output instructions."""
    user = f"""<problem>
{problem_prompt}
</problem>

<selected_parent>
{parent}
</selected_parent>

<guidance>
{guidance.strip()}
</guidance>

Use the problem statement as authoritative. Use the selected parent code as the
runnable baseline and apply the guidance faithfully while preserving unaffected
working behavior. Score direction: {score_direction}.

{solution_contract}

Think carefully before producing the improved program. Your final answer must
contain exactly two top-level XML blocks and end with `</summary>`:

<solution>
```{fenced_language}
{placeholder}
```
</solution>

<summary>
{summary_contract}
</summary>
"""
    return Prompt(
        system=(
            f"You are the execution model. Improve the supplied parent {language_name} candidate by applying "
            "the guidance, then return one complete runnable candidate and its canonical summary."
        ),
        user=user,
    )


def extract_terminal_tag_or_none(text: str, tag: str) -> str | None:
    stripped = text.strip()
    close_tag = f"</{tag}>"
    if not stripped.endswith(close_tag):
        return None
    start = stripped.rfind(f"<{tag}>")
    if start < 0:
        return None
    value = stripped[start + len(tag) + 2 : -len(close_tag)].strip()
    return value or None


def extract_strict_guidance(text: str) -> tuple[str | None, str | None]:
    """Accept one complete, non-empty terminal block; never synthesize or retry."""
    stripped = text.strip()
    open_tag = "<guidance>"
    close_tag = "</guidance>"
    if stripped.count(open_tag) != 1 or stripped.count(close_tag) != 1:
        return None, "expected exactly one complete <guidance> block"
    start = stripped.find(open_tag)
    end = stripped.find(close_tag, start + len(open_tag))
    if end < 0 or stripped[end + len(close_tag) :].strip():
        return None, "the <guidance> block must be terminal"
    guidance = stripped[start + len(open_tag) : end].strip()
    if not guidance:
        return None, "the <guidance> block is empty"
    return guidance, None
