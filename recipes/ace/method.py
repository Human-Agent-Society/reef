"""ACE, Agentic Context Engineering (Zhang et al., arXiv 2510.04618), as a Reef harness method.

ACE keeps a playbook of short lessons, called bullets, in the context of the agent. The served agent is ACE's
Generator: it reads the playbook in ``rules.md``. Reef calls ``ACE.step`` when a batch of traces waits. For each
run in the batch, the Reflector (``roles.reflect``) analyses the run and tags the bullets that applied, and the
Curator (``roles.curate``) turns the analysis into new bullets. Code, not a model, merges the result
(``playbook.Playbook``). After the last run, the step refines the playbook by the counts and renders it into
``rules.md``; this refine rule is Reef's, not upstream's (see the README).

``state/playbook.json`` holds the playbook between steps.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.config import config_option
from reef.core.plugins import PluginSettings
from reef.harness.tree import TreePatch
from reef.train.harness_backend import Method, StepContext

from .playbook import Playbook, normalized, one_line, rules_with
from .roles import curate, reflect

PLAYBOOK_FILE = "playbook.json"
RULES_FILE = "rules.md"
#: The default sections of the playbook.
SECTIONS = ("Strategies and insights", "Code and commands", "Common mistakes to avoid", "Others")


@dataclass(frozen=True)
class ACESettings(PluginSettings):
    """The config of ``ACE``.

    Values: ``model`` is a non-empty name that ``ACE.step`` looks up in ``Services.models``: ``served`` or an
    ``evolution.models`` entry. ``sections`` holds one or more names; each name is non-empty and equal to its
    ``one_line`` form, and no two names are equal by ``normalized``. ``max_bullets`` ≥ 1 and ``prune_margin`` ≥ 1.
    ``max_traces_per_step`` ≥ 0, where 0 sets no limit.
    Purpose: ``ACE.from_config`` parses ``evolution.method.config`` into it; ``ACE.step`` reads it.
    """

    model: str = config_option("served", help="The model binding of the Reflector and the Curator.")
    sections: tuple[str, ...] = config_option(SECTIONS, help="The playbook's sections; each new bullet goes in one.")
    max_bullets: int = config_option(100, help="The most bullets that the playbook keeps.")
    prune_margin: int = config_option(
        2, help="Remove a bullet when its harmful count exceeds its helpful count by this many or more."
    )
    max_traces_per_step: int = config_option(
        0, help="The most runs (traces with a score or a reply) that one step reads. 0 sets no limit."
    )

    def __post_init__(self) -> None:
        sections = tuple(one_line(section) for section in self.sections)
        if not self.model.strip():
            raise ValueError("model must name a model binding: served or an evolution.models entry")
        if not sections or not all(sections) or len({normalized(section) for section in sections}) < len(sections):
            raise ValueError("sections must be one or more names, each non-empty and different from the others")
        if self.max_bullets < 1 or self.prune_margin < 1:
            raise ValueError("max_bullets and prune_margin must be at least 1")
        if self.max_traces_per_step < 0:
            raise ValueError("max_traces_per_step must be at least 0 (0 sets no limit)")
        # Each name becomes one heading line of rules.md.
        object.__setattr__(self, "sections", sections)


class ACE(Method):
    """ACE's Reflector, Curator and merge as a harness method; the served agent is ACE's Generator.

    Values: ``settings`` is the parsed config.
    Purpose: a deployment names the class in ``evolution.method``, and the backend calls ``step`` once per step.
    The step's notes count what happened:

    ``traces``, ``runs``, ``read``
        The traces of the step's material, the runs among them (traces with a score or a reply), and the runs
        that the Reflector read.
    ``skipped``
        Each read run whose Reflector or Curator reply was not the asked JSON, as ``{trace, reason}``. A run whose
        Curator reply failed keeps the tags of its reflection.
    ``helpful_tags``, ``harmful_tags``
        The tags that the reflections added to the counts.
    ``added``, ``duplicates``, ``refused``, ``ignored``
        The Curator's operations: new bullets, texts that a bullet has, texts that a screen of
        ``Playbook.merged`` turned down, and operations that are not an ``ADD`` of text to a configured section.
    ``pruned``, ``capped``, ``bullets``
        The bullets removed for their harmful count, the bullets removed by the cap, and the bullets kept.
    """

    def __init__(self, settings: ACESettings) -> None:
        self.settings = settings

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        return cls(ACESettings.parse(config, plugin_class=cls))

    def step(self, context: StepContext) -> None:
        workspace, services, settings = context.workspace, context.services, self.settings
        model = services.models[settings.model]
        playbook_path = workspace.state / PLAYBOOK_FILE
        playbook = (
            Playbook.from_json(playbook_path.read_text(encoding="utf-8")) if playbook_path.is_file() else Playbook()
        )

        # 1. The runs. A model call that ends on a tool call is the middle of a run: it has no reply, and the call
        #    that ends with the reply carries the whole run.
        traces = workspace.traces()
        runs = [trace for trace in traces if trace.score is not None or trace.reply]
        read = runs[: settings.max_traces_per_step or len(runs)]

        # 2. Per run, in order: the Reflector tags bullets, the Curator asks for new ones, and code merges them.
        tags: Counter[str] = Counter()
        outcomes: Counter[str] = Counter()
        skipped: list[dict[str, str]] = []
        for trace in read:
            reflection = reflect(model, playbook, settings.sections, trace)
            if reflection is None:
                skipped.append({"trace": trace.record, "reason": "the Reflector's reply is not the asked JSON"})
                continue
            playbook = playbook.tagged(reflection.tags)
            tags.update(reflection.tags.values())
            curation = curate(model, playbook, settings.sections, settings.max_bullets, reflection, trace)
            if curation is None:
                skipped.append({"trace": trace.record, "reason": "the Curator's reply is not the asked JSON"})
                continue
            playbook, merge_outcomes = playbook.merged(curation.additions, workspace.step)
            outcomes.update(merge_outcomes)
            outcomes["ignored"] += curation.ignored

        # 3. Refine once per step: remove the harmful bullets, then the lowest-scored past the cap.
        pruned = playbook.pruned(settings.prune_margin)
        kept = pruned.capped(settings.max_bullets)
        notes: dict[str, int | list[dict[str, str]]] = {
            "traces": len(traces),
            "runs": len(runs),
            "read": len(read),
            "skipped": skipped,
            "helpful_tags": tags["helpful"],
            "harmful_tags": tags["harmful"],
            **{outcome: outcomes[outcome] for outcome in ("added", "duplicates", "refused", "ignored")},
            "pruned": len(playbook.bullets) - len(pruned.bullets),
            "capped": len(pruned.bullets) - len(kept.bullets),
            "bullets": len(kept.bullets),
        }

        # 4. Render the playbook into rules.md. The state and the tree change together or not at all, so they agree.
        tree = workspace.tree()
        rules = rules_with(tree.files.get(RULES_FILE, ""), kept.text(settings.sections))
        candidate = tree.apply(TreePatch({RULES_FILE: rules or None}))
        validation = services.check(candidate)
        if not validation.passed:
            workspace.write_output(
                f"kept the last playbook: the new rules fail validation: {validation.message()}", notes
            )
            return
        playbook_path.write_text(kept.to_json(), encoding="utf-8")
        workspace.write_tree(candidate)
        workspace.write_output(
            f"traces: {len(read)} of {len(traces)} read, {len(skipped)} skipped; tags: {tags['helpful']} helpful, "
            f"{tags['harmful']} harmful; bullets: {outcomes['added']} added, "
            f"{len(playbook.bullets) - len(kept.bullets)} removed, {len(kept.bullets)} kept",
            notes,
        )


__all__ = ["ACE", "ACESettings"]
