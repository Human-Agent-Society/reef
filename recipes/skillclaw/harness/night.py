"""The skill pool on disk and the official evolve cycle.

The pool is skills/<name>/SKILL.md files; version history lives in a
registry outside the pool the agent sees. night_step runs their order:
summarize, judge, group, one decision per skill group, the no skill
bucket last, with every non-skip decision selected.
The frozen control run returns before any LLM call.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from . import evolver
from .sessions import EMPTY_GROUP, group_sessions

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_HISTORY_CAP = 20


def read_pool(root: Path) -> dict[str, str]:
    return {
        skill_md.parent.name: skill_md.read_text(encoding="utf-8")
        for skill_md in sorted(root.glob("skills/*/SKILL.md"))
    }


def parse_skill(name: str, text: str) -> dict[str, Any]:
    """Their parse: yaml frontmatter first, regex fallback, extras kept."""
    skill: dict[str, Any] = {
        "name": name,
        "description": "",
        "category": "general",
        "content": "",
        "extra_frontmatter": {},
    }
    if not text.startswith("---"):
        skill["content"] = text
        return skill
    end = text.find("\n---", 3)
    if end == -1:
        skill["content"] = text
        return skill
    front_text = text[3:end].strip()
    try:
        front = yaml.safe_load(front_text) or {}
    except Exception:
        front = {}
        for key in ("name", "description", "category"):
            match = re.search(rf'^{key}:\s*["\']?(.*?)["\']?\s*$', front_text, re.MULTILINE)
            if match:
                front[key] = match.group(1)
    if isinstance(front, dict):
        skill["description"] = str(front.get("description", ""))
        skill["category"] = str(front.get("category", "general"))
        extra = {key: value for key, value in front.items() if key not in ("name", "description", "category")}
        if extra:
            skill["extra_frontmatter"] = extra
    skill["content"] = text[end + 4 :].strip()
    return skill


def build_skill_md(skill: dict[str, Any]) -> str:
    """Their frontmatter assembly: quoting rule and extra keys included.

    Field values are used raw, as theirs are: a null value raises, and the
    per group handler drops that action instead of corrupting the pool.
    """
    description = skill.get("description", "")
    if any(c in description for c in ":{}[],\"'#&*!|>%@`\n"):
        escaped = description.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        description = f'"{escaped}"'
    lines = [
        f"name: {skill.get('name', 'unknown')}",
        f"description: {description}",
        f"category: {skill.get('category', 'general')}",
    ]
    extra = skill.get("extra_frontmatter")
    if isinstance(extra, dict) and extra:
        for key, value in extra.items():
            if key not in ("name", "description", "category"):
                lines.append(f"{key}: {yaml.dump(value, default_flow_style=True).strip()}")
    return "---\n" + "\n".join(lines) + "\n---\n\n" + skill.get("content", "") + "\n"


def sanitize_name(raw: str) -> str:
    """Their name rule: a lowercase slug, or every other character to a dash."""
    name = raw.strip().lower()
    if _SLUG.match(name):
        return name
    return re.sub(r"[^a-z0-9_-]", "-", name).strip("-") or "unnamed-skill"


class Registry:
    """Their skill registry: versions and content shas outside the pool."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._map: dict[str, dict[str, Any]] = {}
        if path.exists():
            self._map = json.loads(path.read_text(encoding="utf-8"))

    def content_sha(self, name: str) -> str:
        entry = self._map.get(name)
        return entry.get("content_sha", "") if entry else ""

    def version(self, name: str) -> int:
        entry = self._map.get(name)
        return entry.get("version", 0) if entry else 0

    def record_update(self, name: str, content_sha: str, action: str) -> int:
        entry = self._map.setdefault(
            name,
            {"skill_id": hashlib.sha256(name.encode()).hexdigest()[:12], "version": 0, "history": []},
        )
        entry["version"] = entry.get("version", 0) + 1
        entry["content_sha"] = content_sha
        history = entry.setdefault("history", [])
        history.append(
            {
                "version": entry["version"],
                "content_sha": content_sha,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "action": action,
            }
        )
        if len(history) > _HISTORY_CAP:
            entry["history"] = history[-_HISTORY_CAP:]
        return entry["version"]

    def save(self) -> None:
        self._path.write_text(json.dumps(self._map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def materialize(pool: Path, registry: Registry, skill: dict[str, Any], action: str, llm: evolver.ChatFn) -> str:
    """Their resolve and upload: conflict against the registry sha, merge, write."""
    name = skill["name"]
    rendered = build_skill_md(skill)
    existing_sha = registry.content_sha(name)
    incoming_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    skill_md = pool / "skills" / name / "SKILL.md"
    if existing_sha and existing_sha != incoming_sha and skill_md.exists():
        existing = parse_skill(name, skill_md.read_text(encoding="utf-8"))
        existing["_version"] = registry.version(name)
        merged = evolver.merge(existing, skill, llm)
        if merged and merged.get("name"):
            merged["name"] = name
            rendered = build_skill_md(merged)
            action = "merge"
    skill_md.parent.mkdir(parents=True, exist_ok=True)
    skill_md.write_text(rendered, encoding="utf-8")
    registry.record_update(name, hashlib.sha256(rendered.encode("utf-8")).hexdigest(), action)
    return action


def _apply(
    target: Path,
    *,
    registry: Registry,
    decision: dict[str, Any],
    current: dict[str, Any] | None,
    llm: Any,
    group: str,
) -> dict[str, Any]:
    if decision["action"] == "skip":
        return {"group": group, "decision": decision}
    skill = dict(decision["skill"])
    # Their inherit: optimize keeps the body; gaps fill from the incumbent.
    if decision["action"] == "optimize_description" and current:
        skill["content"] = current.get("content", "")
        skill["category"] = current.get("category", "general")
    if current:
        skill.setdefault("content", current.get("content", ""))
        skill.setdefault("category", current.get("category", "general"))
        skill.setdefault("extra_frontmatter", current.get("extra_frontmatter") or {})
    if decision["action"] == "improve_skill" and current and current.get("name"):
        skill["name"] = current["name"]
    elif skill.get("name"):
        skill["name"] = sanitize_name(str(skill["name"]))
    else:
        # Their materialize drops a nameless skill instead of minting one.
        return {"group": group, "decision": decision, "dropped": "no name"}
    landed = materialize(target, registry, skill, decision["action"], llm)
    return {"group": group, "decision": {**decision, "action": landed, "skill_name": skill["name"]}, "applied": True}


def night_step(
    *,
    sessions: list[dict[str, Any]],
    incumbent: Path,
    work_dir: Path,
    frozen: bool = False,
    llm: evolver.ChatFn,
) -> dict[str, Any]:
    if frozen:
        return {"pool": incumbent, "advanced": False, "audit": [{"frozen": True}]}
    for session in sessions:
        evolver.attach_metadata(session)
    # Their summarize and judge stages gather every call concurrently.
    with ThreadPoolExecutor(max_workers=max(1, len(sessions))) as workers:
        for session, future in [(s, workers.submit(evolver.summarize, s, llm)) for s in sessions]:
            session["_summary"] = future.result()
    judged = [session for session in sessions if evolver.judge_needed(session)]
    if judged:
        with ThreadPoolExecutor(max_workers=len(judged)) as workers:
            for future in [workers.submit(evolver.judge, s, llm) for s in judged]:
                future.result()

    # Their name order sorts the full SKILL.md paths, not the bare names.
    existing_names = sorted(read_pool(incumbent), key=lambda name: f"{name}/SKILL.md")
    groups = group_sessions(sessions)
    no_skill = groups.pop(EMPTY_GROUP, [])

    target = work_dir / "pool"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(incumbent, target)
    # Their registry starts empty each night, so the merge path fires only
    # on a same night double upload of one name.
    registry = Registry(work_dir / "skill_registry.json")

    audit: list[dict[str, Any]] = []
    applied = 0
    for name, group in groups.items():
        try:
            current_md = target / "skills" / name / "SKILL.md"
            current = parse_skill(name, current_md.read_text(encoding="utf-8")) if current_md.exists() else None
            decision = evolver.decide(
                skill_name=name, current=current, sessions=group, existing_names=existing_names, llm=llm
            )
            entry = _apply(target, registry=registry, decision=decision, current=current, llm=llm, group=name)
            audit.append(entry)
            applied += int(entry.get("applied", False))
        except Exception as exc:  # noqa: PERF203 - their per group fault isolation
            audit.append({"group": name, "error": str(exc)[:300]})
    if no_skill:
        # Their server catches a failed cycle one level up and retries later;
        # with one night per day, the isolation moves onto the bucket itself
        # so a failed create loses only the bucket, like a failed group.
        try:
            decision = evolver.create(sessions=no_skill, existing_names=existing_names, llm=llm)
            if decision["action"] != "skip":
                # Their no skill handler always lands with the create action.
                decision = {**decision, "action": "create_skill"}
            entry = _apply(target, registry=registry, decision=decision, current=None, llm=llm, group="(no skill)")
        except Exception as exc:
            entry = {"group": "(no skill)", "error": str(exc)[:300]}
        audit.append(entry)
        applied += int(entry.get("applied", False))
    registry.save()
    return {"pool": target if applied else incumbent, "advanced": applied > 0, "audit": audit}
