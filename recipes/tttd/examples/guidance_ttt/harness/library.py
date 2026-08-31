from __future__ import annotations

import hashlib
import json
import math
import threading
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Any
from uuid import uuid4

from .puct import PUCT_Q_BLEND, normalize_puct_q_mode, rank_archive_nodes
from .state import LibraryEntry, LibraryNode


ARCHIVE_CANDIDATE_KEY = "archive_candidate"


class GuidanceLibrary:
    """JSON-backed PUCT library for guidance + execution TTT."""

    def __init__(
        self,
        path: str | Path,
        *,
        initial_nodes: list[LibraryNode] | None = None,
        rollout_n: int = 1,
        puct_c: float = 1.0,
        puct_q_mode: str = PUCT_Q_BLEND,
        max_buffer_size: int = 1000,
        topk_children: int = 2,
        discover_compat: bool = False,
        groups_per_batch: int = 1,
        score_direction: str = "max",
    ) -> None:
        self.path = Path(path)
        self.rollout_n = int(rollout_n)
        self.puct_c = float(puct_c)
        self.puct_q_mode = normalize_puct_q_mode(puct_q_mode)
        self.max_buffer_size = int(max_buffer_size)
        self.topk_children = int(topk_children)
        self.discover_compat = bool(discover_compat)
        self.groups_per_batch = int(groups_per_batch)
        self.score_direction = self._normalize_score_direction(score_direction)
        self._thread_lock = threading.RLock()
        self._nodes: dict[str, LibraryNode] = {}
        self._entries: dict[str, LibraryEntry] = {}
        self._groups: dict[str, dict[str, Any]] = {}
        self._best_node_id: str | None = None
        self._puct_n: dict[str, int] = {}
        self._puct_m: dict[str, float] = {}
        self._puct_T: int = 0

        with self._thread_lock, self._file_lock():
            if self.path.exists():
                self._load()
            else:
                for node in initial_nodes or []:
                    if self.discover_compat:
                        node.metadata[ARCHIVE_CANDIDATE_KEY] = True
                    self._nodes[node.id] = node
                self._normalize_node_values()
                self._refresh_best()
                self._save()

    @contextmanager
    def _file_lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("w") as lock_file:
            flock(lock_file, LOCK_EX)
            try:
                yield
            finally:
                flock(lock_file, LOCK_UN)

    def snapshot(self) -> dict[str, Any]:
        with self._thread_lock, self._file_lock():
            self._reload()
            return self._to_store()

    def configure_pristine_archive(
        self,
        *,
        rollout_n: int,
        puct_c: float,
        puct_q_mode: str = PUCT_Q_BLEND,
        max_buffer_size: int,
        topk_children: int,
        discover_compat: bool = False,
        groups_per_batch: int = 1,
        score_direction: str = "max",
    ) -> None:
        """Apply run-specific sampling config to an unused seed archive."""
        expected = self._coerce_runtime_config(
            rollout_n=rollout_n,
            puct_c=puct_c,
            puct_q_mode=puct_q_mode,
            max_buffer_size=max_buffer_size,
            topk_children=topk_children,
            discover_compat=discover_compat,
            groups_per_batch=groups_per_batch,
            score_direction=score_direction,
        )
        with self._thread_lock, self._file_lock():
            self._reload()
            if self._groups or self._puct_T or self._puct_n or self._puct_m:
                raise ValueError(
                    f"Seed library {self.path} is not pristine: run-local groups or PUCT statistics "
                    "are already present. Use a clean bootstrap seed or a new output directory."
                )
            self.rollout_n = expected["rollout_n"]
            self.puct_c = expected["puct_c"]
            self.puct_q_mode = expected["puct_q_mode"]
            self.max_buffer_size = expected["max_buffer_size"]
            self.topk_children = expected["topk_children"]
            self.discover_compat = expected["discover_compat"]
            self.groups_per_batch = expected["groups_per_batch"]
            self.score_direction = expected["score_direction"]
            self._normalize_node_values()
            self._save()

    def assert_runtime_config(
        self,
        *,
        rollout_n: int,
        puct_c: float,
        puct_q_mode: str = PUCT_Q_BLEND,
        max_buffer_size: int,
        topk_children: int,
        discover_compat: bool = False,
        groups_per_batch: int = 1,
        score_direction: str = "max",
    ) -> None:
        """Fail fast when a persisted archive disagrees with the active recipe."""
        expected = self._coerce_runtime_config(
            rollout_n=rollout_n,
            puct_c=puct_c,
            puct_q_mode=puct_q_mode,
            max_buffer_size=max_buffer_size,
            topk_children=topk_children,
            discover_compat=discover_compat,
            groups_per_batch=groups_per_batch,
            score_direction=score_direction,
        )
        actual = self._runtime_config()
        mismatches = []
        for key, expected_value in expected.items():
            actual_value = actual[key]
            matches = (
                math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1e-12)
                if key == "puct_c"
                else actual_value == expected_value
            )
            if not matches:
                mismatches.append(f"{key}: archive={actual_value!r}, recipe={expected_value!r}")
        if mismatches:
            raise ValueError(
                f"Guidance library runtime config mismatch for {self.path}: "
                + "; ".join(mismatches)
                + ". Existing run libraries are immutable with respect to sampling config; "
                "use a new output directory."
            )
        if self.discover_compat:
            root_count = sum(1 for node in self._nodes.values() if node.parent_id is None)
            if root_count != self.groups_per_batch:
                raise ValueError(
                    f"Discover-compatible library {self.path} has {root_count} root states, "
                    f"expected groups_per_batch={self.groups_per_batch}. Start from a fresh output directory."
                )
        self._assert_group_accounting()

    @staticmethod
    def _coerce_runtime_config(
        *,
        rollout_n: int,
        puct_c: float,
        puct_q_mode: str = PUCT_Q_BLEND,
        max_buffer_size: int,
        topk_children: int,
        discover_compat: bool = False,
        groups_per_batch: int = 1,
        score_direction: str = "max",
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "rollout_n": int(rollout_n),
            "puct_c": float(puct_c),
            "puct_q_mode": normalize_puct_q_mode(puct_q_mode),
            "max_buffer_size": int(max_buffer_size),
            "topk_children": int(topk_children),
            "discover_compat": bool(discover_compat),
            "groups_per_batch": int(groups_per_batch),
            "score_direction": GuidanceLibrary._normalize_score_direction(score_direction),
        }
        if config["rollout_n"] <= 0:
            raise ValueError(f"rollout_n must be positive, got {config['rollout_n']!r}")
        if config["groups_per_batch"] <= 0:
            raise ValueError(f"groups_per_batch must be positive, got {config['groups_per_batch']!r}")
        if config["discover_compat"] and config["puct_q_mode"] != "best_child":
            raise ValueError("Discover-compatible sampling requires puct_q_mode='best_child'")
        return config

    def _runtime_config(self) -> dict[str, Any]:
        return {
            "rollout_n": self.rollout_n,
            "puct_c": self.puct_c,
            "puct_q_mode": self.puct_q_mode,
            "max_buffer_size": self.max_buffer_size,
            "topk_children": self.topk_children,
            "discover_compat": self.discover_compat,
            "groups_per_batch": self.groups_per_batch,
            "score_direction": self.score_direction,
        }

    @staticmethod
    def _normalize_score_direction(score_direction: str) -> str:
        normalized = str(score_direction).strip().lower()
        if normalized not in {"min", "max"}:
            raise ValueError(f"score_direction must be 'min' or 'max', got {score_direction!r}")
        return normalized

    def _assert_group_accounting(self) -> None:
        errors: list[str] = []
        finalized_count = 0
        submitted_count = 0
        for group_uid, group in self._groups.items():
            submitted = int(group.get("submitted", 0))
            submitted_count += submitted
            finalized = bool(group.get("finalized", False))
            if submitted < 0 or submitted > self.rollout_n:
                errors.append(f"group {group_uid!r} has submitted={submitted}, expected 0..{self.rollout_n}")
            if finalized:
                finalized_count += 1
                if submitted != self.rollout_n:
                    errors.append(
                        f"group {group_uid!r} is finalized with submitted={submitted}, expected {self.rollout_n}"
                    )
            elif submitted >= self.rollout_n:
                errors.append(
                    f"group {group_uid!r} is not finalized with submitted={submitted}, "
                    f"expected less than {self.rollout_n}"
                )
        expected_puct_updates = submitted_count if self.discover_compat else finalized_count
        if self._puct_T != expected_puct_updates:
            unit = "submitted rollout" if self.discover_compat else "finalized group"
            errors.append(f"puct_T={self._puct_T}, expected one update for each of {expected_puct_updates} {unit}s")
        if errors:
            raise ValueError(
                f"Guidance library group accounting is inconsistent for {self.path}: " + "; ".join(errors)
            )

    def acquire_group(
        self,
        group_uid: str,
        *,
        visible_timestep_exclusive: int | None = None,
        require_solution: bool = False,
    ) -> LibraryNode:
        with self._thread_lock, self._file_lock():
            self._reload()
            group = self._groups.get(group_uid)
            if group is not None:
                selected = self._nodes[group["selected_node_id"]]
                if require_solution and not self._node_has_solution(selected):
                    raise RuntimeError(
                        f"Existing group {group_uid!r} is attached to node {selected.id!r}, which has no "
                        "solution code. Start a new run from a code-bearing bootstrap library."
                    )
                return selected
            selected = self._select_node(
                visible_timestep_exclusive=visible_timestep_exclusive,
                blocked_node_ids=self._same_step_blocked_node_ids(visible_timestep_exclusive),
                require_solution=require_solution,
            )
            selected.visits += 1
            self._groups[group_uid] = {
                "selected_node_id": selected.id,
                "submitted": 0,
                "children": [],
                "finalized": False,
                "visible_timestep_exclusive": visible_timestep_exclusive,
            }
            self._save()
            return selected

    def ensure_pristine_root_count(self, count: int) -> None:
        """Replicate seed roots so each same-batch group starts from an independent lineage."""
        target_count = int(count)
        if target_count <= 0:
            raise ValueError(f"root count must be positive, got {target_count!r}")
        with self._thread_lock, self._file_lock():
            self._reload()
            if self._groups or self._puct_T or self._puct_n or self._puct_m:
                raise ValueError(
                    f"Cannot change root count for non-pristine library {self.path}; use a fresh output directory."
                )
            roots = [node for node in self._nodes.values() if node.parent_id is None]
            if not roots:
                raise ValueError(f"Cannot replicate roots for empty library {self.path}")
            if len(roots) > target_count:
                raise ValueError(
                    f"Library {self.path} already has {len(roots)} roots, exceeding requested {target_count}"
                )
            source_roots = list(roots)
            source_index = 0
            while len(roots) < target_count:
                source = source_roots[source_index % len(source_roots)]
                source_index += 1
                clone = LibraryNode.from_dict(source.to_dict())
                clone.id = str(uuid4())
                clone.parent_id = None
                clone.children = []
                clone.visits = 0
                clone.metadata = dict(clone.metadata)
                clone.metadata[ARCHIVE_CANDIDATE_KEY] = True
                clone.metadata["replicated_seed_root"] = True
                if source.entry_id:
                    source_entry = self._entries.get(source.entry_id)
                    if source_entry is None:
                        raise ValueError(f"Root {source.id!r} references missing seed entry {source.entry_id!r}")
                    cloned_entry = LibraryEntry.from_dict(source_entry.to_dict())
                    cloned_entry.id = str(uuid4())
                    cloned_entry.parent_id = clone.id
                    cloned_entry.metadata = dict(cloned_entry.metadata)
                    cloned_entry.metadata["replicated_seed_entry"] = True
                    self._entries[cloned_entry.id] = cloned_entry
                    clone.entry_id = cloned_entry.id
                self._nodes[clone.id] = clone
                roots.append(clone)
            self._normalize_node_values()
            self._refresh_best()
            self._save()

    def submit_child(self, group_uid: str, entry: LibraryEntry) -> LibraryNode | None:
        with self._thread_lock, self._file_lock():
            self._reload()
            if group_uid not in self._groups:
                raise KeyError(f"Unknown group_uid: {group_uid}")
            group = self._groups[group_uid]
            submitted = int(group.get("submitted", 0))
            if bool(group.get("finalized", False)) or submitted >= self.rollout_n:
                raise RuntimeError(
                    f"Group {group_uid!r} is already complete: submitted={submitted}, "
                    f"rollout_n={self.rollout_n}. Refusing an extra child submission."
                )
            parent_id = entry.parent_id if entry.parent_id in self._nodes else group["selected_node_id"]
            parent = self._nodes[parent_id]

            self._entries[entry.id] = entry
            child: LibraryNode | None = None
            should_archive = not self.discover_compat or (
                entry.verifier_status == "valid" and entry.verifier_raw_score is not None
            )
            puct_child_value = self._entry_puct_value(entry) if should_archive else None
            if should_archive and self.discover_compat and self._entry_duplicates_archive_candidate(entry):
                should_archive = False
            if should_archive:
                child = LibraryNode(
                    id=str(uuid4()),
                    problem_id=entry.problem_id,
                    timestep=entry.timestep,
                    entry_id=entry.id,
                    value=self._entry_puct_value(entry),
                    raw_score=entry.verifier_raw_score,
                    visits=0,
                    parent_id=parent.id,
                    children=[],
                    metadata={
                        "verifier_status": entry.verifier_status,
                        ARCHIVE_CANDIDATE_KEY: True,
                    },
                )
                self._nodes[child.id] = child
                parent.children.append(child.id)
            group["submitted"] = submitted + 1
            group.setdefault("entry_ids", []).append(entry.id)
            if child is not None:
                group["children"].append(child.id)
            if self.discover_compat:
                self._record_puct_rollout(
                    parent.id,
                    child_value=puct_child_value,
                )
            if group["submitted"] == self.rollout_n:
                group["finalized"] = True
                if self.discover_compat:
                    if self._batch_is_complete(group.get("visible_timestep_exclusive")):
                        self._filter_archive()
                else:
                    self._update_puct_stats_for_group(group)
                    self._filter_archive()
                self._refresh_best()
            else:
                self._refresh_best()
            self._save()
            return child

    def rollback_incomplete_steps_after(self, max_timestep: int) -> dict[str, int]:
        """Discard an in-flight post-checkpoint step without changing completed history.

        A trainer checkpoint is written only after the corresponding rollout batch has
        completed. If the process exits during the next batch, the JSON library can be
        slightly ahead of the actor/optimizer checkpoint. Discover-compatible runs
        update PUCT accounting per submitted rollout, so restoring the checkpoint also
        needs to undo those in-flight updates.

        Fully completed post-checkpoint batches are rejected because archive filtering
        may already have pruned older nodes; rolling those back would require a library
        snapshot from the checkpoint boundary.
        """
        checkpoint_timestep = int(max_timestep)
        with self._thread_lock, self._file_lock():
            self._reload()
            future_groups = {
                group_uid: group
                for group_uid, group in self._groups.items()
                if int(group.get("visible_timestep_exclusive") or group_uid.split(":", 1)[0]) > checkpoint_timestep
            }
            if not future_groups:
                return {"groups": 0, "entries": 0, "nodes": 0, "submitted": 0}
            if not self.discover_compat:
                raise RuntimeError(
                    "Post-checkpoint library rollback is currently supported only for discover-compatible runs."
                )

            future_steps = {
                int(group.get("visible_timestep_exclusive") or group_uid.split(":", 1)[0])
                for group_uid, group in future_groups.items()
            }
            for timestep in future_steps:
                step_groups = [
                    group
                    for group in self._groups.values()
                    if int(group.get("visible_timestep_exclusive") or -1) == timestep
                ]
                if len(step_groups) >= self.groups_per_batch and all(
                    bool(group.get("finalized", False)) for group in step_groups
                ):
                    raise RuntimeError(
                        f"Library timestep {timestep} is fully finalized but the latest trainer "
                        f"checkpoint is only timestep {checkpoint_timestep}; refusing a lossy rollback."
                    )

            removed_group_uids = set(future_groups)
            removed_entry_ids = {
                str(entry_id) for group in future_groups.values() for entry_id in group.get("entry_ids", [])
            }
            removed_entry_ids.update(
                entry_id
                for entry_id, entry in self._entries.items()
                if entry.timestep > checkpoint_timestep
                or str((entry.metadata or {}).get("group_uid", "")) in removed_group_uids
            )
            submitted_count = sum(int(group.get("submitted", 0)) for group in future_groups.values())
            affected_parent_ids: set[str] = set()

            for group_uid, group in future_groups.items():
                selected_node_id = str(group["selected_node_id"])
                selected = self._nodes.get(selected_node_id)
                if selected is None:
                    raise RuntimeError(
                        f"Cannot roll back in-flight group {group_uid!r}: selected node "
                        f"{selected_node_id!r} is no longer present."
                    )
                selected.visits -= 1
                if selected.visits < 0:
                    raise RuntimeError(
                        f"Cannot roll back in-flight group {group_uid!r}: selected node visits would become negative."
                    )
                affected_parent_ids.add(selected_node_id)
                submitted = int(group.get("submitted", 0))
                for ancestor_id in self._ancestor_node_ids(selected_node_id):
                    remaining = int(self._puct_n.get(ancestor_id, 0)) - submitted
                    if remaining < 0:
                        raise RuntimeError(
                            f"Cannot roll back in-flight group {group_uid!r}: PUCT count for "
                            f"{ancestor_id!r} would become negative."
                        )
                    if remaining:
                        self._puct_n[ancestor_id] = remaining
                    else:
                        self._puct_n.pop(ancestor_id, None)

            self._puct_T -= submitted_count
            if self._puct_T < 0:
                raise RuntimeError("Cannot roll back in-flight groups: total PUCT count would become negative.")

            self._groups = {
                group_uid: group for group_uid, group in self._groups.items() if group_uid not in removed_group_uids
            }
            self._entries = {
                entry_id: entry for entry_id, entry in self._entries.items() if entry_id not in removed_entry_ids
            }
            removed_node_ids = {
                node_id
                for node_id, node in self._nodes.items()
                if node.timestep > checkpoint_timestep or node.entry_id in removed_entry_ids
            }
            self._nodes = {node_id: node for node_id, node in self._nodes.items() if node_id not in removed_node_ids}
            for node in self._nodes.values():
                node.children = [child_id for child_id in node.children if child_id in self._nodes]
            self._puct_n = {node_id: count for node_id, count in self._puct_n.items() if node_id in self._nodes}
            self._puct_m = {node_id: value for node_id, value in self._puct_m.items() if node_id in self._nodes}

            retained_submitted_entry_ids = {
                str(entry_id) for group in self._groups.values() for entry_id in group.get("entry_ids", [])
            }
            for parent_id in affected_parent_ids:
                retained_values = [
                    self._entry_puct_value(entry)
                    for entry_id, entry in self._entries.items()
                    if entry_id in retained_submitted_entry_ids
                    and entry.parent_id == parent_id
                    and entry.verifier_status == "valid"
                    and entry.verifier_raw_score is not None
                ]
                if retained_values:
                    self._puct_m[parent_id] = max(retained_values)
                else:
                    self._puct_m.pop(parent_id, None)

            self._refresh_best()
            self._assert_group_accounting()
            self._save()
            return {
                "groups": len(future_groups),
                "entries": len(removed_entry_ids),
                "nodes": len(removed_node_ids),
                "submitted": submitted_count,
            }

    def attach_entry_to_root(
        self, root_node_id: str, entry: LibraryEntry, *, overwrite_existing: bool = False
    ) -> None:
        with self._thread_lock, self._file_lock():
            self._reload()
            if root_node_id not in self._nodes:
                raise KeyError(f"Unknown root node id: {root_node_id}")
            root = self._nodes[root_node_id]
            if root.parent_id is not None:
                raise ValueError(f"Node {root_node_id} is not a root node")
            if root.entry_id and not overwrite_existing:
                raise ValueError(f"Root node {root_node_id} already has entry {root.entry_id}")

            entry.parent_id = root.id
            entry.timestep = 0
            self._entries[entry.id] = entry
            root.entry_id = entry.id
            root.value = self._entry_puct_value(entry)
            root.raw_score = entry.verifier_raw_score
            root.metadata.update(
                {
                    "bootstrap": bool((entry.metadata or {}).get("bootstrap")),
                    "verifier_status": entry.verifier_status,
                    ARCHIVE_CANDIDATE_KEY: True,
                }
            )
            self._refresh_best()
            self._save()

    def mark_node_visited(self, node_id: str, *, count: int) -> None:
        with self._thread_lock, self._file_lock():
            self._reload()
            self._nodes[node_id].visits = int(count)
            self._puct_n[node_id] = int(count)
            self._save()

    def get_entry(self, entry_id: str | None) -> LibraryEntry | None:
        if entry_id is None:
            return None
        with self._thread_lock, self._file_lock():
            self._reload()
            return self._entries.get(entry_id)

    def context_for_node(
        self,
        node: LibraryNode,
        *,
        visible_timestep_exclusive: int | None = None,
    ) -> dict[str, Any]:
        with self._thread_lock, self._file_lock():
            self._reload()
            selected_node = self._nodes.get(node.id)
            lineage: list[LibraryEntry] = []
            current: LibraryNode | None = selected_node
            while current is not None:
                if (
                    current.entry_id
                    and current.entry_id in self._entries
                    and self._entry_is_visible(
                        self._entries[current.entry_id],
                        visible_timestep_exclusive=visible_timestep_exclusive,
                    )
                ):
                    lineage.append(self._entries[current.entry_id])
                current = self._nodes.get(current.parent_id) if current.parent_id else None
            lineage.reverse()
            best = self._best_visible_node(visible_timestep_exclusive=visible_timestep_exclusive)
            best_entry = self._entries.get(best.entry_id) if best and best.entry_id else None
            failures = [
                entry
                for entry in self._entries.values()
                if entry.parent_id == node.id
                and entry.verifier_status != "valid"
                and self._entry_is_visible(entry, visible_timestep_exclusive=visible_timestep_exclusive)
            ]
            return {
                "selected_entry": self._visible_entry(
                    selected_node.entry_id if selected_node is not None else None,
                    visible_timestep_exclusive=visible_timestep_exclusive,
                ),
                "lineage_entries": lineage,
                "global_best_entries": [best_entry] if best_entry else [],
                "local_failure_entries": failures,
            }

    def _select_node(
        self,
        *,
        visible_timestep_exclusive: int | None = None,
        blocked_node_ids: set[str] | None = None,
        require_solution: bool = False,
    ) -> LibraryNode:
        visible_nodes = [
            node
            for node in self._nodes.values()
            if self._node_is_visible(node, visible_timestep_exclusive=visible_timestep_exclusive)
            and self._node_is_archive_candidate(node)
            and (not require_solution or self._node_has_solution(node))
        ]
        if not visible_nodes:
            if require_solution:
                raise RuntimeError(
                    "GuidanceLibrary has no visible node with solution code. "
                    "Initialize the run from a valid code-bearing bootstrap library."
                )
            raise ValueError("GuidanceLibrary requires at least one root node")
        initial_ids = {node.id for node in visible_nodes if node.parent_id is None}
        ranked = rank_archive_nodes(
            visible_nodes,
            initial_ids=initial_ids,
            visit_counts=self._puct_n,
            best_reachable_values=self._puct_m,
            total_visits=self._puct_T,
            puct_c=self.puct_c,
            q_mode=self.puct_q_mode,
        )
        blocked_node_ids = blocked_node_ids or set()
        for _score, _value, node, _n, _q, _prior, _bonus in ranked:
            if node.id not in blocked_node_ids:
                return node
        return ranked[0][2]

    def _node_has_solution(self, node: LibraryNode) -> bool:
        if not node.entry_id:
            return False
        entry = self._entries.get(node.entry_id)
        return entry is not None and bool(entry.solution.strip())

    def _same_step_blocked_node_ids(self, visible_timestep_exclusive: int | None) -> set[str]:
        if visible_timestep_exclusive is None:
            return set()
        selected_ids = {
            str(group["selected_node_id"])
            for group in self._groups.values()
            if group.get("visible_timestep_exclusive") == visible_timestep_exclusive
            and group.get("selected_node_id") in self._nodes
        }
        children_map = self._build_children_map()
        blocked: set[str] = set()
        for node_id in selected_ids:
            blocked.update(self._full_lineage_node_ids(node_id, children_map))
        return blocked

    def _ancestor_node_ids(self, node_id: str) -> list[str]:
        ancestors: list[str] = []
        current = self._nodes.get(node_id)
        while current is not None:
            ancestors.append(current.id)
            current = self._nodes.get(current.parent_id) if current.parent_id else None
        return ancestors

    def _build_children_map(self) -> dict[str, set[str]]:
        children_map: dict[str, set[str]] = {}
        for node in self._nodes.values():
            if node.parent_id:
                children_map.setdefault(node.parent_id, set()).add(node.id)
        return children_map

    def _full_lineage_node_ids(self, node_id: str, children_map: dict[str, set[str]]) -> set[str]:
        lineage = set(self._ancestor_node_ids(node_id))
        queue = [node_id]
        seen = {node_id}
        while queue:
            current_id = queue.pop(0)
            for child_id in children_map.get(current_id, set()):
                if child_id in seen:
                    continue
                seen.add(child_id)
                lineage.add(child_id)
                queue.append(child_id)
        return lineage

    def _update_puct_stats_for_group(self, group: dict[str, Any]) -> None:
        parent_max: dict[str, float] = {}
        for child_id in group.get("children", []):
            child = self._nodes.get(child_id)
            if child is None or child.parent_id is None:
                continue
            parent_max[child.parent_id] = max(parent_max.get(child.parent_id, float("-inf")), float(child.value))
        for parent_id, best_child_value in parent_max.items():
            self._puct_m[parent_id] = max(float(self._puct_m.get(parent_id, best_child_value)), best_child_value)
            for ancestor_id in self._ancestor_node_ids(parent_id):
                self._puct_n[ancestor_id] = int(self._puct_n.get(ancestor_id, 0)) + 1
            self._puct_T += 1

    def _record_puct_rollout(self, parent_id: str, *, child_value: float | None) -> None:
        """Match Discover: every valid or failed rollout increments visits once."""
        if child_value is not None:
            self._puct_m[parent_id] = max(
                float(self._puct_m.get(parent_id, child_value)),
                float(child_value),
            )
        for ancestor_id in self._ancestor_node_ids(parent_id):
            self._puct_n[ancestor_id] = int(self._puct_n.get(ancestor_id, 0)) + 1
        self._puct_T += 1

    def _batch_is_complete(self, visible_timestep_exclusive: int | None) -> bool:
        step_groups = [
            group
            for group in self._groups.values()
            if group.get("visible_timestep_exclusive") == visible_timestep_exclusive
        ]
        return len(step_groups) >= self.groups_per_batch and all(
            bool(group.get("finalized", False)) for group in step_groups
        )

    def _entry_puct_value(self, entry: LibraryEntry) -> float:
        if not self.discover_compat or entry.verifier_raw_score is None:
            return float(entry.verifier_reward)
        raw_score = float(entry.verifier_raw_score)
        return raw_score if self.score_direction == "max" else -raw_score

    def _normalize_node_values(self) -> None:
        if not self.discover_compat:
            return
        for node in self._nodes.values():
            node.metadata[ARCHIVE_CANDIDATE_KEY] = bool(node.metadata.get(ARCHIVE_CANDIDATE_KEY, True))
            entry = self._entries.get(node.entry_id) if node.entry_id else None
            if entry is not None:
                node.value = self._entry_puct_value(entry)
            elif node.raw_score is not None:
                raw_score = float(node.raw_score)
                node.value = raw_score if self.score_direction == "max" else -raw_score

    def _node_is_archive_candidate(self, node: LibraryNode) -> bool:
        return not self.discover_compat or bool(node.metadata.get(ARCHIVE_CANDIDATE_KEY, True))

    def _entry_construction_key(self, entry: LibraryEntry) -> str | None:
        artifacts = (entry.metadata or {}).get("verification_artifacts") or {}
        h_values = artifacts.get("h_values")
        if entry.verifier_status == "valid" and isinstance(h_values, list) and h_values:
            return json.dumps(
                {
                    "n_points": artifacts.get("n_points", len(h_values)),
                    "c5_bound": artifacts.get("c5_bound", entry.verifier_raw_score),
                    "h_values": h_values,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        solution = entry.solution.strip()
        if solution:
            digest = hashlib.sha256(solution.encode("utf-8")).hexdigest()
            return f"solution-sha256:{digest}"
        return entry.summary or None

    def _node_construction_key(self, node: LibraryNode) -> str | None:
        if not node.entry_id:
            return None
        entry = self._entries.get(node.entry_id)
        return None if entry is None else self._entry_construction_key(entry)

    def _entry_duplicates_archive_candidate(self, entry: LibraryEntry) -> bool:
        key = self._entry_construction_key(entry)
        if key is None:
            return False
        return any(
            self._node_construction_key(node) == key
            for node in self._nodes.values()
            if self._node_is_archive_candidate(node)
        )

    def _filter_archive(self) -> None:
        candidate_ids = self._topk_child_node_ids()
        candidate_ids = self._dedup_node_ids(candidate_ids)
        candidate_ids = self._limit_buffer_node_ids(candidate_ids)
        keep_ids = self._with_ancestor_closure(candidate_ids)
        if self.discover_compat:
            for node_id in keep_ids:
                node = self._nodes.get(node_id)
                if node is not None:
                    node.metadata[ARCHIVE_CANDIDATE_KEY] = node_id in candidate_ids
        self._prune_nodes(keep_ids)

    def _topk_child_node_ids(self) -> set[str]:
        if self.topk_children <= 0:
            return {node.id for node in self._nodes.values() if self._node_is_archive_candidate(node)}
        candidate_nodes = [node for node in self._nodes.values() if self._node_is_archive_candidate(node)]
        keep_ids = {node.id for node in candidate_nodes if node.parent_id is None}
        children_by_parent: dict[str, list[LibraryNode]] = {}
        for node in candidate_nodes:
            if node.parent_id is not None:
                children_by_parent.setdefault(node.parent_id, []).append(node)
        for children in children_by_parent.values():
            children.sort(key=lambda node: (node.value, node.id), reverse=True)
            keep_ids.update(child.id for child in children[: self.topk_children])
        return keep_ids

    def _dedup_node_ids(self, candidate_ids: set[str]) -> set[str]:
        roots = {node.id for node in self._nodes.values() if node.parent_id is None}
        sorted_nodes = sorted(
            (self._nodes[node_id] for node_id in candidate_ids if node_id in self._nodes and node_id not in roots),
            key=lambda node: (node.value, node.id),
            reverse=True,
        )
        keep_ids = set(roots)
        seen_keys: set[str] = set()
        for node in sorted_nodes:
            key = self._node_construction_key(node)
            if key is not None and key in seen_keys:
                continue
            keep_ids.add(node.id)
            if key is not None:
                seen_keys.add(key)
        return keep_ids

    def _limit_buffer_node_ids(self, candidate_ids: set[str]) -> set[str]:
        if self.max_buffer_size <= 0 or len(candidate_ids) <= self.max_buffer_size:
            return candidate_ids
        roots = {node.id for node in self._nodes.values() if node.parent_id is None}
        keep_ids = {node_id for node_id in roots if node_id in candidate_ids}
        sorted_nodes = sorted(
            (self._nodes[node_id] for node_id in candidate_ids if node_id in self._nodes and node_id not in keep_ids),
            key=lambda node: (node.value, node.id),
            reverse=True,
        )
        for node in sorted_nodes:
            if len(keep_ids) >= self.max_buffer_size:
                break
            keep_ids.add(node.id)
        return keep_ids

    def _with_ancestor_closure(self, candidate_ids: set[str]) -> set[str]:
        keep_ids = {node_id for node_id in candidate_ids if node_id in self._nodes}
        for node_id in list(keep_ids):
            keep_ids.update(self._ancestor_node_ids(node_id))
        return keep_ids

    def _prune_nodes(self, keep_ids: set[str]) -> None:
        if len(keep_ids) == len(self._nodes):
            return
        self._nodes = {node_id: node for node_id, node in self._nodes.items() if node_id in keep_ids}
        for node in self._nodes.values():
            node.children = [child_id for child_id in node.children if child_id in self._nodes]
        self._puct_n = {node_id: count for node_id, count in self._puct_n.items() if node_id in self._nodes}
        self._puct_m = {node_id: value for node_id, value in self._puct_m.items() if node_id in self._nodes}

    def _node_is_visible(self, node: LibraryNode, *, visible_timestep_exclusive: int | None) -> bool:
        if visible_timestep_exclusive is None:
            return True
        if node.parent_id is None:
            return True
        return node.timestep < int(visible_timestep_exclusive)

    def _entry_is_visible(self, entry: LibraryEntry, *, visible_timestep_exclusive: int | None) -> bool:
        if visible_timestep_exclusive is None:
            return True
        return entry.timestep < int(visible_timestep_exclusive)

    def _visible_entry(
        self,
        entry_id: str | None,
        *,
        visible_timestep_exclusive: int | None,
    ) -> LibraryEntry | None:
        if entry_id is None:
            return None
        entry = self._entries.get(entry_id)
        if entry is None or not self._entry_is_visible(entry, visible_timestep_exclusive=visible_timestep_exclusive):
            return None
        return entry

    def _best_visible_node(self, *, visible_timestep_exclusive: int | None) -> LibraryNode | None:
        visible_nodes = [
            node
            for node in self._nodes.values()
            if self._node_is_visible(node, visible_timestep_exclusive=visible_timestep_exclusive)
            and self._node_is_archive_candidate(node)
        ]
        if not visible_nodes:
            return None
        return max(visible_nodes, key=lambda node: (node.value, node.id))

    def _refresh_best(self) -> None:
        candidate_nodes = [node for node in self._nodes.values() if self._node_is_archive_candidate(node)]
        if not candidate_nodes:
            self._best_node_id = None
            return
        self._best_node_id = max(candidate_nodes, key=lambda node: (node.value, node.id)).id

    def _to_store(self) -> dict[str, Any]:
        config = {
            "rollout_n": self.rollout_n,
            "puct_c": self.puct_c,
            "max_buffer_size": self.max_buffer_size,
            "topk_children": self.topk_children,
        }
        if self.puct_q_mode != PUCT_Q_BLEND:
            config["puct_q_mode"] = self.puct_q_mode
        if self.discover_compat:
            config.update(
                {
                    "discover_compat": True,
                    "groups_per_batch": self.groups_per_batch,
                    "score_direction": self.score_direction,
                }
            )
        store = {
            "nodes": {node_id: node.to_dict() for node_id, node in self._nodes.items()},
            "entries": {entry_id: entry.to_dict() for entry_id, entry in self._entries.items()},
            "groups": self._groups,
            "best_node_id": self._best_node_id,
            "config": config,
            "rollout_n": self.rollout_n,
            "puct_c": self.puct_c,
            "puct_n": self._puct_n,
            "puct_m": self._puct_m,
            "puct_T": self._puct_T,
        }
        if self.puct_q_mode != PUCT_Q_BLEND:
            store["puct_q_mode"] = self.puct_q_mode
        if self.discover_compat:
            store["discover_compat"] = True
            store["groups_per_batch"] = self.groups_per_batch
            store["score_direction"] = self.score_direction
        return store

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temp_path.write_text(json.dumps(self._to_store(), indent=2, sort_keys=True))
            temp_path.replace(self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _reload(self) -> None:
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.path.read_text())
        self._nodes = {node_id: LibraryNode.from_dict(node) for node_id, node in data.get("nodes", {}).items()}
        self._entries = {
            entry_id: LibraryEntry.from_dict(entry) for entry_id, entry in data.get("entries", {}).items()
        }
        self._groups = dict(data.get("groups", {}))
        self._best_node_id = data.get("best_node_id")
        config = data.get("config", {})
        self.rollout_n = int(config.get("rollout_n", data.get("rollout_n", self.rollout_n)))
        self.puct_c = float(config.get("puct_c", data.get("puct_c", self.puct_c)))
        self.puct_q_mode = normalize_puct_q_mode(config.get("puct_q_mode", data.get("puct_q_mode", self.puct_q_mode)))
        self.max_buffer_size = int(config.get("max_buffer_size", data.get("max_buffer_size", self.max_buffer_size)))
        self.topk_children = int(config.get("topk_children", data.get("topk_children", self.topk_children)))
        self.discover_compat = bool(config.get("discover_compat", data.get("discover_compat", self.discover_compat)))
        self.groups_per_batch = int(
            config.get("groups_per_batch", data.get("groups_per_batch", self.groups_per_batch))
        )
        self.score_direction = self._normalize_score_direction(
            config.get("score_direction", data.get("score_direction", self.score_direction))
        )
        has_puct_n = "puct_n" in data
        self._puct_n = {str(node_id): int(count) for node_id, count in (data.get("puct_n") or {}).items()}
        self._puct_m = {str(node_id): float(value) for node_id, value in (data.get("puct_m") or {}).items()}
        self._puct_T = int(data.get("puct_T", 0) or 0)
        if not has_puct_n:
            self._puct_n = {node_id: int(node.visits) for node_id, node in self._nodes.items() if node.visits}
        self._normalize_node_values()
