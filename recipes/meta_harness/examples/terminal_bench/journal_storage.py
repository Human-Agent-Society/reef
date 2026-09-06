"""Lossless compressed Reef commit journals for long benchmark campaigns.

Each fsynced JSONL frame contains the entire committed algorithm state. There
is no external blob archive or mutable checkpoint. Readers validate all frames
but retain only metadata; a record loads its state from its immutable frame on
demand. Existing plain journals are read without changing them.
"""

import base64
import hashlib
import json
import os
import zlib
from pathlib import Path
from types import MethodType

from reef.dispatcher import Dispatcher
from reef.scenario.commit_log import CommitLog, CommitLogError, CommitRecord

FORMAT = "reef-compressed-commit/1"


def _decode(frame):
    if not isinstance(frame, dict) or frame.get("storage_record") != FORMAT:
        raise CommitLogError("unknown compressed commit frame")
    try:
        payload = base64.b64decode(frame["state_zlib_base64"], validate=True)
        raw = zlib.decompress(payload)
        if len(raw) != frame["state_bytes"] or hashlib.sha256(raw).hexdigest() != frame["state_sha256"]:
            raise ValueError("state digest or length differs")
        state = json.loads(raw)
        if state is not None and not isinstance(state, dict):
            raise ValueError("state must be a mapping or null")
        if frame["commit"]["algorithm_state"] is not None:
            raise ValueError("compressed frame also contains an uncompressed state")
    except (KeyError, TypeError, ValueError, zlib.error) as exc:
        raise CommitLogError("invalid compressed algorithm state") from exc
    return state


class _StoredRecord(CommitRecord):
    @property
    def algorithm_state(self):
        location = getattr(self, "_state_location", None)
        if location is None:
            return self._initial_state
        path, offset, length, frame_sha = location
        with path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read(length)
        if hashlib.sha256(raw).hexdigest() != frame_sha:
            raise CommitLogError("an indexed committed frame changed on disk")
        return _decode(json.loads(raw))

    @algorithm_state.setter
    def algorithm_state(self, value):
        self._initial_state = value


class CompressedCommitLog(CommitLog):
    """The same append/fsync commit point with lossless state compression."""

    def append(self, record):
        value = record.to_dict()
        state = value.pop("algorithm_state")
        raw = json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        frame = {
            "storage_record": FORMAT,
            "commit": {**value, "algorithm_state": None},
            "state_sha256": hashlib.sha256(raw).hexdigest(),
            "state_bytes": len(raw),
            "state_zlib_base64": base64.b64encode(zlib.compress(raw, 3)).decode(),
        }
        line = json.dumps(frame, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        # Never append beyond an unresolved torn tail. Recovery may read the
        # earlier prefix, but treating a possibly billed tail as free is unsafe.
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    raise CommitLogError("reconcile the torn journal tail before appending")
        with self.path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def records(self):
        if not self.path.exists():
            return ()
        records = []
        # Capturing a size makes a concurrent append belong to the next read.
        # A reader never mistakes an incomplete last frame for a committed one.
        limit = self.path.stat().st_size
        with self.path.open("rb") as handle:
            while handle.tell() < limit:
                offset = handle.tell()
                raw = handle.readline(limit - offset)
                if not raw.strip():
                    continue
                try:
                    frame = json.loads(raw)
                except json.JSONDecodeError as exc:
                    if handle.tell() == limit:
                        break
                    raise CommitLogError("compressed journal has a corrupt interior frame") from exc
                _decode(frame)  # Validate every historical state without retaining all of them.
                record = _StoredRecord.from_dict(frame["commit"])
                record._state_location = (self.path, offset, len(raw), hashlib.sha256(raw).hexdigest())
                records.append(record)
        return tuple(records)


def _compressed_log_for(factory, scenario):
    if factory._agent_record_dir is None:
        return None
    key = factory._scenario_key(scenario)
    plain = factory._agent_record_dir / f"{key}.commits.jsonl"
    if plain.exists():
        raise CommitLogError("a compressed campaign must not overwrite an existing plain journal")
    return CompressedCommitLog(factory._agent_record_dir / f"{key}.commits.zjsonl")


class CompressedDispatcher(Dispatcher):
    """Install storage on this dispatcher's factory, never on global classes."""

    def __init__(self, recipe, backend_factory, **kwargs):
        class ExplicitFactory:
            # The benchmark resolves its single scenario explicitly. Do not
            # start an enumeration/preload thread before storage is installed.
            def __call__(self, scenario):
                return backend_factory(scenario)

            def has_registration(self, scenario):
                check = getattr(backend_factory, "has_registration", None)
                return bool(check and check(scenario))

        super().__init__(recipe, ExplicitFactory(), **kwargs)
        factory = self._registry._scenario_factory
        factory._commit_log_for = MethodType(_compressed_log_for, factory)


def committed_state(output):
    """Read only durable state, validating either journal format in bounded memory."""
    paths = [
        *Path(output).joinpath("records").glob("*.commits.jsonl"),
        *Path(output).joinpath("records").glob("*.commits.zjsonl"),
    ]
    if len(paths) != 1:
        raise ValueError("expected exactly one durable campaign journal")
    path = paths[0]
    last = None
    scenario = None
    if path.suffix == ".zjsonl":
        records = CompressedCommitLog(path).records()
    else:

        def stream_plain():
            limit = path.stat().st_size
            with path.open("rb") as handle:
                while handle.tell() < limit:
                    raw = handle.readline(limit - handle.tell())
                    if not raw.strip():
                        continue
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        if handle.tell() == limit:
                            break
                        raise CommitLogError("plain journal has a corrupt interior record") from exc
                    yield CommitRecord.from_dict(value)

        records = stream_plain()
    for count, record in enumerate(records, 1):
        if record.step != count or (scenario is not None and scenario != record.scenario):
            raise ValueError("campaign journal has non-contiguous steps or mixed scenarios")
        last, scenario = record, record.scenario
    if last is None or not (state := last.algorithm_state) or "terminal_bench_campaign" not in state:
        raise ValueError("journal has no committed Terminal-Bench state")
    return state, {
        "source": str(path),
        "committed_step": last.step,
        "state_sha256": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
    }
