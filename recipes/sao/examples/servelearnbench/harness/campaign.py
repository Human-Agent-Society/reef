"""Durable episode results and category scores for the Retail campaign."""

import fcntl
import json
import os
import time
from pathlib import Path


class ResultLog:
    """One exclusively locked JSONL file, with ordered rows and byte offsets.

    Every append is flushed before a report can start training. Resume discards
    only an incomplete final JSON line. Other malformed rows are errors.
    The driver identifies which complete rows belong to an unfinished batch.
    """

    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.file = path.open("r+b" if resume else "x+b")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.rows: list[dict] = []
            self.offsets = [0]
            for line in self.file:
                if not line.endswith(b"\n"):
                    self.file.truncate(self.offsets[-1])
                    break
                self.rows.append(json.loads(line))
                self.offsets.append(self.offsets[-1] + len(line))
            self.file.seek(0, os.SEEK_END)
        except BaseException:
            self.file.close()
            raise

    def append(self, *rows: dict) -> None:
        for row in rows:
            data = (json.dumps(row, ensure_ascii=False) + "\n").encode()
            self.file.write(data)
            self.rows.append(row)
            self.offsets.append(self.offsets[-1] + len(data))
        self.file.flush()
        os.fsync(self.file.fileno())

    def discard_after(self, count: int) -> None:
        """Archive unreported partial-batch rows, then return to the last complete boundary."""
        if count == len(self.rows):
            return
        archive = self.path.with_name(f"{self.path.stem}.interrupted-{time.time_ns()}.jsonl")
        with archive.open("x", encoding="utf-8") as output:
            for row in self.rows[count:]:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        self.file.seek(self.offsets[count])
        self.file.truncate()
        self.file.flush()
        os.fsync(self.file.fileno())
        self.rows = self.rows[:count]
        self.offsets = self.offsets[: count + 1]

    def close(self) -> None:
        self.file.close()


def category_scores(episodes: list[dict]) -> dict[str, dict]:
    """Return task-weighted test means and counts; absent categories have a null score."""
    result = {}
    for name in ("hidden", "fully_specified"):
        rewards = [row["reward"] for row in episodes if row["phase"] == "test" and row["category"] == name]
        result[name] = {"count": len(rewards), "score": 100 * sum(rewards) / len(rewards) if rewards else None}
    return result
