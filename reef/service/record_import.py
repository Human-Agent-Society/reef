"""Stream a records JSONL file into Reef with bounded batches and resume state."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlsplit

import aiohttp


def record_batches(source: BinaryIO, *, batch_size: int, max_bytes: int) -> Iterator[tuple[bytes, int, int]]:
    """Yield encoded requests, end byte offsets and record counts, with at most one record of look-ahead."""
    prefix, suffix = b'{"records":[', b"]}"
    records: list[bytes] = []
    body_bytes = len(prefix) + len(suffix)
    end_offset = source.tell()
    while line := source.readline(max_bytes + 1):
        if len(line) > max_bytes:
            raise ValueError(f"JSONL record at byte {end_offset} exceeds the batch byte limit")
        if not line.strip():
            end_offset = source.tell()
            continue
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid JSONL record at byte {end_offset}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record at byte {end_offset} must be an object")
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(prefix) + len(encoded) + len(suffix) > max_bytes:
            raise ValueError(f"JSONL record at byte {end_offset} exceeds the batch byte limit")
        extra_bytes = len(encoded) + bool(records)
        if records and (len(records) == batch_size or body_bytes + extra_bytes > max_bytes):
            yield prefix + b",".join(records) + suffix, end_offset, len(records)
            records = []
            body_bytes = len(prefix) + len(suffix)
        records.append(encoded)
        body_bytes += len(encoded) + (len(records) > 1)
        end_offset = source.tell()
    if records:
        yield prefix + b",".join(records) + suffix, end_offset, len(records)


async def upload_batch(
    session: aiohttp.ClientSession, endpoint: str, body: bytes, *, scenario: str, max_retries: int
) -> None:
    """Retry transient failures with unchanged IDs; require every record receipt."""
    expected = [
        {"agent_record_id": record["agent_record_id"], "request_type": record["request_type"], "scenario": scenario}
        for record in json.loads(body)["records"]
    ]
    for attempt in range(max_retries + 1):
        delay_seconds = min(2**attempt, 30)
        try:
            async with session.post(endpoint, data=body, allow_redirects=False) as response:
                if response.status == 200:
                    result = await response.json()
                    if not isinstance(result, dict) or result.get("records") != expected:
                        raise ValueError("batch response did not acknowledge the submitted records")
                    return
                if response.status not in (408, 429, 500, 502, 503, 504) or attempt == max_retries:
                    raise ValueError(f"batch rejected with HTTP {response.status}; import progress was not advanced")
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdecimal():
                    delay_seconds = min(int(retry_after), 60)
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, TimeoutError):
            if attempt == max_retries:
                raise
        await asyncio.sleep(delay_seconds)


def save_progress(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as target:
        json.dump(state, target)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


async def import_records_file(
    source_path: Path,
    *,
    url: str,
    scenario: str,
    progress_path: Path | None = None,
    token: str | None = None,
    batch_size: int = 128,
    max_batch_bytes: int = 512 * 1024,
    max_retries: int = 3,
) -> int:
    """Import immutable JSONL records; return the cumulative acknowledged count.

    A local checkpoint is bound to the file checksum, destination and scenario.
    It advances only after a whole batch is acknowledged. An interrupted batch
    is resent with the source IDs, so server-side deduplication handles a lost
    response. Concurrent use of the same checkpoint is rejected.
    """
    url = url.rstrip("/")
    destination = urlsplit(url)
    if destination.scheme not in ("http", "https") or not destination.hostname:
        raise ValueError("url must be an HTTP(S) Reef base URL")
    if destination.username or destination.password or destination.query or destination.fragment:
        raise ValueError("url must not contain credentials, a query or a fragment; use REEF_TOKEN")
    scenario = scenario.strip()
    if not scenario:
        raise ValueError("scenario must not be empty")
    if not 1 <= batch_size <= 1000 or not 128 <= max_batch_bytes < 1024 * 1024 or max_retries < 0:
        raise ValueError("require batch-size 1..1000, max-batch-bytes 128..1048575 and nonnegative retries")
    source_path = source_path.resolve()
    progress_path = (progress_path or source_path.with_name(source_path.name + ".reef-import.json")).resolve()
    lock_path = progress_path.with_name(progress_path.name + ".lock")
    if source_path in (progress_path, lock_path, progress_path.with_name(progress_path.name + ".tmp")):
        raise ValueError("progress files must not overwrite the source")
    with lock_path.open("a") as lock, source_path.open("rb") as source:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another importer is using this progress file") from exc
        initial_stat = os.fstat(source.fileno())
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
        identity = {"version": 1, "sha256": checksum, "url": url, "scenario": scenario, "size": initial_stat.st_size}
        offset = count = 0
        if progress_path.exists():
            state = json.loads(progress_path.read_text())
            if not isinstance(state, dict) or any(state.get(key) != value for key, value in identity.items()):
                raise ValueError("progress belongs to a different file, destination or scenario")
            saved_offset, saved_count = state.get("offset"), state.get("count")
            if (
                type(saved_offset) is not int
                or type(saved_count) is not int
                or not 0 <= saved_offset <= initial_stat.st_size
                or saved_count < 0
            ):
                raise ValueError("invalid import progress")
            offset, count = saved_offset, saved_count
        source.seek(offset)
        save_progress(progress_path, {**identity, "offset": offset, "count": count})
        headers = {"x-reef-scenario": scenario, "Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as session:
            for body, end_offset, batch_count in record_batches(
                source, batch_size=batch_size, max_bytes=max_batch_bytes
            ):
                current_stat = os.fstat(source.fileno())
                if (current_stat.st_size, current_stat.st_mtime_ns) != (
                    initial_stat.st_size,
                    initial_stat.st_mtime_ns,
                ):
                    raise ValueError("source file changed during import")
                try:
                    await upload_batch(
                        session, url + "/reef/records/batch", body, scenario=scenario, max_retries=max_retries
                    )
                except (ValueError, KeyError) as exc:
                    raise ValueError(f"import stopped at byte {offset}: {exc}") from exc
                offset = end_offset
                count += batch_count
                save_progress(progress_path, {**identity, "offset": offset, "count": count})
                print(f"Acknowledged {count} records ({offset}/{initial_stat.st_size} bytes)", file=sys.stderr)
        return count


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="reef import", description=__doc__)
    parser.add_argument("file", type=Path, help="JSONL file containing one record envelope per line")
    parser.add_argument("--url", required=True, help="Reef service base URL")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--progress", type=Path, help="checkpoint path; defaults to FILE.reef-import.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batch-bytes", type=int, default=512 * 1024)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        count = asyncio.run(
            import_records_file(
                args.file,
                url=args.url,
                scenario=args.scenario,
                progress_path=args.progress,
                token=os.environ.get("REEF_TOKEN"),
                batch_size=args.batch_size,
                max_batch_bytes=args.max_batch_bytes,
                max_retries=args.retries,
            )
        )
    except (OSError, ValueError, aiohttp.ClientError, TimeoutError) as exc:
        parser.exit(1, f"reef import: {exc}\n")
    print(f"Import complete: {count} records acknowledged. Training may still be running.")
