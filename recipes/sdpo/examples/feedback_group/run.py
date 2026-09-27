"""Run a pre/post reef-eval trial around one grouped SDPO training step."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reef_client import ReefClient
from reef_eval import Lab

from recipes.sdpo.preparer import SDPOAttempt, prepare_group

HERE = Path(__file__).resolve().parent
TASK = HERE / "harbor" / "arithmetic"
AGENT = {"name": "harness:HarborAgent", "model_name": "reef"}


def training_releases() -> int:
    url = f"{os.environ['REEF_SERVICE_URL'].rstrip('/')}/reef/scenarios/{os.environ['REEF_SCENARIO']}/releases"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {os.environ['REEF_TOKEN']}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            rows = json.load(response)["releases"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return 0
        raise
    return sum(row.get("operation") == "training" for row in rows)


def wait_for_release(target: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if training_releases() >= target:
            return
        time.sleep(5)
    raise TimeoutError(f"Reef did not publish training release {target} within {timeout_s:.0f}s")


async def trial(lab: Lab, key: str, attempts_path: Path) -> tuple[Any, list[SDPOAttempt]]:
    attempts_path.unlink(missing_ok=True)
    os.environ["SDPO_ATTEMPTS_PATH"] = str(attempts_path)
    row = await lab.run(str(TASK), AGENT, key=key, tags={"phase": key})
    if row.tags.get("error") or not attempts_path.is_file():
        raise RuntimeError(f"Harbor trial {key} failed or did not save its attempts: {row.tags.get('error')}")
    attempts = [SDPOAttempt(**item) for item in json.loads(attempts_path.read_text(encoding="utf-8"))]
    return row, attempts


async def run(campaign: str) -> None:
    work = Path(os.environ["SDPO_WORK"])
    work.mkdir(parents=True, exist_ok=True)
    before_path = work / f"{campaign}-before-attempts.json"
    after_path = work / f"{campaign}-after-attempts.json"
    if before_path.exists() or after_path.exists():
        raise FileExistsError(f"campaign {campaign!r} already has attempt files; choose a new campaign name")
    lab = Lab(work / "lab")
    client = ReefClient(os.environ["REEF_SERVICE_URL"], token=os.environ["REEF_TOKEN"], timeout_s=1800)
    before = training_releases()
    pre, attempts = await trial(lab, f"{campaign}-before", before_path)
    prepared = prepare_group(attempts, expected_rollouts=8)
    for item in prepared:
        await asyncio.to_thread(client.report, os.environ["REEF_SCENARIO"], item.payload())
    await asyncio.to_thread(
        wait_for_release, before + 1, timeout_s=float(os.environ.get("SDPO_TRAIN_TIMEOUT_S", "3600"))
    )
    post, after = await trial(lab, f"{campaign}-after", after_path)
    if attempts[0].artifact_version == after[0].artifact_version:
        raise RuntimeError("post-training inference still used the pre-training artifact version")
    summary = {
        "campaign": campaign,
        "training_release": before + 1,
        "pre_artifact_version": attempts[0].artifact_version,
        "post_artifact_version": after[0].artifact_version,
        "active_responses": sum(bool(item.teacher_context) for item in prepared),
        "pre_verifier_reward": pre.rewards.get("reward"),
        "post_verifier_reward": post.rewards.get("reward"),
        "pre_successes": sum(item.score >= 0.5 for item in attempts),
        "post_successes": sum(item.score >= 0.5 for item in after),
    }
    (work / f"{campaign}-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=datetime.now(UTC).strftime("sdpo-feedback-%Y%m%dT%H%M%SZ"))
    asyncio.run(run(parser.parse_args().campaign))
