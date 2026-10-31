"""The task_check command of Claude Code as the AutoEnvScaling author: its task through the oracle and nop check.

Claude Code runs it with its Bash tool as ``python3 "$REEF_AUTHOR_CHECK"``. It needs only the Python standard
library and holds no credential: the gateway at ``REEF_AUTHOR_URL`` runs the check through Reef's generator service.
A check outlasts one Bash call (a build alone may take 20 minutes), so the command starts a check, or finds the one
that runs, and polls it. It stops waiting before the Bash call's own limit and prints ``state: running``;
``--wait CHECK_ID`` then waits for that check again, also after it ended.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping

POLL_SECONDS = 20.0
# Below the Bash tool's 600 s limit (BASH_MAX_TIMEOUT_MS), so the call ends with an answer, not a timeout.
WAIT_SECONDS = 540.0
REQUEST_TIMEOUT_SECONDS = 120.0
FINISH_TEXT = "Finish now: leave the best task you have in ./task and stop; no more checks."


def gateway(url: str, route: str, body: dict[str, object]) -> dict[str, object]:
    """The JSON object the gateway at ``url`` answers for ``body`` posted to ``route``; ``RuntimeError`` when it
    refuses or does not answer. The gateway is on the loopback address, so no proxy setting applies."""
    request = urllib.request.Request(
        f"{url}{route}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as error:
        text = error.read()[:2000].decode("utf-8", errors="replace")
        raise RuntimeError(f"{route} answered {error.code}: {text}") from error
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise RuntimeError(f"{route} did not answer: {error}") from error
    if not isinstance(answer, dict):
        raise RuntimeError(f"{route} answered no JSON object")
    return answer


def main(arguments: list[str], environ: Mapping[str, str]) -> int:
    """Start a check (or, with ``--wait``, take a started one) and print its result once it ends, then the time
    left until ``REEF_AUTHOR_DEADLINE_MS`` in ``environ``.

    Prints ``state: running`` and how to keep waiting when the check outlasts :data:`WAIT_SECONDS` or the run's
    deadline. Answers 0 with a result or a running check, and 1 when the gateway did not answer.
    """
    parser = argparse.ArgumentParser(description="Check ./task: the oracle and nop trials of the task.")
    parser.add_argument("--wait", type=int, metavar="CHECK_ID", help="wait for this check instead of starting one")
    options = parser.parse_args(arguments)
    url = environ["REEF_AUTHOR_URL"]
    deadline = float(environ["REEF_AUTHOR_DEADLINE_MS"]) / 1000
    started = time.monotonic()
    try:
        if options.wait is None:
            answer = gateway(url, "/check", {})
        else:
            answer = gateway(url, "/check", {"check_id": options.wait})
        while answer.get("state") == "running":
            waited = time.monotonic() - started
            if time.time() >= deadline or waited >= WAIT_SECONDS:
                break
            time.sleep(min(POLL_SECONDS, WAIT_SECONDS - waited))
            answer = gateway(url, "/check", {"check_id": answer["check_id"]})
    except RuntimeError as error:
        print(f"The check service failed ({error}). This uses up no check: run the check again.")
        return 1
    remaining = max(0, int(deadline - time.time()))
    if answer.get("state") != "running":
        print(json.dumps(answer, indent=2))
    else:
        check_id = answer["check_id"]
        print(f"state: running\ncheck_id: {check_id}")
        if remaining <= 0:
            print("The run's time is up: finish with the task as it stands.")
        else:
            print(
                f'The check still runs. Run `python3 "$REEF_AUTHOR_CHECK" --wait {check_id}` to wait for its result.'
            )
    print(f"Remaining time: {remaining} seconds.")
    if remaining <= float(environ["REEF_AUTHOR_FINISH_SECONDS"]):
        print(FINISH_TEXT)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], os.environ))
