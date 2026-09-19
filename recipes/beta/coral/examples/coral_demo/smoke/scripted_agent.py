#!/usr/bin/env python3
"""The scripted smoke agent: ask the gateway, write the file, ``coral eval``.

One loop iteration is one attempt. The chat call goes through CORAL's
gateway (``OPENAI_BASE_URL``/``OPENAI_API_KEY`` from the runtime), so it is
identified, journaled, and receipted like any real agent's call;
``coral eval`` commits and blocks until the grader daemon scores the
attempt. SIGINT (the manager's interrupt) exits cleanly.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request


def chat(prompt: str) -> str:
    body = json.dumps(
        {
            "model": os.environ.get("SMOKE_MODEL", "reef-policy"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        }
    ).encode()
    request = urllib.request.Request(
        os.environ["OPENAI_BASE_URL"].rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"]


def main() -> int:
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    for attempt in range(1000):
        try:
            answer = chat("Write merge_sorted(a, b): merge two ascending lists, stable on ties.")
        except Exception as error:  # gateway warming up / transient — retry, never crash-loop
            print(f"attempt {attempt}: chat failed ({error}); retrying", flush=True)
            time.sleep(3)
            continue
        with open("solution.py", "w", encoding="utf-8") as f:
            f.write(answer)
        result = subprocess.run(
            ["coral", "eval", "-m", f"scripted attempt {attempt}"],
            capture_output=True,
            text=True,
        )
        print(f"attempt {attempt}: rc={result.returncode}\n{result.stdout}{result.stderr}", flush=True)
        time.sleep(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
