"""The task container's side of an evaluation's bridge (RFC #986): one exchange with Reef's sandbox relay.

``AdapterAgent`` uploads this file into the container and runs it through Harbor's exec, with the standard library
only; nothing in Reef imports it.

- ``python bridge_client.py PORT SECRET_FILE ready`` exits 0 once the relay answers on its loopback ``PORT``.
- ``python bridge_client.py PORT SECRET_FILE [REPLY_ID PIECE_FILE]`` first posts the JSON piece in ``PIECE_FILE`` as
  the reply to ``REPLY_ID`` and removes the file, unless the file is gone. It then waits for the relay's next request
  and prints it as JSON, or prints nothing when none came.
"""

import sys
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    base = f"http://127.0.0.1:{sys.argv[1]}"
    headers = {"x-relay-secret": Path(sys.argv[2]).read_text(encoding="utf-8").strip()}
    if sys.argv[3:] == ["ready"]:
        urllib.request.urlopen(urllib.request.Request(f"{base}/ready", headers=headers), timeout=5).read()
        return
    # A piece file that is gone was delivered by an earlier try of this exchange.
    if len(sys.argv) == 5 and Path(sys.argv[4]).exists():
        reply_id, piece = sys.argv[3], Path(sys.argv[4])
        request = urllib.request.Request(
            f"{base}/reply/{reply_id}",
            data=piece.read_bytes(),
            headers={**headers, "Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(request, timeout=30).read()
        except urllib.error.HTTPError as error:
            # 410: the caller stopped waiting, or an earlier try of this exchange delivered the piece.
            if error.code != 410:
                raise
        piece.unlink()
    with urllib.request.urlopen(urllib.request.Request(f"{base}/next", headers=headers), timeout=60) as response:
        if response.status == 200:
            sys.stdout.write(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
