#!/bin/bash
# Oracle: first-fit — valid, reward exactly 1.0. Proves the pipeline.
set -euo pipefail
python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("/app/items.json").read_text())
items, cap = data["items"], data["capacity"]
bins, loads = [], []
for i, s in enumerate(items):
    for k in range(len(bins)):
        if loads[k] + s <= cap:
            bins[k].append(i); loads[k] += s; break
    else:
        bins.append([i]); loads.append(s)
Path("/tmp/solution.json").write_text(json.dumps({"bins": bins}))
PY
python3 - <<'PY'
import json, os, urllib.request
req = urllib.request.Request(
    os.environ["JUDGE_URL"] + "/submit", data=open("/tmp/solution.json", "rb").read()
)
print(json.loads(urllib.request.urlopen(req, timeout=60).read()))
PY
