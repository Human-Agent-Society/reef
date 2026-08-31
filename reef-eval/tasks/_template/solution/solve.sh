#!/bin/bash
# Oracle baseline: write a valid, deliberately beatable solution and submit
# it to the judge once. TODO(task): replace with your reference solution.
set -euo pipefail
printf '{"x": 0.5}' > /tmp/solution.json
python3 - <<'PY'
import json, os, urllib.request
req = urllib.request.Request(
    os.environ["JUDGE_URL"] + "/submit", data=open("/tmp/solution.json", "rb").read()
)
print(json.loads(urllib.request.urlopen(req, timeout=60).read()))
PY
