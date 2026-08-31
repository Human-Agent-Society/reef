#!/bin/bash
# Oracle: the linear trend only — decent on train, imperfect held-out.
# Deliberately NOT the true formula (which also has an oscillation);
# proves the pipeline with a mid score.
set -euo pipefail
printf '{"expr": "0.5 * x"}' > /tmp/solution.json
python3 - <<'PY'
import json, os, urllib.request
req = urllib.request.Request(
    os.environ["JUDGE_URL"] + "/submit", data=open("/tmp/solution.json", "rb").read()
)
print(json.loads(urllib.request.urlopen(req, timeout=60).read()))
PY
