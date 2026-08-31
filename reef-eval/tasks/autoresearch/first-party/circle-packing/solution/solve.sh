#!/bin/bash
# Oracle: a valid greedy packing (three r=0.25 circles, sum 0.75), submitted
# once. Proves the pipeline: env builds, the judge scores, the verifier
# finalizes. Beating this is the agent's job.
set -euo pipefail
cat > /tmp/solution.json <<'EOF'
{"circles": [[0.25, 0.25, 0.25], [0.75, 0.25, 0.25], [0.25, 0.75, 0.25]]}
EOF
python3 - <<'PY'
import json, os, urllib.request
req = urllib.request.Request(
    os.environ["JUDGE_URL"] + "/submit", data=open("/tmp/solution.json", "rb").read()
)
print(json.loads(urllib.request.urlopen(req, timeout=60).read()))
PY
