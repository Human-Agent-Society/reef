#!/bin/bash
# Oracle: the identity tour — valid, reward exactly 1.0. Proves the pipeline.
set -euo pipefail
python3 - <<'PY'
import json
n = len(json.load(open("/app/cities.json"))["cities"])
json.dump({"tour": list(range(n))}, open("/tmp/solution.json", "w"))
PY
python3 - <<'PY'
import json, os, urllib.request
req = urllib.request.Request(
    os.environ["JUDGE_URL"] + "/submit", data=open("/tmp/solution.json", "rb").read()
)
print(json.loads(urllib.request.urlopen(req, timeout=60).read()))
PY
