#!/bin/bash
# Oracle: zlib level 9 — the honest baseline (ratio ~3.5).
set -euo pipefail
python3 - <<'PY'
import base64, json, zlib
corpus = open("/app/corpus.txt").read()
payload = base64.b64encode(zlib.compress(corpus.encode(), 9)).decode()
dec = ("import sys,base64,zlib\n"
       "sys.stdout.write(zlib.decompress(base64.b64decode(PAYLOAD)).decode())\n")
json.dump({"decompressor": dec, "payload": payload}, open("/tmp/solution.json", "w"))
PY
python3 - <<'PY'
import json, os, urllib.request
req = urllib.request.Request(
    os.environ["JUDGE_URL"] + "/submit", data=open("/tmp/solution.json", "rb").read()
)
print(json.loads(urllib.request.urlopen(req, timeout=60).read()))
PY
