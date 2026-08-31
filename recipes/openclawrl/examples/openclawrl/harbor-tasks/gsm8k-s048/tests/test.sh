#!/bin/bash
mkdir -p /logs/verifier
curl -s "$JUDGE_URL/final" > /logs/verifier/final.json
python3 -c "import json; print(json.load(open('/logs/verifier/final.json'))['reward'])" > /logs/verifier/reward.txt
