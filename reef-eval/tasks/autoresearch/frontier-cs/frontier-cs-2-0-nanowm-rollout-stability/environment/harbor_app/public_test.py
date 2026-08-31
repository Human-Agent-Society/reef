#!/usr/bin/env python3
"""Agent-facing public test: static patch-policy check (no GPU).

Mirrors the judge's `validate_patch` so the agent can confirm a submission will
be accepted before enqueuing. Does NOT run the rollout or score speedup — that
happens on the judge's GPU.
"""
import sys
from pathlib import Path

# locate the shared policy: prefer the baked task package, else the local copy.
for cand in ("/judge", "/opt/nanowm/task", str(Path(__file__).resolve().parents[2])):
    if (Path(cand) / "evaluator.py").exists():
        sys.path.insert(0, cand)
        break

try:
    from evaluator import validate_patch
except Exception:
    print("NOTE: judge evaluator not available in this workspace; cannot run the "
          "exact policy check here. Submit and read the judge feedback.")
    sys.exit(0)

patch = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/solution.patch")
ok, msg, metrics = validate_patch(patch)
print(f"valid_patch={int(ok)}  files={metrics.get('files_changed')}")
print(msg)
sys.exit(0 if ok else 1)
