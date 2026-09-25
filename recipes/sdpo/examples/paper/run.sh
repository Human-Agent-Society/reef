#!/bin/bash
# Two-step synthetic qualification in a configured Reef GPU environment.
# Download Qwen/Qwen3-0.6B to /tmp/models/Qwen3-0.6B first; use fresh state.
set -euo pipefail
example_dir="$(cd "$(dirname "$0")" && pwd)"
output_dir="${SDPO_SMOKE_OUTPUT:-$example_dir/work}"
mkdir -p "$output_dir"
python -m reef serve -c "$example_dir/smoke.yaml" > "$output_dir/reef.log" 2>&1 &
reef_pid=$!
cleanup() {
  kill "$reef_pid" 2>/dev/null || true
  wait "$reef_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready=0
for attempt in $(seq 1 360); do
  if ! kill -0 "$reef_pid" 2>/dev/null; then
    tail -100 "$output_dir/reef.log"
    exit 1
  fi
  if curl --silent --fail --max-time 5 http://127.0.0.1:28902/healthz >/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
if [ "$ready" != 1 ]; then
  echo "Reef did not become ready; see $output_dir/reef.log" >&2
  exit 1
fi
python "$example_dir/harness/smoke.py" --output "$output_dir/two-step-smoke.json"
