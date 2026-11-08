#!/usr/bin/env bash
# Exercise auto-selected student rollouts and teacher controls against an existing native OPD service.
set -euo pipefail
usage="Usage: SR_IMAGE=repository@sha256:... REEF_TOKEN=... $0 /path/to/deepmath.jsonl"
if [[ ${1:-} == --help ]]; then echo "$usage"; exit 0; fi
[[ $# == 1 && -s $1 && -f $1 ]] || { echo "$usage" >&2; exit 2; }
: "${SR_IMAGE:?Set SR_IMAGE to the digest of your local pinned Semantic Router image}"
: "${REEF_TOKEN:?Set REEF_TOKEN to the token of your native OPD service}"
[[ $SR_IMAGE =~ ^[a-z0-9./:_-]+@sha256:[a-f0-9]{64}$ ]] || { echo "SR_IMAGE requires a repository digest." >&2; exit 2; }

example=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$example/../../.." && pwd)
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
python=$("${REEF_PYTHON:-python3}" -c 'import sys; print(sys.executable)')
export REEF_PYTHON="$python"
revision=$("$python" -c 'from reef.service.deploy.routing.containers import SEMANTIC_ROUTER_REVISION; print(SEMANTIC_ROUTER_REVISION)')
installed=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$SR_IMAGE")
[[ $installed == "$revision" ]] || { echo "SR_IMAGE must have revision $revision." >&2; exit 1; }
docker compose version >/dev/null

umask 077
parent=$(cd "${WORK_DIR:-${TMPDIR:-/tmp}}" && pwd -P)
run_dir=$(mktemp -d "$parent/reef-opd-routing.XXXXXXXX")
echo "Logs, parameters, timings and receipts: $run_dir"
export XDG_STATE_HOME="$run_dir/state" ROUTE_EXP_DIR="$run_dir/results" ROUTER_RECORD_DIR="$run_dir/records"
deployment_id=opd-$("$python" -c 'import uuid; print(uuid.uuid4().hex)')
export OPD_ROUTING_SCENARIO="route-$deployment_id"
echo "Native OPD scenario: $OPD_ROUTING_SCENARIO"
ROUTER_TOKEN=$("$python" -c 'import secrets; print(secrets.token_hex(32))')
export ROUTER_TOKEN
# Select four distinct loopback ports while all four sockets are held.
ports=$("$python" - <<'PY'
import socket
from contextlib import ExitStack

with ExitStack() as stack:
    sockets = [stack.enter_context(socket.socket()) for _ in range(4)]
    for sock in sockets:
        sock.bind(("127.0.0.1", 0))
    print(*(sock.getsockname()[1] for sock in sockets))
PY
)
read -r STUDENT_BRIDGE_PORT TEACHER_BRIDGE_PORT reef_port sr_port <<<"$ports"
export STUDENT_BRIDGE_PORT TEACHER_BRIDGE_PORT OPD_ROUTER_URL="http://127.0.0.1:$reef_port"
export OPD_ROUTING_RUN_DIR="$run_dir"

cat > "$run_dir/manifest.yaml" <<YAML
schema_version: 2
automatic_model: auto
deployment: {semantic_router_image: '$SR_IMAGE'}
targets:
  student: {provider_model: student, revision: native-opd-service, url: 'http://127.0.0.1:$STUDENT_BRIDGE_PORT', role: student}
  teacher: {provider_model: teacher, revision: frozen-opd-service, url: 'http://127.0.0.1:$TEACHER_BRIDGE_PORT', role: teacher}
scenarios:
  $OPD_ROUTING_SCENARIO:
    targets: [student, teacher]
    default_target: student
    rules: [{name: teacher_control, keywords: [REEF_TEACHER_CONTROL], target: teacher}]
YAML
cat > "$run_dir/reef.yaml" <<YAML
reef:
  host: 127.0.0.1
  port: $reef_port
  tokens: ['\${ROUTER_TOKEN:?}']
  run_dir: '\${OPD_ROUTING_RUN_DIR:?}/run'
  routing_manifest: '\${OPD_ROUTING_RUN_DIR:?}/manifest.yaml'
  routing_deployment: {deployment_id: '$deployment_id', sr_management_port: $sr_port}
recipe: {implementation: recipe}
inference: {upstream_url: 'http://127.0.0.1:$STUDENT_BRIDGE_PORT', upstream_model: student, timeout_s: 180}
storage:
  artifact_repository: '\${OPD_ROUTING_RUN_DIR:?}/artifacts.git'
  artifact_work_dir: '\${OPD_ROUTING_RUN_DIR:?}/artifact-work'
  artifact_cache_dir: '\${OPD_ROUTING_RUN_DIR:?}/artifact-cache'
  agent_record_dir: '\${ROUTER_RECORD_DIR:?}'
  task_file_dir: '\${OPD_ROUTING_RUN_DIR:?}/task-files'
YAML
outer_env=(env -u REEF_TOKEN -u OPD_TEACHER_API_KEY)
"${outer_env[@]}" "$python" -m reef.service.deploy.routing.cli validate --config "$run_dir/reef.yaml"

bridge_pid= reef_pid= driver_pid=
cleanup() {
    local status=$? pid
    trap - EXIT
    trap '' INT TERM HUP
    for pid in "$driver_pid" "$reef_pid" "$bridge_pid"; do
        if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    if [[ -d $XDG_STATE_HOME/reef/routing/$deployment_id ]]; then
        "${outer_env[@]}" "$python" -m reef.service.deploy.routing.cli cleanup --deployment-id "$deployment_id" \
            >"$run_dir/cleanup.log" 2>&1 || { echo "See $run_dir/cleanup.log" >&2; status=1; }
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

env -u ROUTER_TOKEN "$python" -u "$example/bridge.py" >"$run_dir/bridge.log" 2>&1 &
bridge_pid=$!
deadline=$((SECONDS + 30))
until curl --silent --fail --max-time 3 "http://127.0.0.1:$STUDENT_BRIDGE_PORT/v1/models" >/dev/null &&
      curl --silent --fail --max-time 3 "http://127.0.0.1:$TEACHER_BRIDGE_PORT/v1/models" >/dev/null; do
    if ! kill -0 "$bridge_pid" 2>/dev/null || (( SECONDS >= deadline )); then
        echo "Start the native OPD and teacher services first. See $run_dir/bridge.log." >&2
        exit 1
    fi
    sleep 1
done
"${outer_env[@]}" "$python" -u -m reef.cli serve -c "$run_dir/reef.yaml" >"$run_dir/reef.log" 2>&1 &
reef_pid=$!
deadline=$((SECONDS + 180))
until grep -q '^\[reef-routing\].*"state": "ready"' "$run_dir/reef.log"; do
    if ! kill -0 "$bridge_pid" 2>/dev/null || ! kill -0 "$reef_pid" 2>/dev/null || (( SECONDS >= deadline )); then
        echo "Routing startup failed. See $run_dir/reef.log and $run_dir/bridge.log." >&2
        exit 1
    fi
    sleep 1
done
env -u OPD_TEACHER_API_KEY "$python" -u "$example/run_steps.py" --train-data "$1" >"$run_dir/driver.log" 2>&1 &
driver_pid=$!
if wait "$driver_pid"; then
    cat "$run_dir/driver.log"
else
    echo "Driver stopped. See $run_dir/driver.log; check the native service before starting another run." >&2
    exit 1
fi
