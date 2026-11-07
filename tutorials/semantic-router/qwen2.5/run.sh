#!/usr/bin/env bash
# Serve both Qwen models on one GPU, run the routing example, then stop this run.
set -euo pipefail
: "${SR_IMAGE:?Set SR_IMAGE to the digest of your local pinned Semantic Router image}"
example=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$example/../../.." && pwd)
cd "$repo"

python=${REEF_PYTHON:-python3}
python=$("$python" -c 'import sys; print(sys.executable)')
export REEF_PYTHON="$python"
gpu=${GPU_DEVICE:-0}
[[ $gpu =~ ^([0-9]+|GPU-[[:xdigit:]-]+)$ ]] || { echo "GPU_DEVICE must name one GPU index or UUID" >&2; exit 1; }
vllm_image=vllm/vllm-openai@sha256:014a95f21c9edf6abe0aea6b07353f96baa4ec291c427bb1176dc7c93a85845c
student_model=Qwen/Qwen2.5-0.5B-Instruct
teacher_model=Qwen/Qwen2.5-1.5B-Instruct
student_revision=7ae557604adf67be50417f59c2c2f167def9a775
teacher_revision=989aa7980e4cf806f80c7fef2b1adb7bc71aa306

[[ $SR_IMAGE =~ @sha256:[[:xdigit:]]{64}$ ]] || { echo "SR_IMAGE requires repository@sha256:<digest>" >&2; exit 1; }
expected_revision=$("$python" -c 'from reef.service.deploy.routing.containers import SEMANTIC_ROUTER_REVISION; print(SEMANTIC_ROUTER_REVISION)')
image_revision=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$SR_IMAGE")
[[ $image_revision == "$expected_revision" ]] || { echo "SR_IMAGE must have revision $expected_revision" >&2; exit 1; }
docker compose version >/dev/null
if ! docker image inspect "$vllm_image" >/dev/null 2>&1; then
    if [[ ${HF_HUB_OFFLINE:-0} == 1 ]]; then
        echo "Offline run requires the local image $vllm_image" >&2
        exit 1
    fi
    docker pull "$vllm_image"
fi

umask 077
work_parent=$(cd "${WORK_DIR:-${TMPDIR:-/tmp}}" && pwd -P)
run_dir=$(mktemp -d "$work_parent/reef-qwen.XXXXXXXX")
deployment_id=qwen-$("$python" -c 'import uuid; print(uuid.uuid4().hex)')
label="io.reef.qwen-example=$deployment_id"
export XDG_STATE_HOME="$run_dir/state"
REEF_TOKEN=$("$python" -c 'import secrets; print(secrets.token_hex(32))')
export REEF_TOKEN
printf 'Run directory: %s\n' "$run_dir"

cleanup() {
    local result=$? children containers remaining id role
    trap - EXIT
    trap '' INT TERM HUP
    set +e
    children=$(jobs -pr)
    for id in $children; do kill -TERM "$id" 2>/dev/null; done
    for id in $children; do wait "$id" 2>/dev/null; done
    if [[ -d "$XDG_STATE_HOME/reef/routing/$deployment_id" ]]; then
        if ! "$python" -m reef.service.deploy.routing.cli cleanup --deployment-id "$deployment_id" >"$run_dir/cleanup.log" 2>&1; then
            echo "Reef cleanup failed. Inspect $run_dir/cleanup.log." >&2
            result=1
        fi
    fi
    if containers=$(docker ps -aq --no-trunc --filter "label=$label"); then
        for id in $containers; do
            role=$(docker inspect --format '{{.Name}}' "$id")
            docker stop -t 10 "$id" >/dev/null || result=1
            docker logs "$id" >"$run_dir/${role##*-}.log" 2>&1
            docker rm -fv "$id" >/dev/null || result=1
        done
        if remaining=$(docker ps -aq --filter "label=$label") && [[ -z $remaining ]]; then
            rm -rf -- "$run_dir/weights"
        else
            echo "Could not confirm removal of the model containers with label $label." >&2
            result=1
        fi
    else
        result=1
    fi
    printf 'Logs and records: %s\n' "$run_dir"
    exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

prepare_model() {
    local role=$1 model=$2 revision=$3 source=$4 directory="$run_dir/weights/$1"
    local files=() filename
    while read -r _ filename; do files+=("$filename"); done <"$example/$role.sha256"
    if [[ -z $source ]]; then
        "$python" - "$model" "$revision" "${files[@]}" >"$run_dir/$role.source" <<'PY' &
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=sys.argv[3:], token=False))
PY
        wait "$!"
        read -r source <"$run_dir/$role.source"
    fi
    mkdir -p "$directory"
    # Dereference HF snapshot links. Only the copies are made readable for the container user.
    for filename in "${files[@]}"; do cp -L --reflink=auto -- "$source/$filename" "$directory/$filename"; done
    chmod 755 "$directory"
    chmod 644 "$directory"/*
    (cd "$directory" && sha256sum --strict --check "$example/$role.sha256")
}

start_model() {
    local role=$1 model=$2 fraction=$3 id port deadline
    id=$(docker run -d --pull=never --name "$deployment_id-$role" --label "$label" \
        --gpus "device=$gpu" --cpus 4 --memory 10g --shm-size 1g \
        --cap-drop ALL --security-opt no-new-privileges -p 127.0.0.1::8000 \
        -v "$run_dir/weights/$role:/model:ro" \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
        "$vllm_image" --model /model --served-model-name "$model" --dtype bfloat16 \
        --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 \
        --gpu-memory-utilization "$fraction" --enforce-eager --generation-config vllm \
        --disable-log-requests --host 0.0.0.0 --port 8000)
    printf 'Starting %s\n' "$model"
    deadline=$((SECONDS + 420))
    while [[ $(docker inspect --format '{{.State.Running}}' "$id") == true ]] && ((SECONDS < deadline)); do
        # Docker Desktop can publish the dynamic port after returning the container ID.
        port=$(docker inspect --format '{{with index .NetworkSettings.Ports "8000/tcp"}}{{(index . 0).HostPort}}{{end}}' "$id")
        if [[ ! $port =~ ^[0-9]+$ ]]; then sleep 1; continue; fi
        if curl -fs --noproxy '*' --max-time 3 "http://127.0.0.1:$port/v1/models" | \
            "$python" -c 'import json, sys; sys.exit(sys.argv[1] not in [m["id"] for m in json.load(sys.stdin)["data"]])' "$model" 2>/dev/null; then
            model_port=$port
            return
        fi
        sleep 2
    done
    echo "$model did not become ready. See $run_dir/$role.log after cleanup." >&2
    return 1
}

prepare_model student "$student_model" "$student_revision" "${STUDENT_MODEL_DIR:-}"
prepare_model teacher "$teacher_model" "$teacher_revision" "${TEACHER_MODEL_DIR:-}"
start_model student "$student_model" 0.20
student_port=$model_port
start_model teacher "$teacher_model" 0.25
teacher_port=$model_port

# Ask the kernel for two distinct free ports while both probe sockets are held.
ports=$("$python" - <<'PY'
import socket
with socket.socket() as reef, socket.socket() as sr:
    for sock in (reef, sr):
        sock.bind(("127.0.0.1", 0))
    print(reef.getsockname()[1], sr.getsockname()[1])
PY
)
read -r reef_port sr_port <<<"$ports"
"$python" "$example/check.py" manifests --output-directory "$run_dir" --image "$SR_IMAGE" \
    --student-revision "$student_revision" --teacher-revision "$teacher_revision" \
    --student-port "$student_port" --teacher-port "$teacher_port"

# Environment references in paths let Reef handle spaces and YAML-sensitive characters.
export QWEN_RUN_DIR="$run_dir"
cat >"$run_dir/reef.yaml" <<YAML
reef:
  host: 127.0.0.1
  port: $reef_port
  tokens: ['\${REEF_TOKEN:?}']
  run_dir: '\${QWEN_RUN_DIR:?}/run'
  routing_manifest: '\${QWEN_RUN_DIR:?}/manifest.yaml'
  routing_deployment:
    deployment_id: $deployment_id
    sr_management_port: $sr_port
recipe: {implementation: recipe}
inference:
  upstream_url: http://127.0.0.1:$student_port
  upstream_model: $student_model
  timeout_s: 120
storage:
  artifact_repository: '\${QWEN_RUN_DIR:?}/artifacts.git'
  artifact_work_dir: '\${QWEN_RUN_DIR:?}/artifact-work'
  artifact_cache_dir: '\${QWEN_RUN_DIR:?}/artifact-cache'
  agent_record_dir: '\${QWEN_RUN_DIR:?}/records'
  task_file_dir: '\${QWEN_RUN_DIR:?}/task-files'
YAML
"$python" -m reef.service.deploy.routing.cli validate --config "$run_dir/reef.yaml"
"$python" -m reef.cli serve -c "$run_dir/reef.yaml" >"$run_dir/reef.log" 2>&1 &
reef_pid=$!
deadline=$((SECONDS + 150))
until grep -q '^\[reef-routing\].*"state": "ready"' "$run_dir/reef.log"; do
    if ! kill -0 "$reef_pid" 2>/dev/null || ((SECONDS >= deadline)); then
        echo "Reef did not become ready. See $run_dir/reef.log." >&2
        exit 1
    fi
    sleep 1
done
printf 'Reef ready on 127.0.0.1:%s\n' "$reef_port"
"$python" "$example/check.py" check --run-directory "$run_dir" --deployment-id "$deployment_id" \
    --reef-pid "$reef_pid" --reef-port "$reef_port" "$@" >"$run_dir/result.json" &
status=0
wait "$!" || status=$?
cat "$run_dir/result.json"
exit "$status"
