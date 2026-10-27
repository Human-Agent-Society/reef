#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

: "${MODEL_DIR:?Set MODEL_DIR to a Qwen3 checkpoint}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the scratch checkpoint directory}"
: "${REEF_WORK:?Set REEF_WORK to the run directory}"
export MODEL_DIR CHECKPOINT_DIR REEF_WORK
export REEF_PORT=${REEF_PORT:-8900}
export REEF_SERVICE_URL="http://127.0.0.1:${REEF_PORT}"
export REEF_SCENARIO=${REEF_SCENARIO:-retail-l1}
export SAO_GPUS=${SAO_GPUS:-4}
method=${METHOD:-sao}
python=${REEF_PYTHON:-python3}
# Only validated integer exports come from this module.
configuration_exports=$("$python" -m harness.config)
eval "$configuration_exports"
resume=0
check_only=0
for argument in "$@"; do
    if [[ "$argument" == --resume ]]; then resume=1; fi
    if [[ "$argument" == --check-only ]]; then check_only=1; fi
done
case "$method" in
    sao|blind|oracle) ;;
    *) echo "METHOD must be sao, blind, or oracle" >&2; exit 1 ;;
esac
"$python" run.py --method "$method" --scenario "$REEF_SCENARIO" \
    --output "$REEF_WORK/results.jsonl" --max-tokens 16384 --max-steps 100 --batch-size "$SAO_BATCH" "$@" --check-only
if [[ "$check_only" == 1 ]]; then exit 0; fi
"$python" - "$REEF_WORK" "$CHECKPOINT_DIR" "$resume" <<'PY'
import sys
from pathlib import Path
resume = sys.argv[3] == "1"
for value in sys.argv[1:3]:
    path = Path(value)
    if resume:
        if not path.is_dir() or not any(path.iterdir()):
            raise SystemExit(f"Resume requires the original populated directory: {path}")
        continue
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"Use a fresh empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
PY
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$CHECKPOINT_DIR/cache}
export HF_HOME=${HF_HOME:-$XDG_CACHE_HOME/huggingface}
export MEGATRON_CONFIG_LOCK_DIR=${MEGATRON_CONFIG_LOCK_DIR:-$XDG_CACHE_HOME/megatron-locks}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$XDG_CACHE_HOME/torch-extensions}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/inductor}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$XDG_CACHE_HOME/cuda}
mkdir -p "$MEGATRON_CONFIG_LOCK_DIR"
export REEF_TOKEN=${REEF_TOKEN:-$("$python" -c 'import secrets; print(secrets.token_hex(24))')}
export REEF_MODEL_NAME=${REEF_MODEL_NAME:-$(basename "$MODEL_DIR")}
config="$PWD/serve.yaml"
# All methods share exact-token history and engine settings. Baselines send
# no reports, so their training backend remains idle throughout the campaign.
"$python" -m reef serve -c "$config" >> "$REEF_WORK/reef.log" 2>&1 &
reef_pid=$!
trap 'kill "$reef_pid" 2>/dev/null || true; wait "$reef_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready=0
for ((attempt=0; attempt<1800; attempt++)); do
    if ! kill -0 "$reef_pid" 2>/dev/null; then
        break
    fi
    if curl -fsS --max-time 2 "$REEF_SERVICE_URL/healthz" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done
if [[ "$ready" != 1 ]]; then
    echo "Reef failed to become ready. See $REEF_WORK/reef.log" >&2
    exit 1
fi
"$python" run.py --method "$method" --scenario "$REEF_SCENARIO" \
    --output "$REEF_WORK/results.jsonl" --max-tokens 16384 --max-steps 100 --batch-size "$SAO_BATCH" "$@"
