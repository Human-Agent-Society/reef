#!/usr/bin/env bash
# Start the OPD math recipe with a four-sample batch and the tiny Qwen3 models.
set -euo pipefail
usage="Usage: $0 [--config-only] (set OPD_MODEL_PATH, OPD_TEACHER_PATH, OPD_RUN_DIR and REEF_TOKEN)"
if [[ ${1:-} == --help ]]; then echo "$usage"; exit 0; fi
[[ $# == 0 || ($# == 1 && $1 == --config-only) ]] || { echo "$usage" >&2; exit 2; }
: "${OPD_MODEL_PATH:?Set the absolute path to your Qwen3-0.6B checkpoint}"
: "${OPD_TEACHER_PATH:?Set the absolute path to your Qwen3-1.7B checkpoint}"
: "${OPD_RUN_DIR:?Choose an absolute, new directory for this native OPD run}"
: "${REEF_TOKEN:?Set a private token for the native Reef endpoint}"
for path in "$OPD_MODEL_PATH" "$OPD_TEACHER_PATH"; do
    [[ $path == /* && -f $path/config.json ]] || { echo "Expected a model directory with config.json: $path" >&2; exit 2; }
done
[[ $OPD_RUN_DIR == /* ]] || { echo "OPD_RUN_DIR must be absolute." >&2; exit 2; }
umask 077
mkdir -m 700 -- "$OPD_RUN_DIR"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo"
python=${REEF_PYTHON:-python3}
"$python" - "$OPD_RUN_DIR/serve.yaml" <<'PY'
import os
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path("recipes/opd/examples/math/serve.yaml").read_text())
config["reef"]["host"] = os.environ.get("OPD_BIND_HOST", "127.0.0.1")
inference = config["inference"]
inference["options"].update({"context-length": 2048, "attention-backend": "triton"})
inference["handler-config"]["sampling_defaults"]["max_new_tokens"] = 256
config["teacher"]["options"].update({"context-length": 2048, "attention-backend": "flashinfer", "max-running-requests": 4})
config["pipeline"]["record2sample"]["config"]["max_teacher_tokens"] = 2048
training = config["training"]
training["ray-address"] = os.environ.get("OPD_RAY_ADDRESS", training["ray-address"])
training["config"]["global_batch_size"] = 4
training["config"]["checkpoint_retention"].update({"max_storage": "16GB", "min_free_space": "10GB"})
training["options"].update({"seq-length": 2048, "rollout-max-response-len": 256, "rollout-max-context-len": 2048,
                            "max-tokens-per-gpu": 2048, "log-probs-chunk-size": "512"})
config["storage"]["task-file-dir"] = "${OPD_RUN_DIR:?}/task-files"
with open(sys.argv[1], "x") as output:
    yaml.safe_dump(config, output, sort_keys=False)
PY
echo "Native OPD configuration: $OPD_RUN_DIR/serve.yaml"
if [[ ${1:-} == --config-only ]]; then exit 0; fi
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -m reef serve -c "$OPD_RUN_DIR/serve.yaml"
