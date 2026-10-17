#!/usr/bin/env bash
# Build the Reef vLLM training image (docker/Dockerfile.reef-vllm) in three stages.
#
#   1. slimerl/sglang:$SGLANG_STAGE_TAG   SGLang's torch 2.13 / CUDA 13 image retagged under the name
#                                         Slime's Dockerfile expects, with pip allowed to install into
#                                         its externally managed interpreter.
#   2. slimerl/slime:$SLIME_STAGE_TAG     Slime's own docker/Dockerfile, taken from SLIME_DOCKERFILE_REF
#                                         (the revision that carries its CUDA 13 branches), installing
#                                         SLIME_COMMIT from Reef's runtime group, without Slime's SGLang
#                                         patches. Compiles Transformer Engine and DeepEP for CUDA_ARCHS.
#   3. $IMAGE                             docker/Dockerfile.reef-vllm with this checkout as the context.
#
# Run from any directory on a host with docker and the GPU drivers. Stage 2 takes about 16 minutes on an
# H200 host for one architecture; the three stages need roughly 20 GB of docker storage beyond the base.
# Every pin below is an environment variable so a deployment can qualify a newer base or Slime revision.
set -euo pipefail
cd "$(dirname "$0")/.."

SGLANG_IMAGE=${SGLANG_IMAGE:-lmsysorg/sglang:v0.5.18-cu130}   # sglang 0.5.18, torch 2.13.0+cu130
SGLANG_STAGE_TAG=${SGLANG_STAGE_TAG:-reef-cu130}
SLIME_DOCKERFILE_REF=${SLIME_DOCKERFILE_REF:-2f2318653f6f794dddd321eff7c9d4b7b174643f}
SLIME_COMMIT=${SLIME_COMMIT:-$(grep -o 'slime.git@[0-9a-f]*' pyproject.toml | cut -d@ -f2)}
SLIME_STAGE_TAG=${SLIME_STAGE_TAG:-reef-torch213}
CUDA_ARCHS=${CUDA_ARCHS:-90}            # Transformer Engine form, ";"-separated: 90, 100a, 103a, 120f
IMAGE=${IMAGE:-reef-vllm}
WORK=${WORK:-$(mktemp -d)}

if [ -z "$SLIME_COMMIT" ]; then
    echo "SLIME_COMMIT is empty and pyproject.toml has no slime.git pin" >&2
    exit 1
fi
# DeepEP wants the dotted arch form: 90 -> 9.0, 100a -> 10.0a.
DEEPEP_ARCHS=$(printf '%s' "$CUDA_ARCHS" | sed -E 's/([0-9]+)([0-9])([a-z]*)/\1.\2\3/g')

echo "=== stage 1: $SGLANG_IMAGE -> slimerl/sglang:$SGLANG_STAGE_TAG"
printf 'FROM %s\nENV PIP_BREAK_SYSTEM_PACKAGES=1\nRUN pip config set global.break-system-packages true\n' \
    "$SGLANG_IMAGE" > "$WORK/Dockerfile.sglang-stage"
docker build -f "$WORK/Dockerfile.sglang-stage" -t "slimerl/sglang:$SGLANG_STAGE_TAG" "$WORK"

echo "=== stage 2: Slime $SLIME_COMMIT (Dockerfile at $SLIME_DOCKERFILE_REF) -> slimerl/slime:$SLIME_STAGE_TAG"
if [ ! -d "$WORK/slime/.git" ]; then
    git init -q "$WORK/slime"
    git -C "$WORK/slime" remote add origin https://github.com/THUDM/slime.git
fi
git -C "$WORK/slime" fetch -q --depth 1 origin "$SLIME_DOCKERFILE_REF"
git -C "$WORK/slime" checkout -q FETCH_HEAD
docker build -f "$WORK/slime/docker/Dockerfile" "$WORK/slime" \
    --build-arg SGLANG_IMAGE_TAG="$SGLANG_STAGE_TAG" \
    --build-arg ENABLE_SGLANG_PATCH=0 \
    --build-arg SLIME_COMMIT="$SLIME_COMMIT" \
    --build-arg TRANSFORMER_ENGINE_CUDA_ARCHS="$CUDA_ARCHS" \
    --build-arg DEEPEP_CUDA_ARCH_LIST="$DEEPEP_ARCHS" \
    -t "slimerl/slime:$SLIME_STAGE_TAG"

echo "=== stage 3: docker/Dockerfile.reef-vllm -> $IMAGE"
docker build -f docker/Dockerfile.reef-vllm --build-arg SLIME_IMAGE_TAG="$SLIME_STAGE_TAG" -t "$IMAGE" .
echo "=== built $IMAGE"
