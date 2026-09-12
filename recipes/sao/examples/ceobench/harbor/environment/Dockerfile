FROM python:3.13-slim

# CEO-Bench at a pinned commit: the bash agent, the simulator engine, and the
# SQLCipher reader all come from this checkout. reef.patch is the example's
# only change to it (see the README); the public bundle is rebuilt so the
# engine carries the patch too.
ARG CEOBENCH_REPO=https://github.com/zlab-princeton/ceobench-src.git
ARG CEOBENCH_COMMIT=d2b7b32e5301a571b77f5f68bd1032adbcd5b464

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /usr/local/bin/uv

WORKDIR /opt/ceobench
RUN git clone --quiet "$CEOBENCH_REPO" . && git checkout --quiet "$CEOBENCH_COMMIT"
COPY reef.patch /opt/ceobench-reef.patch
RUN git apply /opt/ceobench-reef.patch \
    && uv sync --frozen \
    && uv run --no-sync python scripts/build_public.py

WORKDIR /workspace
