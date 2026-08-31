FROM python:3.12-slim
# hermes-agent main as of 2026-08-21; the stream's agent must not drift between runs.
ARG HERMES_AGENT_COMMIT=b6bcb3e791c673e63974029bbab40cc9326803ff
RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/* \
    && git init -q /opt/hermes-agent \
    && git -C /opt/hermes-agent fetch -q --depth 1 https://github.com/NousResearch/hermes-agent "$HERMES_AGENT_COMMIT" \
    && git -C /opt/hermes-agent checkout -q FETCH_HEAD \
    && pip install --no-cache-dir -e /opt/hermes-agent
COPY problem.json /agent/problem.json
WORKDIR /workspace
