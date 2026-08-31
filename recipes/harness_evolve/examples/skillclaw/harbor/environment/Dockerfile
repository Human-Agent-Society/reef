FROM python:3.12-slim

# The full campaign day runs every task inside the WildClawBench release
# image; this one task builds standalone (the benchmark Warmup block's
# `pip install -q fastapi uvicorn` is baked in) so it runs anywhere docker
# does. curl is for the agent; the instruction shows the curl call shape.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir fastapi uvicorn

# The vendored task workspace at the benchmark's own root: the two mock
# services, their ground-truth fixtures (deleted at startup, after the
# servers load them), the benchmark's tool notes (utils.py), and the task's
# skill card (the API documentation the instruction points the agent at).
COPY workspace/ /tmp_workspace/
COPY start-services.sh /usr/local/bin/start-services.sh
RUN chmod +x /usr/local/bin/start-services.sh && mkdir -p /tmp_workspace/results

WORKDIR /workspace
