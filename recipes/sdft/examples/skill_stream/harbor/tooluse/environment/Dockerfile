# The agent's container: the stage runner and the reference datasets. No
# scoring lives here; the stage asks the judge at $JUDGE_URL to score the
# served model, and reaches the Reef service on the host.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && git clone --quiet https://github.com/idanshen/Self-Distillation /opt/self-distillation \
    && git -C /opt/self-distillation checkout --quiet d77573212fa0 \
    && pip install --no-cache-dir "reef-client>=0.2.0" "datasets>=3,<5"
WORKDIR /opt/skills
COPY skills.py stage.py task.json /opt/skills/
