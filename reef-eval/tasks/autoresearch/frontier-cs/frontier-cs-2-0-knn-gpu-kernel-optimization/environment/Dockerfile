FROM frontiercs/knn-gpu-kernel-optimization-agent:experimental-v0.6.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip ca-certificates bash curl ripgrep && \
    apt-get install -y --no-install-recommends bash ca-certificates git python3 python3-pip && \
    pip3 install --break-system-packages requests && \
    pip3 install --break-system-packages modal && \
    rm -rf /var/lib/apt/lists/*

# Pre-install Claude Code so the agent setup does not need apt at trial time.
RUN curl -fsSL https://claude.ai/install.sh | bash -s -- && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Pre-install Codex CLI for parity with the algorithmic Harbor adapter.
ENV NVM_DIR="/root/.nvm"
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && \
    . "$NVM_DIR/nvm.sh" && \
    nvm install 22 && nvm alias default 22 && \
    npm install -g @openai/codex@latest && \
    ln -sf "$(which node)" /usr/local/bin/node && \
    ln -sf "$(which codex)" /usr/local/bin/codex

ENV CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000

WORKDIR /app

# NOTE: config.yaml and task_config.json are JUDGE/operator internals (they carry
# the full `evaluation` block — seed, val-file paths, scoring knobs, and any
# undisclosed parameters like a randomized scored horizon). They are deliberately
# NOT copied into the agent workspace: the agent's task description is `readme`, and
# the submit scripts use `submission_config.json` + JUDGE_URL only. The judge reads
# its own `/judge/task_config.json`. Shipping them to /app leaked judge internals to
# the agent (e.g. the exact rollout length / tail index for nanowm_rollout_stability).
COPY readme submission_config.json AGENT.md \
    submit.py submit.sh submissions.py submissions.sh \
    wait_submission.py wait_submission.sh cancel_submission.py cancel_submission.sh /app/
COPY harbor_app/ /app/

RUN if command -v git >/dev/null 2>&1; then \
        find /app -mindepth 2 -maxdepth 4 -type d -name .git -print | \
        while read -r gitdir; do \
            repo="$(dirname "$gitdir")"; \
            git -C "$repo" config user.email frontier-cs@example.invalid; \
            git -C "$repo" config user.name "Frontier-CS"; \
            git -C "$repo" add -A; \
            if ! git -C "$repo" diff --cached --quiet; then \
                git -C "$repo" commit -q --amend --no-edit || git -C "$repo" commit -q -m base; \
            fi; \
        done; \
    fi
RUN chmod +x /app/submit.sh /app/submissions.sh /app/wait_submission.sh /app/cancel_submission.sh
