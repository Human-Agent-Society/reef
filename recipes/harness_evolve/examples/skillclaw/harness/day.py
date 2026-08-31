"""The pinned benchmark and the official task lifecycle.

Ported from benchmarks/skill_claw/day.py at commit 0519eefb (branch
bo/skillclaw-paper). ensure_benchmark clones WildClawBench at the pin; the
bootstrap pool is the benchmark's own skills directory. run_task copies the
full task workspace (gt included, as their setup does), applies their
runtime configuration, runs the agent under their preamble for exactly the
task timeout, and grades with their budget and transient retries. Any task
error voids grading: the score is None, never a fake zero.

Adaptations to the rebuild: paths root under this example's ./work, the
``pool`` argument is any directory containing a ``skills/`` tree (the driver
passes the pulled artifact's ``pi-agent`` directory), and parse failures on
the task file land in the task's own error verdict instead of killing the
day's worker pool.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from harness.config import AGENT_MODEL, GATEWAY_KEY, HERE, JUDGE_MODEL, OPENROUTER_BASE, read_judge_key
from harness.prompts import AGENT_PREAMBLE, OCRBENCH_HINT, OCRBENCH_TASK_ID

WILDCLAWBENCH_URL = "https://github.com/InternLM/WildClawBench"
WILDCLAWBENCH_PIN = "1e32bacb904335881de26766defeda3475757c6a"
HF_DATASET = "internlm/WildClawBench"

RELEASE_DIR = Path(os.environ.get("REEF_SC_RELEASE", str(HERE / "work")))
WILDCLAWBENCH_ROOT = RELEASE_DIR / "wildclawbench"


GATEWAY_PORT = 18789
GRADE_TIMEOUT_S = 600
GRADE_RETRY_DELAYS_S = (5.0, 15.0, 30.0, 60.0)
WARMUP_RETRY_DELAYS_S = (5.0, 15.0, 30.0)
SETUP_RETRY_DELAYS_S = (1.0, 2.0, 4.0, 8.0, 16.0)
SETUP_TIMEOUT_S = 180
EXEC_READY_TIMEOUT_S = 20
CONTEXT_WINDOW = 200000
AGENT_MAX_TOKENS = 32768
SYMLINK_THRESHOLD_BYTES = 200 * 1024 * 1024

_DESTRUCTIVE_WARMUP_LINES = {"rm -f -r /tmp_workspace/tmp", "rm -rf /tmp_workspace/fixtures"}
_PIP_INSTALL = re.compile(r"^((?:\S+/)?pip(?:3)?)\s+install\b")

_RETRYABLE_GRADE_TOKENS = (
    "CONNECTION ERROR",
    "API CONNECTION ERROR",
    "SERVER DISCONNECTED",
    "TIMED OUT",
    "READ TIMED OUT",
    "TIMEOUT",
    "CONNECTION RESET",
    "ECONNRESET",
    "BAD GATEWAY",
    "SERVICE UNAVAILABLE",
    "GATEWAY TIMEOUT",
)

_RETRYABLE_WARMUP_TOKENS = (
    "CONNECTION TIMED OUT",
    "READ TIMED OUT",
    "CONNECTIONRESETERROR",
    "ECONNRESET",
    "ETIMEDOUT",
    "ECONNREFUSED",
    "EAI_AGAIN",
    "NETWORK ABORTED",
    "SSL: DECRYPTION_FAILED",
    "CERTIFICATE_VERIFY_FAILED",
    "INCOMPLETE DOWNLOAD",
    "CONNECTIONERROR",
    "COULD NOT FIND A VERSION THAT SATISFIES",
    "NO MATCHING DISTRIBUTION",
    "DO NOT MATCH THE HASHES",
)

_RETRYABLE_AGENT_BROWSER_TOKENS = (
    "ERR_SSL_DECRYPTION_FAILED",
    "ERR_SSL_CIPHER_OPERATION_FAILED",
)

_SETUP_SCRIPT = """
set -e
python3 - <<'PY'
import os
import shutil
from pathlib import Path

SRC = Path('/app')
DST = Path('/tmp_workspace')
THRESHOLD = {threshold}
DST.mkdir(parents=True, exist_ok=True)

def _copy_tree(src, dst):
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        dst_root = dst / rel
        dst_root.mkdir(parents=True, exist_ok=True)
        for name in dirs:
            (dst_root / name).mkdir(parents=True, exist_ok=True)
        for name in files:
            source = root_path / name
            target = dst_root / name
            if target.is_symlink() or target.exists():
                target.unlink()
            if source.stat().st_size >= THRESHOLD:
                link_target = os.readlink(source) if source.is_symlink() else str(source)
                os.symlink(link_target, target)
            else:
                shutil.copy2(source, target)

_copy_tree(SRC, DST)
exec_root = SRC / 'exec'
if exec_root.is_dir():
    _copy_tree(exec_root, DST)
PY
find /tmp_workspace -type d -exec chmod u+rwx {{}} +
find /tmp_workspace -type f -exec chmod u+rw {{}} +
if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  ln -sf "$(command -v python3)" /usr/local/bin/python
fi
if command -v bash >/dev/null 2>&1; then
  ln -sf "$(command -v bash)" /bin/sh
fi
mkdir -p /tmp_workspace/results
PW_CHROME="$(find /root/.cache/ms-playwright -path '*/chrome-linux64/chrome' -type f 2>/dev/null | head -n 1)"
if [ -n "$PW_CHROME" ]; then
  ln -sf "$PW_CHROME" /usr/local/bin/google-chrome
  ln -sf "$PW_CHROME" /usr/local/bin/chromium
fi
mkdir -p /root/memory
today="$(date +%F)"
yesterday="$(date -d yesterday +%F 2>/dev/null || python3 -c 'from datetime import date, timedelta; print((date.today() - timedelta(days=1)).isoformat())')"
touch "/root/memory/${{today}}.md" "/root/memory/${{yesterday}}.md"
rm -rf /root/.openclaw/workspace
mkdir -p /root/.openclaw
ln -s /tmp_workspace /root/.openclaw/workspace
"""


def ensure_benchmark() -> Path:
    """Clone at the pin and require the task data a run cannot fetch itself."""
    if not (WILDCLAWBENCH_ROOT / ".git").exists():
        WILDCLAWBENCH_ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", WILDCLAWBENCH_URL, str(WILDCLAWBENCH_ROOT)], check=True)
    subprocess.run(
        ["git", "-C", str(WILDCLAWBENCH_ROOT), "checkout", "--force", "--quiet", WILDCLAWBENCH_PIN], check=True
    )
    if not (WILDCLAWBENCH_ROOT / "workspace").is_dir():
        raise RuntimeError(
            f"task workspaces missing; download the {HF_DATASET} dataset into {WILDCLAWBENCH_ROOT} first"
        )
    return WILDCLAWBENCH_ROOT


def image_tag() -> str:
    existing = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "wildclawbench-ubuntu"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if existing:
        return existing[0]
    tarballs = sorted((WILDCLAWBENCH_ROOT / "Images").glob("*.tar"))
    if not tarballs:
        raise RuntimeError(f"no image loaded and no tarball under {WILDCLAWBENCH_ROOT / 'Images'}")
    loaded = subprocess.run(
        ["docker", "load", "-i", str(tarballs[-1])], check=True, capture_output=True, text=True
    ).stdout
    match = re.search(r"Loaded image:\s*(\S+)", loaded)
    if match is None:
        raise RuntimeError(f"docker load reported no image tag: {loaded[:200]}")
    return match.group(1)


def _sh(container: str, command: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "exec", container, "bash", "-lc", command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _wait_exec_ready(container: str) -> None:
    """Their exec ready gate: poll until docker exec answers or time out."""
    deadline = time.monotonic() + EXEC_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if subprocess.run(["docker", "exec", container, "/bin/true"], capture_output=True).returncode == 0:
            return
        time.sleep(0.5)
    raise RuntimeError(f"{container} did not answer docker exec within {EXEC_READY_TIMEOUT_S}s")


def _setup_workspace(container: str) -> None:
    """Their setup with the exec ready gate, retry ladder, and final raise."""
    last_error = ""
    for attempt in range(len(SETUP_RETRY_DELAYS_S) + 1):
        _wait_exec_ready(container)
        try:
            proc = _sh(container, _SETUP_SCRIPT.format(threshold=SYMLINK_THRESHOLD_BYTES), timeout=SETUP_TIMEOUT_S)
            if proc.returncode == 0:
                return
            last_error = f"{proc.stderr or ''}\n{proc.stdout or ''}".strip()
        except subprocess.TimeoutExpired as exc:
            last_error = f"timeout after {SETUP_TIMEOUT_S}s: {exc}"
        if attempt >= len(SETUP_RETRY_DELAYS_S):
            break
        time.sleep(SETUP_RETRY_DELAYS_S[attempt])
    raise RuntimeError(f"workspace setup failed after {len(SETUP_RETRY_DELAYS_S) + 1} attempts: {last_error[:1200]}")


def _inject_models(container: str, *, base_url: str, key: str) -> None:
    """Their models injection: the provider block written into openclaw.json."""
    models = {
        "providers": {
            "custom-gateway": {
                "baseUrl": base_url,
                "apiKey": key,
                "api": "openai-completions",
                "models": [
                    {
                        "id": AGENT_MODEL,
                        "name": AGENT_MODEL,
                        "input": ["text", "image"],
                        "reasoning": True,
                        "contextWindow": CONTEXT_WINDOW,
                        "maxTokens": AGENT_MAX_TOKENS,
                    }
                ],
            }
        }
    }
    inject = (
        "python3 - <<'PY'\n"
        "import json, pathlib\n"
        "config_path = pathlib.Path('/root/.openclaw/openclaw.json')\n"
        f"models = json.loads({json.dumps(json.dumps(models, ensure_ascii=False))})\n"
        "config = json.loads(config_path.read_text()) if config_path.exists() else {}\n"
        "config['models'] = models\n"
        "config_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "config_path.write_text(json.dumps(config, indent=2))\n"
        "PY"
    )
    proc = _sh(container, inject, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"models injection failed: {proc.stderr.strip()[:400]}")


def _strip_fence(text: str) -> str:
    """Their rule: strip one leading and one trailing fence; interior fences survive."""
    return re.sub(r"\n?```$", "", re.sub(r"^```[^\n]*\n?", "", text.strip())).strip()


def parse_task(task_file: Path) -> dict[str, Any]:
    """Frontmatter and sections of one task markdown, benchmark format."""
    text = task_file.read_text(encoding="utf-8")
    front = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if front is None:
        raise ValueError(f"no frontmatter in {task_file}")
    meta: dict[str, str] = {}
    for line in front.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    sections: dict[str, str] = {}
    current = None
    for line in front.group(2).splitlines():
        header = re.match(r"^##\s+(.+)$", line)
        if header:
            current = header.group(1)
            sections[current] = ""
        elif current:
            sections[current] += line + "\n"
    return {
        "task_id": meta.get("id") or task_file.stem,
        "timeout_seconds": int(meta.get("timeout_seconds", "120")),
        "prompt": sections.get("Prompt", "").strip(),
        "workspace": _strip_fence(sections.get("Workspace Path", "")),
        "checks": _strip_fence(sections.get("Automated Checks", "")),
        "warmup": _strip_fence(sections.get("Warmup", "")),
        "env": _strip_fence(sections.get("Env", "")),
    }


def compose_prompt(task: dict[str, Any]) -> str:
    """Their composed preamble, the shipped per task hint, then the task."""
    hint = OCRBENCH_HINT if task["task_id"] == OCRBENCH_TASK_ID else ""
    return AGENT_PREAMBLE.format(timeout_seconds=task["timeout_seconds"]) + hint + task["prompt"]


def _enhance_pip(cmd: str) -> str:
    """Their pip hardening for concurrent-warmup bandwidth contention."""
    match = _PIP_INSTALL.match(cmd.strip())
    if match is None or "--timeout" in cmd or "--retries" in cmd:
        return cmd
    return f"{match.group(1)} install --timeout 300 --retries 5{cmd.strip()[match.end() :]}"


def _retryable_warmup(cmd: str, error_text: str) -> bool:
    upper = error_text.upper()
    if any(token in upper for token in _RETRYABLE_WARMUP_TOKENS):
        return True
    if not cmd.strip().startswith("npm install -g agent-browser"):
        return False
    return any(token in upper for token in _RETRYABLE_AGENT_BROWSER_TOKENS)


def _run_warmup(container: str, warmup: str, output_dir: Path) -> None:
    """Their warmup semantics; a persistent failure raises a task error."""
    bg_index = 0
    for line in warmup.splitlines():
        cmd = line.strip()
        if not cmd or cmd.startswith("#"):
            continue
        if cmd in _DESTRUCTIVE_WARMUP_LINES:
            # The setup copy already placed the fixtures these lines wipe.
            continue
        if cmd.endswith("&"):
            log = output_dir / "warmup" / f"bg-{bg_index:02d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            bg_index += 1
            subprocess.Popen(
                ["docker", "exec", container, "bash", "-c", f"cd /tmp_workspace && {cmd[:-1].rstrip()}"],
                stdout=log.open("w", encoding="utf-8"),
                stderr=subprocess.STDOUT,
                encoding="utf-8",
            )
            continue
        cmd = _enhance_pip(cmd)
        proc: subprocess.CompletedProcess[str] | None = None
        for attempt in range(len(WARMUP_RETRY_DELAYS_S) + 1):
            proc = subprocess.run(["docker", "exec", container, "bash", "-c", cmd], capture_output=True, text=True)
            if proc.returncode == 0:
                break
            error_text = f"{proc.stderr or ''}\n{proc.stdout or ''}".strip()
            if attempt >= len(WARMUP_RETRY_DELAYS_S) or not _retryable_warmup(cmd, error_text):
                break
            time.sleep(WARMUP_RETRY_DELAYS_S[attempt])
        if proc is not None and proc.returncode != 0:
            error_text = f"{proc.stderr or ''}\n{proc.stdout or ''}".strip()
            raise RuntimeError(f"warmup failed: {cmd[:120]} :: {error_text[:300]}")


def _retryable_grade(payload: dict[str, Any]) -> bool:
    """Their transient grade failure test over the verdict payload."""
    if not isinstance(payload, dict):
        return False
    texts = [
        str(payload.get("error") or ""),
        str(payload.get("judge_error") or ""),
        str(payload.get("llm_judge_error") or ""),
    ]
    if str(payload.get("judge_method") or "").strip().lower() == "failed":
        texts.append("failed")
    combined = "\n".join(texts).upper()
    if any(token in combined for token in _RETRYABLE_GRADE_TOKENS):
        return True
    return bool(re.search(r"\bERROR CODE:\s*(429|502|503|504)\b", combined))


def _grade(container: str, task: dict[str, Any], workspace: Path, output_dir: Path) -> dict[str, Any]:
    """Their grade loop: refresh gt, retry transients, dict only parse."""
    try:
        if (workspace / "gt").is_dir():
            _sh(container, "rm -rf /tmp_workspace/gt && mkdir -p /tmp_workspace/gt", timeout=60)
            subprocess.run(
                ["docker", "cp", str(workspace / "gt") + "/.", f"{container}:/tmp_workspace/gt/"],
                capture_output=True,
                timeout=120,
            )
        if not task["checks"].strip():
            return {}
        # Their settle step: give container writes a second to land before grading.
        _sh(container, "sync; sleep 1; ls -lah /tmp_workspace/results >/dev/null 2>&1 || true", timeout=30)
    except subprocess.TimeoutExpired:
        # A wedged exec before grading fails the task, never the campaign.
        return {"error": "container exec wedged during grade setup"}
    grade_src = (
        "import json, sys\n"
        + task["checks"]
        + "\nresult = grade(transcript=[], workspace_path='/tmp_workspace')\nprint(json.dumps(result))\n"
    )
    (output_dir / "_grade.py").write_text(grade_src, encoding="utf-8")
    try:
        subprocess.run(
            ["docker", "cp", str(output_dir / "_grade.py"), f"{container}:/tmp/_grade.py"],
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {"error": "container exec wedged during grade setup"}
    for attempt in range(len(GRADE_RETRY_DELAYS_S) + 1):
        # Declared robustness deviation: their unguarded timeout lets a grade
        # hang raise and kill the whole campaign. A wedged docker exec killed
        # two multi hour runs here, so the hang degrades to a task level
        # failure with one retry. The timed out exec client leaves the grader
        # alive inside the container, so it is killed before the retry.
        try:
            graded = _sh(container, "python3 /tmp/_grade.py", timeout=GRADE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            if attempt == 0:
                try:
                    _sh(container, "pkill -f /tmp/_grade.py || true", timeout=30)
                except subprocess.TimeoutExpired:
                    return {"error": f"grader timed out after {GRADE_TIMEOUT_S}s; exec wedged"}
                time.sleep(GRADE_RETRY_DELAYS_S[0])
                continue
            return {"error": f"grader timed out after {GRADE_TIMEOUT_S}s"}
        if graded.returncode != 0:
            verdict: dict[str, Any] = {"error": graded.stderr.strip() or "grader failed"}
        else:
            verdict = {"error": "failed to parse grading JSON"}
            for line in reversed(graded.stdout.splitlines()):
                raw = line.strip()
                if not raw.startswith("{"):
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    verdict = parsed
                    break
        if attempt < len(GRADE_RETRY_DELAYS_S) and _retryable_grade(verdict):
            time.sleep(GRADE_RETRY_DELAYS_S[attempt])
            continue
        return verdict
    return verdict


def _container_env(task: dict[str, Any], *, brave_key: str, judge_key: str) -> dict[str, str]:
    """Their container environment: empty proxies, the keys, the task's own Env lines."""
    env = {
        # Empty overrides remove the baked-in lab proxy setting.
        "http_proxy": "",
        "https_proxy": "",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ERROR_RATE": "0",
        "BRAVE_API_KEY": brave_key,
        "OPENROUTER_API_KEY": judge_key,
        "OPENROUTER_BASE_URL": OPENROUTER_BASE,
        "JUDGE_MODEL": JUDGE_MODEL,
    }
    for line in task["env"].splitlines():
        entry = line.strip()
        # Bare keys are all injected above already.
        if entry and not entry.startswith("#") and "=" in entry:
            name, value = entry.split("=", 1)
            env[name] = value
    return env


def _start_container(
    container: str, *, task: dict[str, Any], image: str, workspace: Path, brave_key: str, judge_key: str
) -> None:
    env_args = [
        arg
        for name, value in _container_env(task, brave_key=brave_key, judge_key=judge_key).items()
        for arg in ("-e", f"{name}={value}")
    ]
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container,
            "--add-host",
            "host.docker.internal:host-gateway",
            *env_args,
            "-v",
            f"{workspace}:/app:ro",
            image,
            "bash",
            "-c",
            "tail -f /dev/null",
        ],
        check=True,
        capture_output=True,
    )


def _configure_runtime(container: str, *, agent_base_url: str, key: str) -> None:
    """Their model injection, model binding, and runtime configuration."""
    _inject_models(container, base_url=agent_base_url, key=key)
    # Their runtime configuration, verbatim.
    proc = _sh(
        container,
        " && ".join(
            [
                "openclaw config set tools.profile full",
                "openclaw config set agents.defaults.sandbox.mode off",
                "openclaw config set tools.exec.host gateway",
                "openclaw config set tools.exec.ask off",
                "openclaw config set tools.exec.security full",
                "openclaw config set agents.defaults.thinkingDefault off",
                "openclaw config set browser.defaultProfile openclaw",
                "openclaw config set browser.headless true",
                "openclaw config set browser.noSandbox true",
                "openclaw config set browser.remoteCdpTimeoutMs 10000",
                "openclaw config set browser.remoteCdpHandshakeTimeoutMs 30000",
                "PW_CHROME=\"$(find /root/.cache/ms-playwright -path '*/chrome-linux64/chrome' -type f 2>/dev/null | head -n 1)\"; "
                'if [ -n "$PW_CHROME" ]; then openclaw config set browser.executablePath "$PW_CHROME"; fi',
            ]
        ),
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"runtime configuration failed: {proc.stderr.strip()[:400]}")
    proc = _sh(container, f"openclaw models set 'custom-gateway/{AGENT_MODEL}'", timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"model binding failed: {proc.stderr.strip()[:400]}")
    # Their post binding steps, best effort as theirs are. The OpenRouter
    # profile is the benchmark's own instrumentation, so it carries the judge
    # key; the model under test is only reachable through the gateway above.
    auth_profile = json.dumps(
        {
            "version": 1,
            "profiles": {"openrouter:default": {"type": "api_key", "provider": "openrouter", "key": read_judge_key()}},
        }
    )
    _sh(
        container,
        "mkdir -p /root/.openclaw/agents/main/agent && "
        f"cat > /root/.openclaw/agents/main/agent/auth-profiles.json <<'EOF'\n{auth_profile}\nEOF",
        timeout=30,
    )
    _sh(
        container,
        f"openclaw config set agents.defaults.imageModel.primary 'custom-gateway/{AGENT_MODEL}'",
        timeout=60,
    )


def _score_from_breakdown(breakdown: dict[str, Any]) -> float | None:
    """Their score rule: error voids, overall wins, else numeric mean, else None."""
    if not isinstance(breakdown, dict) or not breakdown or "error" in breakdown:
        return None
    overall = breakdown.get("overall_score")
    if isinstance(overall, (int, float)):
        return float(overall)
    numeric = [float(v) for v in breakdown.values() if isinstance(v, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def run_task(
    *,
    task_file: Path,
    pool: Path,
    image: str,
    brave_key: str,
    output_dir: Path,
    agent_base_url: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # The task parse lives inside the error containment: an unreadable or
    # malformed task file is that task's own error verdict, never an
    # exception that kills the day's worker pool.
    task: dict[str, Any] | None
    parse_error = ""
    try:
        task = parse_task(task_file)
    except Exception as exc:
        task = None
        parse_error = f"task file unreadable: {exc}"
    task_id = task["task_id"] if task else task_file.stem
    prompt = task["prompt"] if task else f"[unreadable task] {task_file.name}"
    stored = output_dir / "score.json"
    if stored.exists():
        # A completed task is never re bought; the stored verdict is final.
        prior = json.loads(stored.read_text())
        return {
            "task_id": task_id,
            "prompt": prompt,
            "score": float(prior["score"]) if isinstance(prior.get("score"), (int, float)) else None,
            "breakdown": prior.get("breakdown") or {},
            "error": str(prior.get("error") or ""),
            "chat": output_dir / "chat.jsonl",
        }
    if task is None:
        (output_dir / "score.json").write_text(
            json.dumps({"score": None, "breakdown": {"error": parse_error}, "error": parse_error}) + "\n"
        )
        return {
            "task_id": task_id,
            "prompt": prompt,
            "score": None,
            "breakdown": {"error": parse_error},
            "error": parse_error,
            "chat": output_dir / "chat.jsonl",
        }
    container = f"sc-{task_id[:40]}-{uuid.uuid4().hex[:6]}"
    workspace = WILDCLAWBENCH_ROOT / task["workspace"]
    score: float | None = None
    breakdown: dict[str, Any] = {}
    error = ""
    try:
        _start_container(
            container, task=task, image=image, workspace=workspace, brave_key=brave_key, judge_key=read_judge_key()
        )
        # Provisioning failures are fatal, as theirs are: no zeros from a broken box.
        _setup_workspace(container)
        subprocess.run(
            ["docker", "cp", str(pool / "skills") + "/.", f"{container}:/root/skills/"], capture_output=True
        )
        # Their order: warmup right after setup, before models and config.
        _run_warmup(container, task["warmup"], output_dir)
        _configure_runtime(container, agent_base_url=agent_base_url, key=GATEWAY_KEY)
        # The container's OpenRouter credential is the benchmark's own (judge)
        # key; the model under test is reachable only through the gateway.
        _sh(
            container,
            f"nohup bash -c 'export OPENROUTER_API_KEY={read_judge_key()} && openclaw gateway --port {GATEWAY_PORT}' > /tmp/gateway.log 2>&1 & sleep 2",
            timeout=60,
        )
        # The escaped text is for the shell line only: the returned prompt
        # must stay the task's raw prompt, the discriminator the driver
        # matches against recorded wire content.
        shell_prompt = compose_prompt(task).replace("'", "'\\''")
        try:
            # Their executor waits exactly the task timeout.
            _sh(
                container,
                f"cd /tmp_workspace && openclaw agent --session-id chat --timeout {task['timeout_seconds']} --message '{shell_prompt}' > /tmp/agent.log 2>&1",
                timeout=task["timeout_seconds"],
            )
        except subprocess.TimeoutExpired:
            error = f"agent timed out after {task['timeout_seconds']}s"
        breakdown = {"error": error} if error else _grade(container, task, workspace, output_dir)
        # The transcript is collected right after grading, where theirs is.
        subprocess.run(
            [
                "docker",
                "cp",
                f"{container}:/root/.openclaw/agents/main/sessions/chat.jsonl",
                str(output_dir / "chat.jsonl"),
            ],
            capture_output=True,
        )
        for log in ("agent.log", "gateway.log"):
            subprocess.run(["docker", "cp", f"{container}:/tmp/{log}", str(output_dir / log)], capture_output=True)
        score = _score_from_breakdown(breakdown)
    except subprocess.TimeoutExpired:
        # A grade hang crashes the run, as theirs does; a rerun redoes the task.
        raise
    except Exception as exc:
        # Keep the untruncated error in the verdict, as their record does.
        error = error or str(exc)
        if not breakdown:
            breakdown = {"error": error}
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    (output_dir / "score.json").write_text(json.dumps({"score": score, "breakdown": breakdown, "error": error}) + "\n")
    return {
        "task_id": task_id,
        "prompt": prompt,
        "score": score,
        "breakdown": breakdown,
        "error": error,
        "chat": output_dir / "chat.jsonl",
    }
