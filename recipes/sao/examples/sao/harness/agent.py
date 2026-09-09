"""Harbor agent that drives several scored rollouts of one IMO problem through Reef.

The problem arrives as the trial's ``instruction``. The agent looks up its gold
answer and runs ``ROLLOUTS`` rollouts through Reef's ``/v1/chat/completions``,
scoring each \\boxed{} answer against the gold with the strict equivalence rule
the Harbor verifier uses and reporting the binary reward (1.0 correct, 0.0 wrong)
against that rollout's receipt — one candidate training step per rollout. The last
completion is written to ``/workspace/answer.txt`` for the Harbor verifier, and a
watcher thread posts the verifier's own reward to Reef once Harbor ends the trial.
"""

import asyncio
import atexit
import json
import os
import re
import shlex
import threading
import time
import urllib.request

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef_client import ReefClient

from .report import post_report

#: The Reef ``run.sh`` starts from ``serve.yaml``.
SERVICE_URL = "http://127.0.0.1:8900"
TOKEN = "reef-local"
#: This workload's isolated lane, and the recipe that serves it.
SCENARIO = "sao-smoke"
RECIPE = "sao"
#: Scored rollouts per task, and the generation window each one gets.
ROLLOUTS = int(os.environ.get("SAO_ROLLOUTS", "6"))
MAX_TOKENS = 2048
#: Where the agent writes the last completion for the Harbor verifier.
ANSWER_PATH = "/workspace/answer.txt"

#: Per-rollout raw records land here so the learning curve is auditable.
#: Override for a multi-arm study; the driver picks the file.
RECORDS_PATH = os.environ.get("SAO_RECORDS_PATH", "work/records/rollouts.jsonl")
#: Run and arm identity so records from different runs concatenate cleanly.
RUN_ID = os.environ.get("SAO_RUN_ID", "sao-smoke")
ARM = os.environ.get("SAO_ARM", "sao")

#: Gold answers, keyed by a prefix that identifies one of the three IMO
#: problems, exactly as they appear in the IMOAnswerBench dump.
GOLD_ANSWERS = {
    "Let $u \\ge 2$": r"$2^{u-2}$",
    "Let $x_0, x_1": r"$-\frac{2023}{2024^2}$",
    "For a real number $T$": r"$\frac{1}{2}$",
}


# Answer extraction and strict equivalence: the same rule the Harbor verifier
# applies (harbor/imo-*/tests/grade.py), copied rather than imported because
# the harness is installed on its own into reef-eval's environment.


def _latex_to_float(expression: str) -> float | None:
    """Numerically evaluate simple competition-answer LaTeX (frac/sqrt/pi)."""
    text = expression.strip().strip("$").replace(" ", "").replace("\\left", "").replace("\\right", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*").replace("\\!", "").replace("\\,", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\pi", "P")
    for _ in range(20):
        new = re.sub(r"\\sqrt\{([^{}]*)\}", r"s(\1)", text)
        new = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", new)
        if new == text:
            break
        text = new
    if "\\" in text or "{" in text or "}" in text:
        return None
    for _ in range(4):
        text = re.sub(r"([\dP)])\s*([Ps(])", r"\1*\2", text)
    try:
        value = eval(text, {"__builtins__": {}}, {"P": 3.141592653589793, "s": lambda v: v**0.5})
        return float(value)
    except Exception:
        return None


def answers_equal(gold: str, predicted: str | None) -> bool:
    if predicted is None:
        return False
    gold_text = str(gold).strip().strip("$").rstrip(".").replace(" ", "")
    predicted_text = predicted.strip().strip("$").rstrip(".").replace(" ", "")
    if predicted_text == gold_text:
        return True
    gold_value = _latex_to_float(gold_text)
    predicted_value = _latex_to_float(predicted_text)
    if gold_value is None or predicted_value is None:
        return False
    return abs(predicted_value - gold_value) <= 1e-6 * max(1.0, abs(gold_value))


def extract_answer(text: str) -> str | None:
    """Prefer the last \\boxed{...}; fall back to the last standalone integer."""
    starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
    for start in reversed(starts):
        depth, i = 1, start
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            candidate = text[start : i - 1].strip()
            if candidate:
                return candidate
    tail_ints = re.findall(r"(?<![\d.])(\d+)(?![\d.])", text[-400:])
    return tail_ints[-1] if tail_ints else None


class HarborAgent(BaseAgent):
    """One Harbor trial — one IMO problem — as ``ROLLOUTS`` scored rollouts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=1800)  # long IMO generations
        # Harbor runs the verifier after run() returns and ends the trial by
        # writing result.json; watch for it from construction, so the verifier
        # reward still reaches Reef when the rollouts themselves failed.
        self._report_watch_from = time.time()
        self._reporter = threading.Thread(target=self._report_trial_result, daemon=True)
        self._reporter.start()
        atexit.register(self._reporter.join, 10.0)  # don't drop an in-flight report

    @staticmethod
    def name() -> str:
        return "reef-sao"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the agent logic runs on the host."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        gold = next(answer for prefix, answer in GOLD_ANSWERS.items() if instruction.startswith(prefix))

        agent_record_ids = []
        task_name = getattr(getattr(context, "task", None), "name", "unknown")
        for index in range(ROLLOUTS):
            serving_before = _current_serving_release()
            response, agent_record_id = await asyncio.to_thread(self._ask_reef, instruction)
            completion = response["choices"][0]["message"]["content"]
            predicted = extract_answer(completion)
            score = 1.0 if answers_equal(gold, predicted) else 0.0
            self._client.report(SCENARIO, {"score": score, "references": [agent_record_id]})
            agent_record_ids.append(agent_record_id)
            _append_rollout_record(
                run_id=RUN_ID,
                arm=ARM,
                task_name=task_name,
                rollout_in_task=index,
                serving_release_id=serving_before,
                agent_record_id=agent_record_id,
                prompt_tokens=(response.get("usage") or {}).get("prompt_tokens"),
                completion_tokens=(response.get("usage") or {}).get("completion_tokens"),
                score=score,
                predicted=predicted,
                gold=gold,
            )
            print(f"[{RECIPE} {index}] score={score:.1f} predicted={predicted!r}", flush=True)

        # The last completion is the one the Harbor verifier scores.
        result = await environment.exec(f"printf %s {shlex.quote(completion)} > {ANSWER_PATH}")
        if result.return_code != 0:
            raise RuntimeError(f"writing {ANSWER_PATH} failed: {result.stderr}")

        context.metadata = {**(context.metadata or {}), "reef": {"agent_record_ids": agent_record_ids}}
        context.n_input_tokens = response["usage"]["prompt_tokens"]
        context.n_output_tokens = response["usage"]["completion_tokens"]

    def _ask_reef(self, instruction: str) -> tuple[dict, str]:
        return self._client.inference_with_record(
            SCENARIO,
            "/v1/chat/completions",
            {
                "model": self.model_name,
                "messages": [{"role": "user", "content": instruction}],
                "max_tokens": MAX_TOKENS,
                "temperature": 1.0,
                "top_p": 1.0,
            },
        )

    def _report_trial_result(self) -> None:
        """Post the verifier reward once Harbor writes result.json in the trial directory."""
        result_path = self.logs_dir.parent / "result.json"
        while not (result_path.exists() and result_path.stat().st_mtime >= self._report_watch_from):
            time.sleep(1.0)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        post_report(result, client=self._client, scenario=SCENARIO)
        self.logger.info("reported verifier reward %s to reef", result["verifier_result"]["rewards"]["reward"])


def _current_serving_release() -> str | None:
    """Best-effort read of the scenario's current release id; None on error.

    The learning curve records which weight version served each rollout so a
    plot can mark the transitions instead of averaging across them.
    """
    request = urllib.request.Request(
        f"{SERVICE_URL}/reef/scenarios/{SCENARIO}/releases",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
    except (OSError, TimeoutError, json.JSONDecodeError):
        return None
    for row in payload.get("releases", []):
        if row.get("current"):
            return row.get("release_id")
    return None


_RECORDS_LOCK = threading.Lock()


def _append_rollout_record(**fields) -> None:
    """Append one rollout's raw record to the retained JSONL.

    The plot's inputs are these lines, one per scored rollout, so a reviewer
    can audit the figure without trusting a summary.
    """
    fields["recorded_at"] = time.time()
    path = os.path.abspath(RECORDS_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(fields, ensure_ascii=False) + "\n"
    with _RECORDS_LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(line)
