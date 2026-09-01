"""Campaign driver: seven days, six nights, one run per REEF_SC_RUN, on Reef.

Ported from benchmarks/skill_claw/__main__.py at commit 0519eefb (branch
bo/skillclaw-paper), adapted to the harness evolution mechanism. Day one is
the baseline; each night evolves the pool from that day and the next day
measures it, so the last pool never goes unmeasured. The run owns its Reef
service in process: task containers reach it through Docker's host gateway,
every ``/v1`` request they make carries the served pool's catalog
(``SkillClawRecipe``'s surface) and is recorded, and the day's last report
completes the trace batch that schedules the night step. The driver waits for
that background step, then pulls the published pool from ``GET /reef/harness``,
materializes it for the next day's containers, and seals the round. The frozen
control run serves the same way but reports nothing, so nothing batches, no
night runs, and its pool never changes.

A grader failure reports the unscored sentinel (-1.0; 0.0 would be a real
failing grade) and stays out of the category means; the night's judge
backfills the score. A task with no traffic records the fact so the day
still counts every task. Rounds checkpoint as round-<n>/summary.json and a
rerun resumes after the last one: a completed task's stored verdict is never
re-bought, a replayed report that the store already holds is skipped on its
409, a night whose trigger report landed before a crash is poked by hand,
and the served pool survives restarts as pool-current with the commit log as
the authority (_committed_pool).

``python3 run.py report`` prints the gain table over both runs' sealed
rounds (``stats.py``, the preregistered criterion).
``PYTHONPATH=../.. python3 run.py solve`` runs the bundled ``harbor/`` task
once through reef-eval instead - the sibling examples' smoke idiom - against
the same embedded service. The repository root on ``PYTHONPATH`` exposes the
cookbook recipe's canonical ``recipes.skillclaw.recipe`` module.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from aiohttp import web
from reef_client import ReefClient, ReefClientError

from harness import day, stats
from harness.config import (
    AGENT_MODEL,
    DATA_DIR,
    HERE,
    NIGHT_TIMEOUT_S,
    NIGHTS,
    PAPER_DAY_ONE,
    REEF_SC_PORT,
    REGIME_TOLERANCE,
    RUN,
    SUCCESS_THRESHOLD,
    UNSCORED_SENTINEL,
    UPSTREAM_BASE,
    USERS,
    WORKDIR,
    brave_key,
    read_key,
)
from harness.sessions import slugify
from reef.artifact.memory import InMemoryRepositoryBackend
from reef.core.records_types import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.harness.adapters import get_adapter
from reef.harness.render import render_composition
from reef.recipe.registry import RecipeRegistry, build_recipe
from reef.records import RecordStore
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.inference import HttpInferenceBackend, provider_request_headers
from reef.service.app import create_app
from reef.service.deploy.config import load_config
from reef.service.wire import SCENARIO_HEADER


class Ledger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
        with self._lock, self._path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")


def load_recipe() -> tuple[str, Any]:
    """The recipe skillclaw.yaml declares, built through its explicit implementation.

    This driver is the deployment: it owns the upstream binding and hands it
    to the recipe as its runtime, the way ``reef.upstream_url`` does in a
    served deployment. The method (harness/) never sees the endpoint.
    """
    config = load_config(HERE / "skillclaw.yaml")
    sections = {key: config[key] for key in ("implementation", "model", "evolution", "data")}
    runtime = InferenceProxyRuntime(
        model_path=AGENT_MODEL,
        base_url=UPSTREAM_BASE,
        api_key=read_key(),
        inference_timeout_s=600.0,
    )
    recipe = build_recipe(str(sections["implementation"]), config=sections, runtime=runtime)
    return recipe.name, recipe


def _default_headers_middleware(scenario: str) -> Any:
    """Bare-client requests ride the run's scenario.

    Task containers send no reef headers (their agent only knows an
    OpenAI-compatible base url), so the missing scenario header is stamped
    in before routing - the embedded equivalent of the sealed service's
    default scenario. The service serves one recipe, so nothing else is needed.
    """

    @web.middleware
    async def stamp(request: web.Request, handler: Any) -> web.StreamResponse:
        if SCENARIO_HEADER not in request.headers:
            request = request.clone(headers={**request.headers, SCENARIO_HEADER: scenario})
        return await handler(request)

    return stamp


class RunService:
    """The in-process Reef service one campaign run owns."""

    def __init__(
        self,
        *,
        scenario: str,
        recipe_name: str,
        recipe: Any,
        bootstrap_pool: Path,
        run_dir: Path,
        upstream_url: str,
        upstream_key: str,
        port: int,
        inference_backend: Any | None = None,
    ) -> None:
        self.scenario = scenario
        self.recipe_name = recipe_name
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.dispatcher = Dispatcher(
            RecipeRegistry({recipe_name: recipe}),
            InMemoryRepositoryBackend.factory(bootstrap_pool, root=run_dir / "artifacts"),
            local_artifact_dir=run_dir / "staged",
            agent_record_dir=run_dir / "reef-data",
        )
        self._app = create_app(
            self.dispatcher,
            inference_backend=inference_backend
            or HttpInferenceBackend(
                upstream_url,
                request_headers=provider_request_headers(upstream_key),
                timeout_s=600.0,
            ),
        )
        self._app.middlewares.append(_default_headers_middleware(scenario))
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None

    def start(self) -> None:
        # Scenario creation (or durable recovery from the commit log) happens
        # before any request lands.
        self.dispatcher.get_or_create_scenario(self.scenario, self.recipe_name)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="reef-embedded", daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._serve(), self._loop).result(timeout=60)

    async def _serve(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        # Containers reach us via Docker's host gateway, so serve on all
        # interfaces. Port 0 (tests) binds an ephemeral port, read back below.
        await web.TCPSite(self._runner, host="0.0.0.0", port=self.port).start()
        if not self.port:
            self.port = int(self._runner.addresses[0][1])
            self.base_url = f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        if self._loop is not None:
            if self._runner is not None:
                asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result(timeout=60)
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=60)
        self.dispatcher.close()

    @property
    def container_base_url(self) -> str:
        """The service as a task container sees it, in provider baseUrl form."""
        return f"http://host.docker.internal:{self.port}/v1"

    def current_pool(self) -> Path:
        """The served artifact tree on disk (the rendered composition)."""
        artifact = self.dispatcher.current_artifact(self.scenario).materialize()
        assert artifact.local_path is not None
        return artifact.local_path

    def records(self) -> RecordStore:
        scenario = self.dispatcher.get_or_create_scenario(self.scenario, self.recipe_name)
        assert scenario is not None
        return scenario.records

    def training_versions(self) -> int:
        """Committed pool versions beyond the bootstrap, the night counter.

        Every completed step appends a training commit row - a skipped night
        included - so the counter is the distinct releases those
        rows introduced, not the row count.
        """
        versions = self.dispatcher.list_releases(self.scenario)
        baseline = {row["release_id"] for row in versions if row.get("operation") != "training"}
        trained = {row["release_id"] for row in versions if row.get("operation") == "training"}
        return len(trained - baseline)

    def training_step(self) -> int:
        current = self.dispatcher.get_or_create_scenario(self.scenario, self.recipe_name)
        assert current is not None
        return current.scenario_step

    def wait_for_training_step(self, after: int) -> None:
        deadline = time.monotonic() + NIGHT_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.training_step() > after:
                return
            if error := self.dispatcher.training_status["error"]:
                raise RuntimeError(f"night training failed: {error}")
            time.sleep(1.0)
        raise TimeoutError(f"night training did not advance past step {after}")

    def poke_night(self) -> bool:
        """Crash recovery: run a pending night whose trigger report already landed.

        If the process died after the trigger report but before its commit, no
        further record will re-trigger it, so a resumed run drives one step.
        """
        current = self.dispatcher.get_or_create_scenario(self.scenario, self.recipe_name)
        assert current is not None
        result = current.prepare_training_step()
        if result is None:
            return False
        current.commit(result)
        return True


def ensure_task_record(
    service: RunService,
    *,
    task_prompt: str,
    composed_prompt: str,
    agent_record_id: str,
    after_sequence: int = 0,
) -> str:
    """The record a task's report references, recording silence when needed.

    A task whose container never reached the model (a warmup or setup
    failure) has no traffic, but the day still has 60 tasks and the night
    trigger counts reports. Recording the composed prompt with an empty
    response states the fact and gives the report a real reference; the
    digest it produces is an empty session, which the night's judge and
    no-skill bucket already handle.
    """
    found = find_task_record(service.records(), service.scenario, task_prompt, after_sequence=after_sequence)
    if found is not None:
        return found.agent_record_id
    item = AgentRecord.create(
        scenario=service.scenario,
        request_type=RequestType.INFERENCE,
        payload={"messages": [{"role": "user", "content": composed_prompt}], "response": {}},
        agent_record_id=agent_record_id,
    )
    service.dispatcher.accept_record(item, recipe=service.recipe_name)
    return item.agent_record_id


def find_task_record(records: RecordStore, scenario: str, task_prompt: str, *, after_sequence: int = 0) -> Any | None:
    """The task's latest recorded request past the floor: its session's state.

    Concurrent containers share one scenario and a bare client sends no
    session id, so the task's raw prompt is the discriminator. The floor is
    the day's start: compaction keeps only each session's referenced final
    request, so without it a task with no traffic today would match its own
    leftover records from an earlier round.
    """
    latest = None
    cursor = after_sequence
    while True:
        page = records.replay_page(scenario, after_sequence=cursor, limit=256)
        if not page:
            return latest
        cursor = page[-1][0]
        for _, item in page:
            if item.request_type is not RequestType.INFERENCE:
                continue
            messages = item.payload.get("messages")
            if not isinstance(messages, list):
                continue
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    if task_prompt in _user_text(message.get("content")):
                        latest = item
                    break


def _user_text(content: Any) -> str:
    """A user message's text as the substring target for prompt matching.

    OpenClaw clients send content as a list of typed parts; ``str(list)``
    would repr-escape newlines and no multiline prompt could ever match, so
    the text parts are joined as written.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content) if content is not None else ""


def _max_sequence(records: RecordStore, scenario: str) -> int:
    last = 0
    while True:
        page = records.replay_page(scenario, after_sequence=last, limit=256)
        if not page:
            return last
        last = page[-1][0]


def _tasks() -> list[tuple[str, dict[str, Any]]]:
    return [
        (category, entry)
        for category, entries in json.loads((DATA_DIR / "tasks.json").read_text())["tasks"].items()
        for entry in entries
    ]


def pull_pool(client: ReefClient, service: RunService, target: Path) -> dict[str, Any]:
    """The pull path: materialize the served tree from ``GET /reef/harness``."""
    manifest = client.get("/reef/harness", extra_headers={SCENARIO_HEADER: service.scenario})
    if target.exists():
        shutil.rmtree(target)
    for relative, text in manifest["files"].items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return manifest


def run_day(
    service: RunService,
    client: ReefClient,
    round_dir: Path,
    round_index: int,
    ledger: Ledger,
) -> list[dict[str, Any]]:
    key = brave_key()
    pool = round_dir / "pool"
    pull_pool(client, service, pool)
    # Records at or below this sequence are earlier rounds' leftovers; a task
    # with no traffic today must not match its own past session.
    day_floor = _max_sequence(service.records(), service.scenario)

    def one(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        category, entry = item
        task_file = day.WILDCLAWBENCH_ROOT / str(entry["file"])
        result = day.run_task(
            task_file=task_file,
            pool=pool / "pi-agent",
            image=day.image_tag(),
            brave_key=key,
            output_dir=round_dir / str(entry["id"]),
            agent_base_url=service.container_base_url,
        )
        task_id = str(result["task_id"])
        prompt = str(result["prompt"])
        slug = slugify(task_id)
        try:
            composed = day.compose_prompt(day.parse_task(task_file))
        except Exception:
            composed = prompt
        reference = ensure_task_record(
            service,
            task_prompt=prompt,
            composed_prompt=composed,
            agent_record_id=f"sc-{RUN}-empty-r{round_index}-{slug}",
            after_sequence=day_floor,
        )
        score = result["score"]
        scored = isinstance(score, (int, float))
        # Their success rule: a clean run and a scored verdict over the threshold.
        success = not result["error"] and scored and float(score) >= SUCCESS_THRESHOLD
        meta = {
            "reference": reference,
            "task_id": task_id,
            "prompt": prompt,
            "category": category,
            "round": round_index,
            "score": float(score) if scored else None,
            "success": success,
            "error": str(result["error"]),
            "breakdown": result["breakdown"],
        }
        # The day ledger the night's digests read (TraceSample carries no
        # metadata channel); written before the report so the trigger report
        # never races its own night.
        report_path = round_dir / "reports" / f"{slug}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
        if RUN != "frozen":
            report(
                service,
                client,
                {
                    "agent_record_id": f"sc-{RUN}-report-r{round_index}-{slug}",
                    "score": float(score) if scored else UNSCORED_SENTINEL,
                    "references": [reference],
                },
            )
        ledger.write({"event": "sc_task", "task": task_id, "score": score})
        return {
            "category": category,
            "task_id": task_id,
            "score": float(score) if scored else None,
            "success": success,
            "error": str(result["error"]),
        }

    with ThreadPoolExecutor(max_workers=USERS) as workers:
        return list(workers.map(one, _tasks()))


def report(service: RunService, client: ReefClient, payload: dict[str, Any]) -> None:
    """Report a task; survive a replayed day and a client timeout.

    A rerun replays completed tasks from their stored verdicts, so the same
    agent_record_id reports again - the store answers 409 for one it already
    holds with different content, and the task is simply already reported.
    A client timeout can race the server-side append, so the durable record is
    checked to determine whether the report landed.
    """
    try:
        client.report(service.scenario, dict(payload), recipe=service.recipe_name)
    except ReefClientError as exc:
        if exc.status == 409:
            return  # an already reported task from a replayed day
        _settle(service)
        if service.records().get(service.scenario, str(payload["agent_record_id"])) is None:
            raise
    except Exception:
        _settle(service)
        if service.records().get(service.scenario, str(payload["agent_record_id"])) is None:
            raise


def _settle(service: RunService) -> None:
    with contextlib.suppress(Exception):
        service.training_versions()


def category_scores(results: list[dict[str, Any]]) -> dict[str, float]:
    """Their safe rate: means over scored tasks only, never a fake zero."""
    totals: dict[str, list[float]] = {}
    for result in results:
        score = result.get("score")
        if isinstance(score, (int, float)):
            totals.setdefault(str(result["category"]), []).append(float(score))
    return {name: round(100.0 * sum(vals) / len(vals), 2) for name, vals in sorted(totals.items())}


def regime_verdict(scores: dict[str, float]) -> dict[str, Any]:
    gaps = {name: round(scores.get(name, 0.0) - reference, 2) for name, reference in PAPER_DAY_ONE.items()}
    return {"gaps": gaps, "in_regime": all(abs(gap) <= REGIME_TOLERANCE for gap in gaps.values())}


def _persist_pool(service: RunService, run_dir: Path) -> None:
    """Keep the served pool on disk so a restarted run reseeds from it."""
    staging = run_dir / "pool-current.next"
    retired = run_dir / "pool-current.old"
    target = run_dir / "pool-current"
    for leftover in (staging, retired):
        if leftover.exists():
            shutil.rmtree(leftover)
    shutil.copytree(service.current_pool(), staging)
    # Rename twice instead of delete then rename: a crash in the gap leaves
    # a complete pool under one of the names, never a missing one.
    if target.exists():
        target.rename(retired)
    staging.rename(target)
    if retired.exists():
        shutil.rmtree(retired)


def _committed_pool(run_dir: Path, scenario: str) -> Path | None:
    """The pool tree of the scenario's last durable commit, if still on disk.

    The commit log is the authority on which night landed; pool-current is
    written after the commit, so a crash between them would otherwise revert
    the served pool while the summaries claim the night happened. Version
    trees under run_dir/artifacts survive restarts even though the in-memory
    repository forgets them.
    """
    key = hashlib.sha256(scenario.encode("utf-8")).hexdigest()
    log = run_dir / "reef-data" / f"{key}.commits.jsonl"
    if not log.exists():
        return None
    version = None
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        ref = record.get("artifact_ref")
        if isinstance(ref, dict) and ref.get("version"):
            version = str(ref["version"])
    if version is None:
        return None
    candidate = run_dir / "artifacts" / version
    return candidate if (candidate / "pi-agent" / "skills").is_dir() else None


def _bootstrap_pool(run_dir: Path, scenario: str, recipe: Any) -> Path:
    """The tree the repository boots from: last commit, else the persisted
    pool, else the seed composition rendered fresh."""
    committed = _committed_pool(run_dir, scenario)
    if committed is not None:
        return committed
    persisted = run_dir / "pool-current"
    if persisted.is_dir():
        return persisted
    rendered = run_dir / "bootstrap"
    if rendered.exists():
        shutil.rmtree(rendered)
    nodes = tuple((str(entry["name"]), entry.get("config")) for entry in recipe.seed if not entry.get("disabled"))
    for relative, text in render_composition(nodes, get_adapter(recipe.adapter)).items():
        path = rendered / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return rendered


def main() -> None:
    run_dir = WORKDIR / RUN
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(run_dir / "ledger.jsonl")
    day.ensure_benchmark()
    recipe_name, recipe = load_recipe()
    service = RunService(
        scenario=RUN,
        recipe_name=recipe_name,
        recipe=recipe,
        # The last durable commit wins over pool-current: a crash between the
        # night's commit and the pool copy must not silently revert the pool.
        bootstrap_pool=_bootstrap_pool(run_dir, RUN, recipe),
        run_dir=run_dir,
        upstream_url=UPSTREAM_BASE,
        upstream_key=read_key(),
        port=REEF_SC_PORT,
    )
    service.start()
    client = ReefClient(service.base_url, timeout_s=NIGHT_TIMEOUT_S)
    try:
        if RUN != "frozen" and service.poke_night():
            # A crash between the day's last report and its commit left a full
            # batch behind; the step just ran, so persist what it published.
            ledger.write({"event": "sc_night_recovered"})
            _persist_pool(service, run_dir)
        for round_index in range(1, NIGHTS + 2):
            round_dir = run_dir / f"round-{round_index}"
            summary_path = round_dir / "summary.json"
            if summary_path.exists():
                continue
            ledger.write({"event": "sc_round_start", "round": round_index, "run": RUN})
            versions_before = service.training_versions()
            step_before = service.training_step()
            results = run_day(service, client, round_dir, round_index, ledger)
            scores = category_scores(results)
            unscored = sum(1 for result in results if result["score"] is None)
            summary: dict[str, Any] = {"round": round_index, "run": RUN, "categories": scores, "unscored": unscored}
            if round_index == 1:
                # Advisory only: one round varies too much to be a gate.
                summary["regime"] = regime_verdict(scores)
                ledger.write({"event": "sc_regime", **summary["regime"]})
            if RUN == "frozen":
                if service.training_versions():
                    raise RuntimeError("the control run changed the pool; its scores would no longer be a reference")
            elif round_index <= NIGHTS:
                service.wait_for_training_step(step_before)
                advanced = service.training_versions() > versions_before
                if advanced:
                    _persist_pool(service, run_dir)
                summary["advanced"] = advanced
                audit_path = round_dir / "night" / "audit.json"
                if audit_path.exists():
                    summary["audit"] = json.loads(audit_path.read_text())["audit"]
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
            ledger.write(
                {
                    "event": "sc_round_done",
                    "round": round_index,
                    "categories": scores,
                    "advanced": summary.get("advanced", False),
                }
            )
        ledger.write({"event": "sc_campaign_done", "run": RUN})
    finally:
        # Never close the store under a night that is still committing. The
        # snapshot is best effort: the commit log is the recovery authority.
        _settle(service)
        with contextlib.suppress(Exception):
            _persist_pool(service, run_dir)
        service.stop()


def solve() -> None:
    """One reef-eval episode on the bundled ``harbor/`` task: the sibling
    examples' smoke idiom (``lab.run(harbor_dir, agent)``), against the same
    embedded service the campaign runs, so the recorded exchange carries the
    served pool's catalog. The task is self-contained (plain base image plus
    the vendored benchmark files), so this works anywhere docker does; the
    campaign's own day loop never goes through reef-eval. A repeated solve with
    unchanged settings returns the stored row without re-running (reef-eval's
    resume key)."""
    from reef_eval import Lab  # solve-only dependency, declared in pyproject.toml

    run_dir = WORKDIR / "solve"
    run_dir.mkdir(parents=True, exist_ok=True)
    recipe_name, recipe = load_recipe()
    service = RunService(
        scenario="solve",
        recipe_name=recipe_name,
        recipe=recipe,
        bootstrap_pool=_bootstrap_pool(run_dir, "solve", recipe),
        run_dir=run_dir,
        upstream_url=UPSTREAM_BASE,
        upstream_key=read_key(),
        port=0,  # ephemeral: the solve agent runs on this host, not in docker
    )
    service.start()
    agent = {
        "name": "harness.harbor_agent:HarborAgent",
        "model_name": AGENT_MODEL,
        "kwargs": {"service_url": service.base_url, "scenario": "solve", "recipe": recipe_name},
    }
    try:
        lab = Lab(run_dir / "lab")
        row = asyncio.run(lab.run(str(HERE / "harbor"), agent))
        print(f"episode reward: {row.rewards}")
    finally:
        service.stop()


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        print(json.dumps(stats.boost_report(WORKDIR / "frozen", WORKDIR / "skillclaw"), indent=2))
    elif sys.argv[1:] == ["solve"]:
        solve()
    else:
        main()
