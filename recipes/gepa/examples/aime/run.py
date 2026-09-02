"""The AIME driver: one embedded Reef service, GEPA's search, two sealed test passes.

The method is the recipe (``gepa.yaml``); this file is only the benchmark
around it. It owns the Reef service in process the way ``recipes/skillclaw``
does, so every problem it solves is real recorded traffic against the served
composition and the method reflects on transcripts it actually produced.

One round is one GEPA step: pull the served tree, run the next epoch-shuffled
minibatch of three training problems as Pi episodes against it with a model
binding pointed back at this service, report each score against the request
that episode recorded, and wait. The third report closes the batch
(``data.batch_size: 3``), which is what schedules the step where the method
proposes, evaluates on the 45 validation problems, and publishes on a strict
mean improvement. Rounds stop at ``--budget`` metric calls; then the seed and
the served compositions are each scored on the sealed 150-problem test split
through per-example checkpoints and ``summary.json`` puts those numbers beside
the retained official record.

The work directory resumes: the archive, the checkpoints, the round counter,
and the scenario's own commit log all survive a restart. ``--dry-run`` boots
the recipe and prints the plan without a model call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aiohttp import web
from harness import aime
from harness.heldout import CheckpointedEvaluator

from reef.artifact.memory import InMemoryRepositoryBackend
from reef.core.records_types import RequestType
from reef.dispatcher import Dispatcher
from reef.harness import render_composition, run_episode
from reef.harness.adapters import get_adapter
from reef.harness.model_binding import ModelBinding
from reef.recipe.registry import RecipeRegistry, build_recipe
from reef.records import RecordStore
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.inference import HttpInferenceBackend, provider_request_headers
from reef.service.app import create_app
from reef.service.deploy.config import load_config
from reef.service.wire import SCENARIO_HEADER

HERE = Path(__file__).resolve().parent
# run.sh exports these; the defaults repeat here so the module imports (and
# the dry run works) without it. gepa.yaml interpolates REEF_WORK, so it has
# to be set before load_recipe reads the file, not before this import block.
os.environ.setdefault("REEF_WORK", str(HERE / "work"))
os.environ.setdefault("REEF_MODEL", "gpt-4.1-mini-2025-04-14")
os.environ.setdefault("REEF_PI_BINARY", "pi")

WORK = Path(os.environ["REEF_WORK"])
SCENARIO = os.environ.get("REEF_GEPA_SCENARIO", "gepa-aime")
# The multi-node variant evolves the rules node and a skill node together;
# the default single-node run is the one the retained record compares against.
MULTI = os.environ.get("REEF_GEPA_MULTI") == "1"
COMPONENTS = ["rules", "skill"] if MULTI else ["rules"]
# The service key the Pi episodes present. The embedded service does not
# authenticate; this must never be the provider key, which would land in a
# rendered models.json inside the episode root.
EMBEDDED_KEY = "reef-embedded"
EPISODE_TIMEOUT_S = 600.0
STEP_TIMEOUT_S = 7200.0
RESULTS = HERE / "results" / "quickstart-seed-0-2026-09-01" / "manifest.json"


def load_recipe(tasks: list[str], *, api_key: str) -> tuple[str, Any]:
    """The recipe gepa.yaml declares, with the validation split filled in.

    This driver is the deployment: it owns the provider endpoint and hands it
    to the recipe as its runtime, the way ``reef.upstream_url`` does in a served
    deployment. The method never sees the endpoint or the key.
    """
    config = load_config(HERE / "gepa.yaml")
    sections = {key: config[key] for key in ("implementation", "model", "evolution", "data")}
    evolution = dict(sections["evolution"])
    evolution["tasks"] = tasks
    if MULTI:
        evolution["seed"] = [
            *evolution["seed"],
            {
                "id": aime.SKILL_SEED_NAME,
                "name": "skill",
                "config": {"name": aime.SKILL_SEED_NAME, "text": aime.SKILL_SEED},
            },
        ]
        evolution["gepa"] = {**evolution["gepa"], "components": COMPONENTS}
    sections["evolution"] = evolution
    runtime = InferenceProxyRuntime(
        model_path=os.environ["REEF_MODEL"],
        base_url=aime.OPENAI_BASE_URL,
        api_key=api_key,
        inference_timeout_s=EPISODE_TIMEOUT_S,
    )
    recipe = build_recipe(str(sections["implementation"]), config=sections, runtime=runtime)
    return recipe.name, recipe


def provider_key() -> str:
    key = os.environ.get(aime.API_KEY_ENV, "").strip()
    if not key:
        raise SystemExit(f"{aime.API_KEY_ENV} is not set; no model calls were made")
    return key


def _default_headers_middleware(scenario: str) -> Any:
    """Pi's requests ride this run's scenario: an episode knows only an
    OpenAI-compatible base url and sends no reef headers, so the missing
    scenario header is stamped in before routing."""

    @web.middleware
    async def stamp(request: web.Request, handler: Any) -> web.StreamResponse:
        if SCENARIO_HEADER not in request.headers:
            request = request.clone(headers={**request.headers, SCENARIO_HEADER: scenario})
        return await handler(request)

    return stamp


class RunService:
    """The in-process Reef service one AIME run owns."""

    def __init__(
        self,
        *,
        scenario: str,
        recipe_name: str,
        recipe: Any,
        bootstrap_tree: Path,
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
            InMemoryRepositoryBackend.factory(bootstrap_tree, root=run_dir / "artifacts"),
            local_artifact_dir=run_dir / "staged",
            agent_record_dir=run_dir / "reef-data",
        )
        self._app = create_app(
            self.dispatcher,
            inference_backend=inference_backend
            or HttpInferenceBackend(
                upstream_url,
                request_headers=provider_request_headers(upstream_key),
                timeout_s=EPISODE_TIMEOUT_S,
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
        # Episodes run on this host. Port 0 (tests) binds an ephemeral port,
        # read back here so the binding can point at it.
        await web.TCPSite(self._runner, host="127.0.0.1", port=self.port).start()
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

    def binding(self) -> ModelBinding:
        """The endpoint an episode reaches this service through."""
        return ModelBinding(self.base_url, os.environ["REEF_MODEL"], api_key=EMBEDDED_KEY, timeout_s=EPISODE_TIMEOUT_S)

    def records(self) -> RecordStore:
        scenario = self.dispatcher.get_or_create_scenario(self.scenario, self.recipe_name)
        assert scenario is not None
        return scenario.records

    def training_step(self) -> int:
        current = self.dispatcher.get_or_create_scenario(self.scenario, self.recipe_name)
        assert current is not None
        return current.scenario_step

    def wait_for_training_step(self, after: int) -> None:
        deadline = time.monotonic() + STEP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.training_step() > after:
                return
            if error := self.dispatcher.training_status["error"]:
                raise RuntimeError(f"the training step failed: {error}")
            time.sleep(1.0)
        raise TimeoutError(f"training did not advance past step {after}")


def find_task_record(records: RecordStore, scenario: str, task_prompt: str, *, after_sequence: int = 0) -> Any | None:
    """The problem's latest recorded request past the floor.

    Episodes share one scenario and send no session id, so the problem statement
    is the discriminator; the floor is the round's start, without which a
    repeated problem would match its own record from an earlier round.
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

    Pi sends content as a list of typed parts unless the quickstart extension
    flattened it; ``str(list)`` would repr-escape newlines and no multiline
    problem could match, so the text parts are joined as written."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content) if content is not None else ""


def max_sequence(records: RecordStore, scenario: str) -> int:
    last = 0
    while True:
        page = records.replay_page(scenario, after_sequence=last, limit=256)
        if not page:
            return last
        last = page[-1][0]


def report(service: RunService, client: Any, payload: dict[str, Any]) -> None:
    """Report one scored problem; a replayed round lands on its 409."""
    from reef_client import ReefClientError

    try:
        client.report(service.scenario, dict(payload), recipe=service.recipe_name)
    except ReefClientError as exc:
        if exc.status != 409:
            raise


def pull_tree(client: Any, service: RunService, target: Path) -> dict[str, Any]:
    """The pull path: materialize the served tree from ``GET /reef/harness``."""
    manifest = client.get("/reef/harness", extra_headers={SCENARIO_HEADER: service.scenario})
    if target.exists():
        shutil.rmtree(target)
    for relative, text in manifest["files"].items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return manifest


def episode_files(files: Mapping[str, str], binding: ModelBinding, descriptor: Any) -> dict[str, str]:
    """A served tree plus the transient endpoint, as the render engine would compose them.

    The published tree carries no provider, so the binding's config nodes are
    rendered separately and merged into the tree's JSON here; the driver reaches
    the same episode input the mechanism's own evaluation renders."""
    merged = dict(files)
    for path, text in render_composition(binding.compose_nodes(descriptor), descriptor).items():
        merged[path] = json.dumps(_merge(json.loads(merged.get(path, "{}")), json.loads(text)), indent=2)
    return merged


def _merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        below = merged.get(key)
        merged[key] = _merge(below, value) if isinstance(below, dict) and isinstance(value, Mapping) else value
    return merged


def solve(files: Mapping[str, str], task: str, binary: str) -> tuple[float, dict[str, Any]]:
    """One Pi episode on one problem, scored the way the mechanism scores it."""
    result = run_episode(get_adapter("pi"), files, task, binary=binary, timeout=EPISODE_TIMEOUT_S)
    response = aime.final_assistant_text(result.trajectory) or result.stdout
    return aime.evaluate(task, result), {"response": response, "exit_code": result.exit_code}


def read_archive() -> dict[str, Any]:
    path = WORK / "gepa" / f"{SCENARIO}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def mean_val(archive: Mapping[str, Any], index: int | None) -> float | None:
    candidates = archive.get("candidates") or []
    if index is None or index >= len(candidates):
        return None
    scores = candidates[index].get("val_scores")
    return sum(scores) / len(scores) if scores else None


def seed_tree(recipe: Any) -> dict[str, str]:
    """The composition the search started from, rendered for the frozen pass."""
    nodes = tuple((str(entry["name"]), entry.get("config")) for entry in recipe.seed if not entry.get("disabled"))
    return render_composition(nodes, get_adapter(recipe.adapter))


def verify_binary(binary: str) -> str:
    try:
        found = subprocess.run(
            [binary, "--version"], cwd=HERE, check=True, capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"the Pi binary {binary!r} is required: {exc}") from exc
    if found != aime.PI_VERSION:
        raise SystemExit(f"Pi version is {found!r}, expected {aime.PI_VERSION}; set REEF_PI_BINARY")
    return found


def search(service: RunService, client: Any, binary: str, trainset: list, budget: int) -> None:
    """Rounds until the archive's metric calls reach the budget.

    The round counter is the driver's own, on disk beside the archive: the
    sampler's epoch order is a function of it, so a resumed run continues the
    training order instead of redrawing minibatches it already paid for."""
    sampler = aime.Minibatches(len(trainset), 3, seed=0)
    rounds = WORK / "rounds.json"
    done = int(json.loads(rounds.read_text(encoding="utf-8"))["step"]) if rounds.is_file() else 0
    for step in range(done):
        sampler.ids(step)  # advance the sampler over the rounds a previous run drew
    for step in range(done, budget):
        archive = read_archive()
        calls = int(archive.get("metric_calls", 0))
        print(f"round {step}: {calls}/{budget} metric calls, {len(archive.get('candidates') or [])} candidates")
        if calls >= budget:
            return
        manifest = pull_tree(client, service, WORK / "served")
        files = episode_files(manifest["files"], service.binding(), get_adapter("pi"))
        # Records at or below this sequence are earlier rounds' leftovers; a
        # repeated problem must not match its own past session.
        floor = max_sequence(service.records(), service.scenario)
        before = service.training_step()
        for index, identifier in enumerate(sampler.ids(step)):
            task = str(trainset[identifier]["input"])
            score, _ = solve(files, task, binary)
            record = find_task_record(service.records(), service.scenario, task, after_sequence=floor)
            if record is None:
                raise RuntimeError(f"the episode for training problem {identifier} recorded no request")
            report(
                service,
                client,
                {"agent_record_id": f"gepa-{step}-{index}", "score": score, "references": [record.agent_record_id]},
            )
        service.wait_for_training_step(before)
        rounds.write_text(json.dumps({"step": step + 1}) + "\n", encoding="utf-8")
    print(f"stopped after {budget} rounds without reaching the metric-call budget")


def test_passes(service: RunService, recipe: Any, client: Any, binary: str, testset: list) -> dict[str, float]:
    """The sealed split, once on the seed composition and once on the served one."""
    evaluator = CheckpointedEvaluator(WORK / "heldout")
    tasks = [str(example["input"]) for example in testset]
    binding = service.binding()
    descriptor = get_adapter("pi")
    trees = {"frozen": seed_tree(recipe), "selected": pull_tree(client, service, WORK / "selected")["files"]}
    scores = {}
    for label, tree in trees.items():
        rendered = episode_files(tree, binding, descriptor)

        def one(index: int, task: str, files: Mapping[str, str] = rendered) -> tuple[float, dict[str, Any]]:
            return solve(files, task, binary)

        measured = evaluator.scores(label, tasks, tree, one)
        scores[label] = sum(measured) / len(measured)
        print(f"{label} test score: {scores[label]:.4f}")
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", type=int, default=aime.SEARCH_BUDGET, help="metric calls before the search stops")
    parser.add_argument("--dry-run", action="store_true", help="boot the recipe and print the plan; no model calls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    binary = os.environ["REEF_PI_BINARY"]
    version = verify_binary(binary)
    trainset, valset, testset = aime.load_aime_splits()
    tasks = [str(example["input"]) for example in valset]
    if args.dry_run:
        # A dry run proves the recipe boots from gepa.yaml with the real
        # validation set; the placeholder credential is never sent anywhere.
        recipe_name, _ = load_recipe(tasks, api_key="dry-run")
        plan = {
            "recipe": recipe_name,
            "scenario": SCENARIO,
            "components": COMPONENTS,
            "pi_version": version,
            "task_model": os.environ["REEF_MODEL"],
            "reflection_model": aime.REFLECTION_MODEL,
            "budget": args.budget,
            "splits": {"train": len(trainset), "validation": len(valset), "test": len(testset)},
            "dataset_sha256": aime.AIME_DATASET_SHA256,
            "planned_test_episodes": 2 * len(testset),
            "work_dir": str(WORK),
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    from reef_client import ReefClient

    key = provider_key()
    recipe_name, recipe = load_recipe(tasks, api_key=key)
    WORK.mkdir(parents=True, exist_ok=True)
    bootstrap = WORK / "bootstrap"
    if not bootstrap.is_dir():
        for relative, text in seed_tree(recipe).items():
            (bootstrap / relative).parent.mkdir(parents=True, exist_ok=True)
            (bootstrap / relative).write_text(text, encoding="utf-8")
    service = RunService(
        scenario=SCENARIO,
        recipe_name=recipe_name,
        recipe=recipe,
        bootstrap_tree=bootstrap,
        run_dir=WORK,
        upstream_url=aime.OPENAI_BASE_URL,
        upstream_key=key,
        port=0,
    )
    service.start()
    try:
        client = ReefClient(service.base_url, timeout_s=STEP_TIMEOUT_S)
        search(service, client, binary, trainset, args.budget)
        scores = test_passes(service, recipe, client, binary, testset)
        archive = read_archive()
        summary = {
            "scenario": SCENARIO,
            "components": COMPONENTS,
            "candidates": len(archive.get("candidates") or []),
            "metric_calls": archive.get("metric_calls", 0),
            "steps": archive.get("steps", 0),
            "validation_seed_score": mean_val(archive, 0),
            "validation_selected_score": mean_val(archive, archive.get("served")),
            "frozen_test_score": scores["frozen"],
            "selected_test_score": scores["selected"],
            "test_delta": scores["selected"] - scores["frozen"],
            # The retained official arm, so the run reads beside its target.
            "official_reference": json.loads(RESULTS.read_text(encoding="utf-8"))["results"]["reference"],
        }
        (WORK / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        service.stop()


if __name__ == "__main__":
    main()
