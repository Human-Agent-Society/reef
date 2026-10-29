"""The AIME driver: one embedded Reef service, GEPA's search, two sealed test passes.

The method is the recipe (``gepa.yaml``); this file is only the benchmark
around it. It owns the Reef service in process, so every problem it
solves is real recorded traffic against the served tree and the method
reflects on transcripts it actually produced.

One round is one GEPA step: pull the served tree, run the next epoch-shuffled
minibatch of three training problems as Pi rollouts against it with a model
binding pointed back at this service, report each score against the request
that rollout recorded, and wait. The third report closes the batch
(``evolution.batch_size: 3``), which is what schedules the step where the method
proposes, the evaluation scores the candidate on the 45 validation problems,
and a strict mean improvement publishes it. Rounds stop at ``--budget`` metric
calls; then the seed and the served trees are each scored on the sealed
150-problem test split through per-example checkpoints and ``summary.json``
puts those numbers beside the retained official record.

The work directory resumes: the scenario's commit history, which carries the
method's ``archive.json``, the harness history under ``harness/``, and the
checkpoints survive a restart. Each round reads the archive from the latest
commit, settles it from the history, and draws the iteration's plan exactly as
the method will (``archive_for_step``, ``Archive.plan``), so the problems it
sends are the minibatch GEPA's order names. ``--dry-run`` boots the recipe and
prints the plan without a model call.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from aiohttp import web

import harness
from harness import aime
from harness.heldout import CheckpointedEvaluator
from recipes.gepa import GEPA
from recipes.gepa.archive import Archive
from recipes.gepa.components import texts_of
from recipes.gepa.method import ARCHIVE_FILE, archive_for_step
from reef.artifact.memory import InMemoryRepositoryBackend
from reef.core.records_types import RequestType
from reef.dispatcher import Dispatcher
from reef.evaluation.harness import History, evaluation_selection
from reef.harness import run_rollout
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.rollouts.model_binding import ModelBinding
from reef.harness.sync import HARNESS_COMPONENT
from reef.harness.tree import HarnessTree, TreePatch, render_tree
from reef.inference.http import HttpInferenceHandler, InferenceProxyRuntime, provider_request_headers
from reef.recipe.config import recipe_config_from_file_layout
from reef.recipe.harness import HISTORY_DIR, HarnessEvolutionRecipe
from reef.recipe.registry import build_recipe
from reef.service.app import create_app
from reef.service.deploy.config_utils import load_config
from reef.service.wire import SCENARIO_HEADER
from reef.storage.records import RecordStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.harness_backend.state import HarnessState

HERE = Path(__file__).resolve().parent
# run.sh exports these; the defaults repeat here so the module imports (and
# the dry run works) without it. gepa.yaml interpolates REEF_WORK, so it has
# to be set before load_recipe reads the file, not before this import block.
os.environ.setdefault("REEF_WORK", str(HERE / "work"))
os.environ.setdefault("REEF_MODEL", "gpt-4.1-mini-2025-04-14")
os.environ.setdefault("REEF_PI_BINARY", "pi")
os.environ.setdefault("REEF_GEPA_WORKERS", "128")
os.environ.setdefault("REEF_GEPA_EXECUTOR", "auto")

WORK = Path(os.environ["REEF_WORK"])
SCENARIO = os.environ.get("REEF_GEPA_SCENARIO", "gepa-aime")
# The multi-node variant evolves the rules node and a skill node together;
# the default single-node run is the one the retained record compares against.
MULTI = os.environ.get("REEF_GEPA_MULTI") == "1"
COMPONENTS = ["rules", "skill"] if MULTI else ["rules"]
# The service key the Pi rollouts present. The embedded service does not
# authenticate; this must never be the provider key, which would land in a
# rendered models.json inside the rollout root.
EMBEDDED_KEY = "reef-embedded"
ROLLOUT_TIMEOUT_S = 600.0
# Rollouts are independent: the evaluation's validation pass (gepa.yaml's
# execution.evolution.workers), the driver's minibatch, and the test passes all run this
# many at once. It only changes wall time, never sampling or scoring.
WORKERS = int(os.environ["REEF_GEPA_WORKERS"])
STEP_TIMEOUT_S = 7200.0
RESULTS = HERE / "results" / "quickstart-seed-0-2026-09-01" / "manifest.json"


def load_recipe(
    tasks: list[str], *, api_key: str, seed: int = 0, trainset_size: int
) -> tuple[str, HarnessEvolutionRecipe]:
    """The recipe gepa.yaml declares, with the validation split filled in.

    This driver is the deployment: it owns the provider endpoint and hands it
    to the recipe as its runtime, the way ``reef.upstream_url`` does in a served
    deployment. The method never sees the endpoint or the key.
    """
    config = load_config(HERE / "gepa.yaml")
    sections = recipe_config_from_file_layout(config)
    evolution = dict(sections["evolution"])
    evolution["evaluation"] = {**evolution["evaluation"], "tasks": tasks}
    # One --seed drives both random draws, the parent sampling and the minibatch order, which the method plans
    # over the training split.
    method = {**evolution["method"]["config"], "seed": seed, "trainset_size": trainset_size}
    seed_tree = HarnessTree.read(HERE / str(evolution["seed"]))
    if MULTI:
        seed_tree = seed_tree.apply(TreePatch({f"skills/{aime.SKILL_SEED_NAME}/SKILL.md": aime.SKILL_SEED}))
        method["components"] = COMPONENTS
    evolution["method"] = {**evolution["method"], "config": method}
    seed_dir = WORK / "seed"
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_tree.write(seed_dir)
    evolution["seed"] = str(seed_dir)
    sections["evolution"] = evolution
    runtime = InferenceProxyRuntime(
        model_path=os.environ["REEF_MODEL"],
        base_url=aime.OPENAI_BASE_URL,
        api_key=api_key,
        inference_timeout_s=ROLLOUT_TIMEOUT_S,
    )
    recipe = build_recipe(str(sections["implementation"]), config=sections, runtime=runtime)
    if not isinstance(recipe, HarnessEvolutionRecipe) or not isinstance(recipe.method, GEPA):
        raise SystemExit("gepa.yaml must name the harness evolution recipe with recipes.gepa:GEPA as its method")
    # Importable classes alone do not carry the driver's registered answers into spawn or Ray workers. The
    # example's own hooks become one snapshot of the answers and contexts; a hook the config names otherwise stays.
    snapshot = aime.AIMEScorer(aime.ANSWERS, aime.CONTEXTS)
    scorer = snapshot if isinstance(recipe.scorer, aime.AIMEAnswerScorer) else recipe.scorer
    feedback = snapshot if isinstance(recipe.method.feedback, aime.AIMEFeedback) else recipe.method.feedback
    return recipe.name, dataclasses.replace(recipe, scorer=scorer, method=GEPA(recipe.method.settings, feedback))


@contextmanager
def worker_runtime(recipe: HarnessEvolutionRecipe):
    """Ship only the example's harness package; nodes provide Reef and Pi.

    Ray reads RAY_ADDRESS when set; without it, start a local Ray runtime.
    Do not tear down an externally owned runtime when embedding this driver.
    """
    selection, _ = evaluation_selection(recipe.scorer, recipe.worker_executor)
    if selection.settings.backend != "ray":
        yield
        return
    import ray

    owned = not ray.is_initialized()
    if owned:
        ray.init(runtime_env={"py_modules": [harness]})
    try:
        yield
    finally:
        if owned:
            ray.shutdown()


def provider_key() -> str:
    key = os.environ.get(aime.API_KEY_ENV, "").strip()
    if not key:
        raise SystemExit(f"{aime.API_KEY_ENV} is not set; no model calls were made")
    return key


def default_headers_middleware(scenario: str) -> Any:
    """Pi's requests ride this run's scenario: a rollout knows only an
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
        recipe: HarnessEvolutionRecipe,
        bootstrap_tree: Path,
        run_dir: Path,
        upstream_url: str,
        upstream_key: str,
        port: int,
        inference_handler: Any | None = None,
    ) -> None:
        self.scenario = scenario
        self.recipe_name = recipe_name
        self.recipe = recipe
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.dispatcher = Dispatcher(
            recipe,
            InMemoryRepositoryBackend.factory(bootstrap_tree, root=run_dir / "artifacts"),
            local_artifact_dir=run_dir / "staged",
            scenario_storage=SQLiteScenarioStorage(run_dir / "reef-data"),
        )
        self.app = create_app(
            self.dispatcher,
            inference_handler=inference_handler
            or HttpInferenceHandler(
                upstream_url,
                request_headers=provider_request_headers(upstream_key),
                timeout_s=ROLLOUT_TIMEOUT_S,
            ),
        )
        self.app.middlewares.append(default_headers_middleware(scenario))
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.runner: web.AppRunner | None = None

    def start(self) -> None:
        # Scenario creation (or durable recovery from the commit history) happens
        # before any request lands.
        self.dispatcher.get_or_create_scenario(self.scenario)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="reef-embedded", daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.serve(), self.loop).result(timeout=60)

    async def serve(self) -> None:
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        # Rollouts run on this host. Port 0 (tests) binds an ephemeral port,
        # read back here so the binding can point at it.
        await web.TCPSite(self.runner, host="127.0.0.1", port=self.port).start()
        if not self.port:
            self.port = int(self.runner.addresses[0][1])
            self.base_url = f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        if self.loop is not None:
            if self.runner is not None:
                asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop).result(timeout=60)
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.thread is not None:
                self.thread.join(timeout=60)
        self.dispatcher.close()

    def binding(self) -> ModelBinding:
        """The endpoint a rollout reaches this service through."""
        return ModelBinding(self.base_url, os.environ["REEF_MODEL"], api_key=EMBEDDED_KEY, timeout_s=ROLLOUT_TIMEOUT_S)

    def committed_state(self) -> HarnessState:
        """The harness backend's state as the latest commit recorded it; the initial state before any."""
        scenario = self.dispatcher.get_or_create_scenario(self.scenario)
        assert scenario is not None
        history = scenario.store.history()
        state = history[-1].algorithm_state if history else None
        return HarnessState() if state is None else HarnessState.from_json(state)

    def history(self) -> History:
        """The scenario's harness history (every candidate, rollout and decision) as the latest step left it."""
        return History(self.recipe.state_root(self.scenario) / HARNESS_COMPONENT / HISTORY_DIR)

    def records(self) -> RecordStore:
        scenario = self.dispatcher.get_or_create_scenario(self.scenario)
        assert scenario is not None
        return scenario.records

    def training_step(self) -> int:
        current = self.dispatcher.get_or_create_scenario(self.scenario)
        assert current is not None
        return current.scenario_step

    def wait_for_training_step(self, after: int) -> None:
        deadline = time.monotonic() + STEP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.training_step() > after:
                return
            if error := self.dispatcher.build_training_status()["error"]:
                raise RuntimeError(f"the training step failed: {error}")
            time.sleep(1.0)
        raise TimeoutError(f"training did not advance past step {after}")


def find_task_record(records: RecordStore, scenario: str, task_prompt: str, *, after_sequence: int = 0) -> Any | None:
    """The problem's latest recorded request past the floor.

    Rollouts share one scenario and send no session id, so the problem statement
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
                    if task_prompt in user_text(message.get("content")):
                        latest = item
                    break


def user_text(content: Any) -> str:
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
        client.report(service.scenario, dict(payload))
    except ReefClientError as exc:
        if exc.status != 409:
            raise


def pull_tree(client: Any, service: RunService, target: Path) -> dict[str, Any]:
    """The pull path: materialize the served source tree from ``GET /reef/harness?form=source``."""
    manifest = client.get("/reef/harness?form=source", extra_headers={SCENARIO_HEADER: service.scenario})
    if target.exists():
        shutil.rmtree(target)
    HarnessTree(manifest["files"]).write(target)
    return manifest


def rollout_files(tree: HarnessTree, binding: ModelBinding, descriptor: AdapterDescriptor) -> dict[str, str]:
    """``tree`` rendered with the transient endpoint: the rollout input the step's evaluation renders."""
    return render_tree(tree, descriptor, binding.config_overlays(descriptor))


def solve(files: Mapping[str, str], task: str, binary: str) -> tuple[float, dict[str, Any]]:
    """One Pi rollout on one problem, scored the way the step's evaluation scores it."""
    result = run_rollout(get_adapter("pi"), files, task, binary=binary, timeout=ROLLOUT_TIMEOUT_S)
    response = aime.final_assistant_text(result.trajectory) or result.stdout
    return aime.evaluate(task, result), {"response": response, "exit_code": result.exit_code}


def committed_archive(service: RunService, tasks: Sequence[str], served: HarnessTree, seed: int) -> Archive:
    """The archive as the next step starts from it: the latest commit's, settled from the history, current with
    ``served``."""
    state = service.committed_state()
    return archive_for_step(
        state.method_files.get(ARCHIVE_FILE),
        service.history(),
        tasks,
        texts_of(served, get_adapter("pi").layout, COMPONENTS),
        repeats=service.recipe.evaluation.repeats,
        rng_seed=seed,
    )


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


def search(
    service: RunService, client: Any, binary: str, trainset: list, validation: Sequence[str], budget: int, seed: int
) -> None:
    """Rounds until the archive's metric calls reach the budget.

    The method plans each iteration - the parent it will reflect from and the
    training problems it wants to see - from one seeded generator in GEPA's own
    order. The plan is a function of the committed archive, the history and the
    served tree, so the driver draws the plan the method will and carries it
    out: the same seed walks the same problems as ``gepa.optimize``, and a
    resumed run replays the iteration it was in rather than drawing a new one."""
    for _ in range(budget):
        served = HarnessTree(pull_tree(client, service, WORK / "served")["files"])
        search_archive = committed_archive(service, validation, served, seed)
        step = search_archive.iteration
        calls = search_archive.metric_calls
        print(f"round {step}: {calls}/{budget} metric calls, {len(search_archive.candidates)} candidates")
        if calls >= budget:
            return
        plan = search_archive.plan(len(trainset), len(validation), 3)
        files = rollout_files(served, service.binding(), get_adapter("pi"))
        # Records at or below this sequence are earlier rounds' leftovers; a
        # repeated problem must not match its own past session.
        floor = max_sequence(service.records(), service.scenario)
        before = service.training_step()
        problems = [str(trainset[identifier]["input"]) for identifier in plan["minibatch"]]
        with ThreadPoolExecutor(max_workers=min(WORKERS, len(problems))) as pool:
            scored = [score for score, _ in pool.map(lambda task, files=files: solve(files, task, binary), problems)]
        # Reports go out only once every rollout of the round has recorded,
        # so the batch never closes on a round that is still in flight.
        for index, (task, score) in enumerate(zip(problems, scored, strict=True)):
            record = find_task_record(service.records(), service.scenario, task, after_sequence=floor)
            if record is None:
                raise RuntimeError(f"the rollout for training problem {index} recorded no request")
            report(
                service,
                client,
                {"agent_record_id": f"gepa-{step}-{index}", "score": score, "references": [record.agent_record_id]},
            )
        service.wait_for_training_step(before)
    print(f"stopped after {budget} rounds without reaching the metric-call budget")


def test_passes(
    service: RunService, recipe: HarnessEvolutionRecipe, client: Any, binary: str, testset: list
) -> dict[str, float]:
    """The sealed split, once on the seed tree and once on the served one."""
    evaluator = CheckpointedEvaluator(WORK / "heldout", workers=WORKERS)
    tasks = [str(example["input"]) for example in testset]
    binding = service.binding()
    descriptor = get_adapter("pi")
    trees = {"frozen": recipe.seed, "selected": HarnessTree(pull_tree(client, service, WORK / "selected")["files"])}
    scores = {}
    for label, tree in trees.items():
        rendered = rollout_files(tree, binding, descriptor)

        def one(index: int, task: str, files: Mapping[str, str] = rendered) -> tuple[float, dict[str, Any]]:
            return solve(files, task, binary)

        measured = evaluator.scores(label, tasks, tree.files, one)
        scores[label] = sum(measured) / len(measured)
        print(f"{label} test score: {scores[label]:.4f}")
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", type=int, default=aime.SEARCH_BUDGET, help="metric calls before the search stops")
    parser.add_argument("--seed", type=int, default=0, help="the search seed: minibatch order and parent sampling")
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
        recipe_name, _ = load_recipe(tasks, api_key="dry-run", seed=args.seed, trainset_size=len(trainset))
        plan = {
            "recipe": recipe_name,
            "scenario": SCENARIO,
            "components": COMPONENTS,
            "pi_version": version,
            "task_model": os.environ["REEF_MODEL"],
            "reflection_model": aime.REFLECTION_MODEL,
            "budget": args.budget,
            "seed": args.seed,
            "splits": {"train": len(trainset), "validation": len(valset), "test": len(testset)},
            "dataset_sha256": aime.AIME_DATASET_SHA256,
            "planned_test_rollouts": 2 * len(testset),
            "workers": WORKERS,
            "work_dir": str(WORK),
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    from reef_client import ReefClient

    key = provider_key()
    WORK.mkdir(parents=True, exist_ok=True)
    recipe_name, recipe = load_recipe(tasks, api_key=key, seed=args.seed, trainset_size=len(trainset))
    bootstrap = WORK / "bootstrap"
    if not bootstrap.is_dir():
        # The base release holds the source tree, as every harness release does.
        recipe.seed.write(bootstrap)
    with worker_runtime(recipe):
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
        try:
            service.start()
            client = ReefClient(service.base_url, timeout_s=STEP_TIMEOUT_S)
            search(service, client, binary, trainset, tasks, args.budget, args.seed)
            scores = test_passes(service, recipe, client, binary, testset)
            final = committed_archive(service, tasks, HarnessTree.read(WORK / "selected"), args.seed)
            summary = {
                "scenario": SCENARIO,
                "seed": args.seed,
                "components": COMPONENTS,
                "candidates": len(final.candidates),
                "metric_calls": final.metric_calls,
                "steps": final.steps,
                "validation_seed_score": final.mean_val(0),
                "validation_selected_score": None if final.served is None else final.mean_val(final.served),
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
