"""Reefine on pi, end to end: ask, step, activate, install, show.

One demo (``./run.sh bugfix`` or ``./run.sh research``):

    ask     - the request goes to ``POST /reef/train`` as a training
              instruction, and the deployment, in ``training_mode: manual``,
              runs one evolve step for it at once
    step    - the service proposer designs the change, writes it and
              reviews it; the evaluation runs the candidate on the recipe's
              health task and the floor decides; the catalog row carries
              the result, the notes and the request it answered under
              ``metrics.training_request``
    activate - a release that changes an extension is made but is not the
              active release; its page says why it exists and what it
              changed, and the demo activates it through the route (a person
              reads the page first)
    install - the active release's source tree
              (``GET /reef/harness?form=source``) renders for pi with a model
              binding at this Reef into work/harness; when the release names
              ``requires``, the demo checks each item on this machine first,
              and an unmet one stops it
    show    - one ``pi -p`` session on the installed tree, with the session's
              tool calls in order and its final answer

The measurement (``./run.sh measure``) posts a fixed list of requests one
after another, each once the step of the one before it settled, and prints
one row per request and the counts: filed, answered, admitted, won (met the
floor), published, inactive. The won count is the first of the two things
RFC #310's stage 6 asks for (requests that passed the checks); the held out
shapes are not here.

Start this through ./run.sh: it starts the Reef these constants point at
from configs/deployment.yaml, installs the active release, and runs a mode.
The demo is the client here: it pulls, renders and runs the release itself.
Every mode writes work/<mode>-<timestamp>.json with the raw rows it read.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from reef_client import ReefClient, ReefClientError

from recipes.beta.reefine.agent import launch_pi, tool_calls
from reef.execution.sandbox.process import LocalExecutor, RolloutTimeout
from reef.harness.adapters import get_adapter
from reef.harness.rollouts.model_binding import ModelBinding
from reef.harness.rollouts.trajectory import final_assistant_text
from reef.harness.rollouts.vendor_install import AdapterProgram
from reef.harness.step_result import result_of
from reef.harness.tree import HarnessTree, locate, render_tree

SERVICE_URL = "http://127.0.0.1:8901"  # deployment.yaml's port
SCENARIO = "reefine-demo"  # this workload's isolated lane
TOKEN = os.environ.get("REEF_TOKEN", "reef-local")  # matches deployment.yaml
MODEL = os.environ.get("REEF_UPSTREAM_MODEL", "gemma4:26b")  # run.sh exports the same default
# A local model reads the API skill and writes the change, then six pi rollouts score it.
STEP_TIMEOUT_S = 1800.0
POLL_S = 5.0

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
WORK = HERE / "work"
# The install writes the rendered tree and the release metadata file here.
INSTALL_ROOT = WORK / "harness"
DEMOS = HERE / "demos"
RELEASE_FILE = ".reef-harness-release"
#: pi's provider file: the model binding writes provider reef there, and pi sends that provider's headers.
PI_MODELS = "pi-agent/models.json"
SHOW_TIMEOUT_S = 600.0
CHECK_TIMEOUT_S = 30.0

SHOW_PROMPTS = {
    "bugfix": "fix the bug in adder.py",
    "research": "what is the best known lower bound for sorting by comparisons, with a source",
}

#: The measurement's requests: short harness changes a skill or the rules answer, no extension.
MEASURE_REQUESTS = (
    "answer in one sentence when the question is arithmetic",
    "always show the command you ran before you show its output",
    "when you edit a file, print the diff after the edit",
    "prefer python over shell for anything longer than one line",
    "when a task names a file that does not exist, say so before you create it",
    "end every answer to a counting question with the number alone on the last line",
    "run the project's tests before you say a change is done",
    "when a command fails, show its exit code and the last ten lines of its output",
    "keep replies under a hundred words unless the user asks for detail",
    "add a skill that explains how to read a csv file with the standard library",
    "add a skill named plan-first that tells you to list the steps before you edit any file",
    "when you finish a task, name the files you changed",
)


def say(text):
    print(time.strftime("%H:%M:%S"), text, flush=True)


def request_text(demo):
    """The request of demos/<demo>.md: the text of its first fenced block, as one line."""
    inside, block = False, []
    for line in (DEMOS / f"{demo}.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            if inside:
                break
            inside = True
            continue
        if inside:
            block.append(line.strip())
    text = " ".join(part for part in block if part)
    if not text:
        raise SystemExit(f"demos/{demo}.md carries no fenced request")
    return text


# -- the service ----------------------------------------------------------------------------------------------


def _client():
    # The first call loads the model on a local server, minutes beside a model already resident.
    return ReefClient(SERVICE_URL, token=TOKEN, timeout_s=300.0)


def _scenario_headers():
    return {"x-reef-scenario": SCENARIO}


def _fetch_text(path):
    """One raw read of a route that answers text, not JSON: a step's page."""
    request = urllib.request.Request(
        f"{SERVICE_URL}{path}", headers={"Authorization": f"Bearer {TOKEN}", **_scenario_headers()}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def _rows(client):
    """The catalog oldest first, as GET /reef/harness/releases lists it; empty before the scenario exists."""
    try:
        return client.get("/reef/harness/releases", extra_headers=_scenario_headers())["releases"]
    except ReefClientError as exc:
        if exc.status != 404:
            raise
        return []


def _training_rows(rows):
    return [row for row in rows if row.get("operation") == "training"]


def _request_text_of(row):
    """The request the step answered, from the row the trainer writes for every instruction step; None otherwise."""
    return ((row.get("metrics") or {}).get("training_request") or {}).get("text")


def _manifest(client):
    return client.get("/reef/harness", extra_headers=_scenario_headers())


def _ensure_scenario(client):
    """A scenario exists once traffic has named it: one minimal inference creates it on its base release."""
    try:
        return _manifest(client)
    except ReefClientError as exc:
        if exc.status != 404:
            raise
    client.inference_with_record(
        SCENARIO,
        "/v1/chat/completions",
        {"model": MODEL, "messages": [{"role": "user", "content": "Reply with one word."}], "max_tokens": 1},
    )
    return _manifest(client)


def changes_of(metrics):
    """The step's patch as ``write path`` and ``remove path`` lines, in path order."""
    patch = metrics.get("patch") or {}
    return [f"{'remove' if text is None else 'write'} {path}" for path, text in sorted(patch.items())]


def kinds_of(metrics):
    """The tree kinds the step's patch touches."""
    return sorted(
        {"?" if (location := locate(path)) is None else location.kind.value for path in metrics.get("patch") or {}}
    )


def tally_parts(metrics):
    """The evaluation's counts over the tasks: the floor's (passed, failed), else (wins, losses, ties); None when no
    evaluation ran.

    The floor records how many tasks the candidate passed and failed. Before
    it, the score comparison selector recorded the three counts and
    ``selection: always`` only the per task scores of both sides, so on those
    rows the counts come from comparing the scores."""
    if all(isinstance(metrics.get(key), int) for key in ("passed", "failed")):
        return metrics["passed"], metrics["failed"]
    if all(isinstance(metrics.get(key), int) for key in ("wins", "losses", "ties")):
        return metrics["wins"], metrics["losses"], metrics["ties"]
    scored = ((metrics.get("selection") or {}).get("evaluation") or {}).get("metrics") or {}
    candidate, current = scored.get("candidate_scores"), scored.get("current_scores")
    if not isinstance(candidate, list) or not isinstance(current, list) or len(candidate) != len(current):
        return None
    # A rollout that failed scores None, which the score comparison ranks below every number.
    ranked = [
        (-float("inf") if c is None else c, -float("inf") if k is None else k)
        for c, k in zip(candidate, current, strict=True)
    ]
    return (
        sum(1 for c, k in ranked if c > k),
        sum(1 for c, k in ranked if c < k),
        sum(1 for c, k in ranked if c == k),
    )


def tally(metrics):
    """The evaluation's counts as one cell: ``passed / failed`` under the floor, ``wins / losses / ties`` before it,
    ``-`` when no evaluation ran."""
    parts = tally_parts(metrics)
    return " / ".join(str(part) for part in parts) if parts else "-"


def evaluation_counts(metrics):
    """The evaluation's counts by name: ``passed`` and ``failed`` under the floor, ``wins``, ``losses`` and ``ties``
    on a row from before it; empty when no evaluation ran."""
    parts = tally_parts(metrics)
    if parts is None:
        return {}
    names = ("passed", "failed") if len(parts) == 2 else ("wins", "losses", "ties")
    return dict(zip(names, parts, strict=True))


def _requires_of(items):
    return ", ".join(f"{item.get('name')} ({item.get('kind', '?')})" for item in items) or "nothing"


# -- the client: ask, install --------------------------------------------------------------------------------


def _installed_release():
    """The release id of the tree under work/harness, from the release metadata file the install wrote."""
    try:
        return json.loads((INSTALL_ROOT / RELEASE_FILE).read_text(encoding="utf-8")).get("release_id")
    except (OSError, ValueError):
        return None


def unmet(items):
    """The requires items this machine does not meet: an env item whose variable is unset, a binary not on PATH,
    an item whose check does not exit 0. The demo runs each check unasked; a person would read it first."""
    missing = []
    for item in items:
        check = item.get("check")
        if item.get("kind") == "env":
            met = bool(os.environ.get(check or item.get("name", "")))
        elif item.get("kind") == "binary" and not check:
            met = shutil.which(str(item.get("name"))) is not None
        else:
            try:
                met = (
                    subprocess.run(
                        ["bash", "-lc", str(check)], capture_output=True, timeout=CHECK_TIMEOUT_S
                    ).returncode
                    == 0
                )
            except subprocess.TimeoutExpired:
                met = False
        if not met:
            missing.append(item)
    return missing


def install(release_id=None):
    """Render the active release (or ``release_id``) for pi with a model binding at this Reef, into work/harness.

    Return the release id, or None when the release requires something this machine lacks.
    """
    client = _client()
    _ensure_scenario(client)
    query = "?form=source" + (f"&release_id={release_id}" if release_id else "")
    manifest = client.get(f"/reef/harness{query}", extra_headers=_scenario_headers())
    requires = manifest.get("requires") or []
    say(f"requires: {_requires_of(requires)}")
    missing = unmet(requires)
    if missing:
        for item in missing:
            say(f"unmet: {item.get('name')} ({item.get('kind')}): {item.get('prompt') or 'no prompt given'}")
        return None
    descriptor = get_adapter("pi")
    binding = ModelBinding(base_url=SERVICE_URL, model=MODEL, api_key=TOKEN)
    files = render_tree(HarnessTree(manifest["files"]), descriptor, binding.config_overlays(descriptor))
    # Reef answers a model call only with the scenario header. The render refuses a tree that sets provider
    # headers, so the client adds its own to its binding after the render; pi sends them on every call.
    models = json.loads(files[PI_MODELS])
    models["providers"]["reef"]["headers"] = _scenario_headers()
    files[PI_MODELS] = json.dumps(models, indent=2, sort_keys=True) + "\n"
    shutil.rmtree(INSTALL_ROOT, ignore_errors=True)
    for relative, content in files.items():
        target = INSTALL_ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    record = {"release_id": manifest["release_id"], "files": sorted(files)}
    (INSTALL_ROOT / RELEASE_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    say(f"installed release {manifest['release_id']} at {INSTALL_ROOT}")
    return manifest["release_id"]


def ask(text):
    """The request as a training instruction; the id of the training record the service accepted.

    The instruction names the installed release and this demo's session, and
    the deployment's manual mode runs one step for it. A refusal (the
    screens, a mode that takes no instructions) is a stop, with the service's
    answer saying why."""
    say(f"ask: POST /reef/train {text!r}")
    payload = {"text": text, "session": SCENARIO, "release_id": _installed_release() or ""}
    try:
        answer, _ = _client().post("/reef/train", SCENARIO, payload)
    except ReefClientError as exc:
        raise SystemExit(f"the request was not accepted: {exc}") from exc
    say(
        f"accepted: training request {answer['agent_record_id']}; its page is {SERVICE_URL}{answer.get('page_path', '')}"
    )
    return answer["agent_record_id"]


def wait_for_step(client, text, before, deadline):
    """The catalog and the training row whose request is ``text``, once that step settled; every new row is
    printed as it lands. ``(rows, None)`` at the deadline.

    Manual mode answers the accepted instructions oldest first, so a request
    a stopped run left queued gets its step before ours; that row is printed
    as an earlier request's and the wait goes on for the row carrying our
    text."""
    shown = before
    rows = []
    while time.monotonic() < deadline:
        try:
            rows = _rows(client)
            error = client.get("/reef/status").get("error")
        except (TimeoutError, OSError) as exc:
            # A step in flight holds the catalog until it ends; a service that is gone does not answer /healthz.
            try:
                client.get("/healthz")
            except (ReefClientError, TimeoutError, OSError):
                raise SystemExit("the service stopped answering; check work/reef.log") from exc
            time.sleep(POLL_S)
            continue
        if error:
            raise SystemExit(f"evolve step failed: {error}; check work/reef.log")
        steps = _training_rows(rows)
        for row in steps[shown:]:
            metrics = row.get("metrics") or {}
            whose = "" if _request_text_of(row) in (None, text) else "; an earlier request's step"
            release = str(row.get("release_id"))[:12]
            say(f"step {metrics.get('steps', '?')}: {result_of(row)} (release {release}){whose}")
        shown = max(shown, len(steps))
        for row in steps[before:]:
            if _request_text_of(row) == text:
                return rows, row
        time.sleep(POLL_S)
    return rows, None


def activate(client, rows, row, run_dir):
    """Save the inactive step's page beside the run, print its URL, activate the row's artifact; the id of the
    active release that the activation names."""
    step = rows.index(row)
    page = f"/reef/harness/releases/{step}/page"
    saved = run_dir / f"step-{step}.html"
    saved.write_text(_fetch_text(page), encoding="utf-8")
    say(f"inactive: the page says why and what changed: {SERVICE_URL}{page} (saved as {saved.relative_to(HERE)})")
    say("activate: the demo is scripted; a person reads the page first")
    body = {"artifact_id": row["artifact_id"]}
    if row.get("component"):
        body["component"] = row["component"]
    answer, _ = client.post(f"/reef/scenarios/{SCENARIO}/activate", SCENARIO, body)
    if answer.get("artifact_id") != row["artifact_id"]:
        raise SystemExit(f"reef answered the activation without artifact {row['artifact_id']}: {answer}")
    say(f"activated: release {answer['release_id']} is the active release")
    return answer["release_id"]


# -- the show session -----------------------------------------------------------------------------------------


def show(mode, run_dir):
    """One pi session on the installed tree in the mode's workspace; the tool calls in order and the final answer.

    The session runs headless, as a rollout does, over a copy of the installed
    files; pi's own session log is the record this driver reads."""
    root = run_dir / "session"
    workspace = root / "workspace"
    if mode == "bugfix":
        # A copy: the session edits adder.py, and the committed fixture must fail again next time.
        shutil.copytree(DEMOS / "workspace", workspace)
    prompt = SHOW_PROMPTS[mode]
    say(f"show: pi -p {prompt!r} in {workspace.relative_to(HERE)}")
    descriptor = get_adapter("pi")
    executor, binary = LocalExecutor().for_program(AdapterProgram(descriptor))
    files = {
        path.relative_to(INSTALL_ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(INSTALL_ROOT.rglob("*"))
        if path.is_file() and path.name != RELEASE_FILE
    }
    try:
        outcome, trajectory = launch_pi(
            descriptor, binary, executor, files, prompt, {}, root=root, timeout=SHOW_TIMEOUT_S
        )
        exit_code = outcome.exit_code
    except RolloutTimeout:
        exit_code, trajectory = None, []
    calls = [f"{call['tool']}({call.get('arguments', call.get('error', ''))[:60]})" for call in tool_calls(trajectory)]
    answer = (final_assistant_text(trajectory) or "").strip()
    say(f"show: tool calls in order: {' -> '.join(calls) or 'none'}")
    say(f"show: final answer: {answer.splitlines()[-1][:160] if answer else '(none)'}")
    return {"prompt": prompt, "exit": exit_code, "tool_calls": calls, "answer": answer}


# -- the modes ------------------------------------------------------------------------------------------------


def _write_record(path, record):
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    say(f"record: {path.relative_to(HERE)}")


def _print_table(headers, rows):
    """A markdown table, so a row can go into the README as it is."""
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for row in rows:
        print("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")


def was_evaluated(result):
    """Whether the evaluation ran for the row: it carries the floor's counts or the older tally."""
    return result.get("passed") is not None or result.get("wins") is not None


def _won(result):
    """Whether the row met the floor; on a row from before the floor, whether it had more wins than losses."""
    if result.get("passed") is not None:
        return result.get("failed") == 0
    return result.get("wins") is not None and result["wins"] > (result.get("losses") or 0)


def totals_of(results):
    """The measurement's counts: filed, answered (a patch came back), admitted (the evaluation ran), won (met the
    floor; on a row from before the floor, more wins than losses), published, inactive. Under ``selection:
    always`` a publish said nothing about the evaluation, so won and published are counted apart; under the floor a
    publish is a win, and the two agree unless a release waits as inactive."""
    return {
        "filed": sum(1 for r in results if r.get("filed")),
        # A patch came back: a step that skipped on a refusal or on a failed step carries none.
        "answered": sum(1 for r in results if r.get("filed") and r.get("kinds") not in (None, "-")),
        "admitted": sum(1 for r in results if was_evaluated(r)),
        "won": sum(1 for r in results if _won(r)),
        "published": sum(1 for r in results if r.get("result", r.get("verdict")) == "selected"),
        "inactive": sum(1 for r in results if r.get("result", r.get("verdict")) == "inactive"),
    }


def demo(mode):
    text = request_text(mode)
    client = _client()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = WORK / f"{mode}-{stamp}"
    run_dir.mkdir(parents=True)
    record_path = WORK / f"{mode}-{stamp}.json"
    record = {"mode": mode, "model": MODEL, "scenario": SCENARIO, "request": text, "started_at": time.time()}
    installed_before = _installed_release()
    say(f"installed before the ask: release {installed_before}")
    before = len(_training_rows(_rows(client)))
    started = time.monotonic()
    record["request_id"] = ask(text)
    rows, row = wait_for_step(client, text, before, started + STEP_TIMEOUT_S)
    record["rows"] = rows
    if row is None:
        _write_record(record_path, record)
        raise SystemExit(f"no step settled the request within {STEP_TIMEOUT_S:.0f} s; check work/reef.log")
    metrics = row.get("metrics") or {}
    selection_result = result_of(row)
    changes = changes_of(metrics)
    say(f"result: {selection_result}; evaluation {tally(metrics)}; proposer {metrics.get('proposer_seconds', '-')} s")
    say(f"changes: {'; '.join(changes) or 'none'}")
    activated = None
    if selection_result == "inactive":
        activated = activate(client, rows, row, run_dir)
    requires = _manifest(client).get("requires") or []
    result = {
        "request": text,
        "result": selection_result,
        "evaluation": tally(metrics),
        "kinds": ", ".join(kinds_of(metrics)) or "-",
        "inactive": selection_result == "inactive",
        "activated": activated or "-",
        "requires": _requires_of(requires),
        "proposer_seconds": metrics.get("proposer_seconds", "-"),
    }
    if selection_result == "selected" or activated:
        installed = install()
        if installed is None:
            record["result"] = {**result, "installed": "-", "setup": "unmet requires"}
            _write_record(record_path, record)
            raise SystemExit(2)
    else:
        say(f"the active release did not change: release {installed_before} stays installed")
        installed = installed_before
    # From the evolve call to the end of the install, or to the result when nothing new installed.
    seconds = round(time.monotonic() - started, 1)
    shown = show(mode, run_dir)
    result.update(
        {
            "installed": installed,
            "show_tool_calls": " -> ".join(shown["tool_calls"]) or "none",
            "ask_to_install_s": seconds,
        }
    )
    record["result"] = result
    record["show"] = shown
    _write_record(record_path, record)
    print()
    _print_table(("Field", "Value"), list(result.items()))


def measure(n):
    texts = MEASURE_REQUESTS[: max(n, 0)]
    if n > len(MEASURE_REQUESTS):
        say(f"the fixed list holds {len(MEASURE_REQUESTS)} requests; filing them all")
    client = _client()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    record_path = WORK / f"measure-{stamp}.json"
    record = {"mode": "measure", "model": MODEL, "scenario": SCENARIO, "started_at": time.time(), "results": []}
    results = record["results"]
    for index, text in enumerate(texts, start=1):
        say(f"request {index} of {len(texts)}: {text!r}")
        before = len(_training_rows(_rows(client)))
        started = time.monotonic()
        try:
            request_id = ask(text)
        except SystemExit as exc:
            results.append({"request": text, "filed": False, "result": str(exc)})
            continue
        _, row = wait_for_step(client, text, before, started + STEP_TIMEOUT_S)
        seconds = round(time.monotonic() - started, 1)
        if row is None:
            results.append({"request": text, "id": request_id, "filed": True, "result": "no step", "seconds": seconds})
            continue
        metrics = row.get("metrics") or {}
        results.append(
            {
                "request": text,
                "id": request_id,
                "filed": True,
                "kinds": ", ".join(kinds_of(metrics)) or "-",
                "result": result_of(row),
                **evaluation_counts(metrics),
                "evaluation": tally(metrics),
                "seconds": seconds,
                "row": row,
            }
        )
        say(f"result: {result_of(row)}; evaluation {tally(metrics)}; {seconds} s")
        # The record after every request, so a run stopped midway still leaves its rows.
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    totals = totals_of(results)
    record["totals"] = totals
    _write_record(record_path, record)
    print()
    _print_table(
        ("Request", "Kind proposed", "Result", "Evaluation", "Seconds"),
        [
            (
                r["request"],
                r.get("kinds", "-"),
                r.get("result", r.get("verdict", "-")),
                r.get("evaluation", r.get("gate", "-")),
                r.get("seconds", "-"),
            )
            for r in results
        ],
    )
    print()
    _print_table(("Filed", "Answered", "Admitted", "Won", "Published", "Inactive"), [tuple(totals.values())])


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Reefine on pi: the two demos and the measurement; start it through ./run.sh",
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("install", help="install the active release into work/harness (run.sh does this first)")
    modes.add_parser("bugfix", help="the bug fix flow demo: demos/bugfix.md")
    modes.add_parser("research", help="the research loop demo: demos/research.md")
    measured = modes.add_parser("measure", help="file the fixed request list and count what passed the checks")
    measured.add_argument("--n", type=int, default=10, help="how many of the fixed requests to file (default 10)")
    args = parser.parse_args(argv)
    if args.mode == "install":
        if install() is None:
            raise SystemExit(2)
    elif args.mode == "measure":
        measure(args.n)
    else:
        demo(args.mode)


if __name__ == "__main__":
    main()
