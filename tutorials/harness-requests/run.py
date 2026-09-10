"""Harness requests v1 on reef-pi, end to end: ask, step, promote, setup, install, show.

One demo (``./run.sh bugfix`` or ``./run.sh research``):

    ask     - ``reef-pi harness "<request>"``: the wrapper posts the request
              to ``POST /reef/train`` as a training instruction, and the
              deployment, in ``training_mode: manual``, runs one evolve step
              for it at once
    step    - the service proposer reads the request and writes the change;
              the gate scores it on the recipe's tasks; the catalog row
              carries the verdict and the request it answered under
              ``metrics.training_request``
    promote - a release that touches a code_extension waits as pending; its
              page says why it exists and what it changed, and the demo
              promotes it through the route (a person reads the page first)
    setup   - when the head's manifest names ``requires``, ``reef-pi setup``
              checks the items off; an unmet item stops the demo
    install - the head installs through the install route into work/harness
    show    - one ``reef-pi -p`` session on the installed tree, with the
              session's tool calls in order and its final answer

The measurement (``./run.sh measure``) posts a fixed list of requests one
after another, each once the step of the one before it settled, and prints
one row per request and the counts: filed, answered, admitted, won,
published, pending. The won count is the first of the two things RFC #310's
stage 6 asks for (requests that won the gate); the held out shapes are not
here.

Start this through ./run.sh: it starts the Reef these constants point at
from configs/deployment.yaml, installs the served tree, and runs a mode.
Every mode writes work/<mode>-<timestamp>.json with the raw rows it read.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from reef_client import ReefClient, ReefClientError

SERVICE_URL = "http://127.0.0.1:8901"  # deployment.yaml's port
SCENARIO = "harness-requests-demo"  # this workload's isolated lane; the install bakes it into reef-pi
TOKEN = os.environ.get("REEF_TOKEN", "reef-local")  # matches deployment.yaml
MODEL = os.environ.get("REEF_UPSTREAM_MODEL", "gemma4:26b")  # run.sh exports the same default
# A local model reads the API skill and writes the change, then six pi episodes score it.
STEP_TIMEOUT_S = 1800.0
POLL_S = 5.0

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = HERE / "work"
# The install script writes the tree, reef-pi wrapper, and release metadata here.
INSTALL_ROOT = WORK / "harness"
CAPTURES = WORK / "captures"  # the wrapper's spool, beside the run so a show session's receipts can be read back
DEMOS = HERE / "demos"
RELEASE_FILE = ".reef-harness-release"

SHOW_PROMPTS = {
    "bugfix": "fix the bug in adder.py",
    "research": "what is the best known lower bound for sorting by comparisons, with a source",
}

#: The measurement's requests: short harness changes a skill or a rules entry answers, no extension.
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
    """One raw read of a route that answers text, not JSON: the install script, a step's page."""
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


def verdict_of(row):
    """The row's verdict as the page words it: pending, selected, rejected, skipped with its reason."""
    if row.get("pending"):
        return "pending"
    metrics = row.get("metrics") or {}
    if isinstance(metrics.get("selected"), bool):
        return "selected" if metrics["selected"] else "rejected"
    if metrics.get("skipped"):
        return f"skipped: {metrics['skipped']}"
    return str(row.get("operation") or "unknown")


def mutations_of(metrics):
    """The step's mutations as ``op id (kind)``: ``mutation`` for one, ``mutations`` for a composite."""
    single = metrics.get("mutation")
    many = [single] if isinstance(single, dict) else list(metrics.get("mutations") or [])
    return [f"{m.get('op')} {m.get('id')} ({(m.get('options') or {}).get('name', '?')})" for m in many]


def kinds_of(metrics):
    single = metrics.get("mutation")
    many = [single] if isinstance(single, dict) else list(metrics.get("mutations") or [])
    return sorted({str((m.get("options") or {}).get("name", "?")) for m in many})


def tally_parts(metrics):
    """The gate's (wins, losses, ties) over the tasks, or None when no gate ran.

    The score comparison selector records the three counts; ``selection:
    always`` records only the per task scores of both sides, so the counts
    come from comparing those."""
    if all(isinstance(metrics.get(key), int) for key in ("wins", "losses", "ties")):
        return metrics["wins"], metrics["losses"], metrics["ties"]
    scored = ((metrics.get("selection") or {}).get("evaluation") or {}).get("metrics") or {}
    candidate, current = scored.get("candidate_scores"), scored.get("current_scores")
    if not isinstance(candidate, list) or not isinstance(current, list) or len(candidate) != len(current):
        return None
    # An episode that failed scores None, which the score comparison ranks below every number.
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
    parts = tally_parts(metrics)
    return " / ".join(str(part) for part in parts) if parts else "- / - / -"


def _requires_of(items):
    return ", ".join(f"{item.get('name')} ({item.get('kind', '?')})" for item in items) or "nothing"


# -- the wrapper ----------------------------------------------------------------------------------------------


def _wrapper_env():
    """What reef-pi and the install script need: this interpreter as python3, with reef importable from the
    checkout, the token, and the spool beside the run."""
    env = dict(os.environ)
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = ":".join(part for part in (str(REPO), env.get("PYTHONPATH")) if part)
    env["REEF_TOKEN"] = TOKEN
    env["REEF_HARNESS_CAPTURES_DIR"] = str(CAPTURES)
    return env


def reef_pi(args, cwd=None):
    """One call of the installed wrapper, its lines echoed indented; the completed process."""
    done = subprocess.run(
        [str(INSTALL_ROOT / "reef-pi"), *args], cwd=cwd, env=_wrapper_env(), capture_output=True, text=True
    )
    for line in (done.stdout + done.stderr).splitlines():
        print("  " + line, flush=True)
    return done


def _installed_release():
    """The release id of the tree under work/harness, from the release metadata file the install script wrote."""
    try:
        return json.loads((INSTALL_ROOT / RELEASE_FILE).read_text(encoding="utf-8")).get("release_id")
    except (OSError, ValueError):
        return None


def install(release_id=None):
    """Install the head (or ``release_id``) through the install route into work/harness.

    Return the release id from its metadata file, or None when the script
    refused (an unmet requires item prints the setup list).
    """
    client = _client()
    _ensure_scenario(client)
    query = "?adapter=pi" + (f"&release_id={release_id}" if release_id else "")
    script = _fetch_text(f"/reef/harness/install{query}")
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "install.sh"
    path.write_text(script, encoding="utf-8")
    say(f"install: bash {path.relative_to(HERE)} {INSTALL_ROOT.relative_to(HERE)}")
    done = subprocess.run(["bash", str(path), str(INSTALL_ROOT)], env=_wrapper_env(), capture_output=True, text=True)
    for line in (done.stdout + done.stderr).splitlines():
        print("  " + line, flush=True)
    if done.returncode != 0:
        say(f"install refused (exit {done.returncode})")
        return None
    release = _installed_release()
    say(f"installed release {release} at {INSTALL_ROOT}")
    return release


def ask(text):
    """The request through the installed wrapper; the id of the training record the service accepted.

    Nothing runs before the ask: the wrapper posts the instruction with the
    installed release id and the originating session id (a spooled session or
    a fresh one), and the deployment's manual mode runs one step for it. A
    refusal (admission's screens, a mode that takes no instructions) is a
    stop, with the wrapper's line saying why."""
    say(f"ask: reef-pi harness {text!r}")
    done = reef_pi(["harness", text])
    match = re.search(r"training request (\S+) accepted", done.stdout)
    if match is None:
        raise SystemExit("the request was not accepted; the wrapper's lines above say why")
    return match.group(1)


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
            say(f"step {metrics.get('steps', '?')}: {verdict_of(row)} (release {release}){whose}")
        shown = max(shown, len(steps))
        for row in steps[before:]:
            if _request_text_of(row) == text:
                return rows, row
        time.sleep(POLL_S)
    return rows, None


def promote(client, rows, row, run_dir):
    """Save the pending step's page beside the run, print its URL, promote the release; the new head."""
    step = rows.index(row)
    page = f"/reef/harness/releases/{step}/page"
    saved = run_dir / f"step-{step}.html"
    saved.write_text(_fetch_text(page), encoding="utf-8")
    say(f"pending: the page says why and what changed: {SERVICE_URL}{page} (saved as {saved.relative_to(HERE)})")
    say("promote: the demo is scripted; a person reads the page first")
    answer, _ = client.post(f"/reef/scenarios/{SCENARIO}/promote", SCENARIO, {"release_id": row["release_id"]})
    say(f"promoted: the head is {answer['release_id']}")
    return answer["release_id"]


def setup_if_required(client):
    """``reef-pi setup --yes`` when the head's manifest names requires; the items and whether every one is met."""
    requires = _manifest(client).get("requires") or []
    say(f"requires: {_requires_of(requires)}")
    if not requires:
        return requires, True
    done = reef_pi(["setup", "--yes"])
    return requires, done.returncode == 0


# -- the show session -----------------------------------------------------------------------------------------


def _preview(arguments):
    """The first argument value of a tool call, cut short: enough to tell a test run from a file edit."""
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except ValueError:
        parsed = arguments
    if isinstance(parsed, dict) and parsed:
        # pi's write and edit put the content first and the path second; the path names the call better.
        named = next((parsed[key] for key in ("path", "file_path", "command") if key in parsed), None)
        parsed = named if named is not None else next(iter(parsed.values()))
    text = str(parsed if parsed is not None else "").replace("\n", " ").strip()
    return text[:57] + "..." if len(text) > 60 else text


def _tool_calls(turns):
    """The tool calls of a session in order, from each captured completion's message, as ``name(preview)``."""
    calls = []
    for turn in turns:
        for choice in (turn.get("response") or {}).get("choices") or []:
            message = choice.get("message") or choice
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                calls.append(f"{function.get('name')}({_preview(function.get('arguments'))})")
    return calls


def _last_answer(turns):
    for turn in reversed(turns):
        for choice in (turn.get("response") or {}).get("choices") or []:
            content = (choice.get("message") or choice).get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _take_show_spool(started_ns, run_dir):
    """The spool entry the show session wrote at exit, moved into the run directory; its turns.

    The wrapper spools every session's receipts for ``report`` to claim, and
    ``harness`` records the oldest spooled session as the session the request
    came from; the show session is shown, never reported, so its entry leaves the spool
    for the run directory, where it is the record this driver reads."""
    if not CAPTURES.is_dir():
        return []
    for path in sorted(CAPTURES.glob("*.pending.json"), reverse=True):
        # <hash of the scenario>-<20 digit ns stamp>-<uuid>.pending.json: the stamp is the second run of digits.
        match = re.search(r"-(\d{20})-", path.name)
        if match and int(match.group(1)) >= started_ns:
            target = run_dir / "show-captures.json"
            os.replace(path, target)
            return json.loads(target.read_text(encoding="utf-8")).get("turns") or []
    return []


def show(mode, run_dir):
    """One session on the installed tree in the mode's workspace; the tool calls in order and the final answer.

    The calls come from the receipts the wrapper's proxy captured, spooled at
    exit: pi's own session file lands under its default directory, outside
    the install root, so the spool is the record this driver reads."""
    workspace = run_dir / "workspace"
    if mode == "bugfix":
        # A copy: the session edits adder.py, and the committed fixture must fail again next time.
        shutil.copytree(DEMOS / "workspace", workspace)
    else:
        workspace.mkdir(parents=True)
    prompt = SHOW_PROMPTS[mode]
    say(f"show: reef-pi -p {prompt!r} in {workspace.relative_to(HERE)}")
    started_ns = time.time_ns()
    done = reef_pi(["-p", prompt], cwd=workspace)
    turns = _take_show_spool(started_ns, run_dir)
    calls = _tool_calls(turns)
    answer = done.stdout.strip() or _last_answer(turns)
    say(f"show: {len(turns)} model call(s), tool calls in order: {' -> '.join(calls) or 'none'}")
    say(f"show: final answer: {answer.splitlines()[-1][:160] if answer else '(none)'}")
    return {
        "prompt": prompt,
        "exit": done.returncode,
        "model_calls": len(turns),
        "tool_calls": calls,
        "answer": answer,
    }


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


def totals_of(results):
    """The measurement's counts: filed, answered (a mutation came back), admitted (the gate ran), won (more wins
    than losses in the recorded verdict), published, pending. Under ``selection: always`` a publish says nothing
    about the gate, so won and published are counted apart."""
    return {
        "filed": sum(1 for r in results if r.get("filed")),
        # A mutation came back: a step that skipped on a refusal or on a failed step carries none.
        "answered": sum(1 for r in results if r.get("filed") and r.get("kinds") not in (None, "-")),
        "admitted": sum(1 for r in results if r.get("wins") is not None),
        "won": sum(1 for r in results if r.get("wins") is not None and r["wins"] > (r.get("losses") or 0)),
        "published": sum(1 for r in results if r.get("verdict") == "selected"),
        "pending": sum(1 for r in results if r.get("verdict") == "pending"),
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
    verdict = verdict_of(row)
    mutations = mutations_of(metrics)
    say(f"verdict: {verdict}; W / L / T {tally(metrics)}; proposer {metrics.get('proposer_seconds', '-')} s")
    say(f"mutations: {'; '.join(mutations) or 'none'}")
    promoted = None
    if row.get("pending"):
        promoted = promote(client, rows, row, run_dir)
    requires, met = setup_if_required(client)
    result = {
        "request": text,
        "verdict": verdict,
        "wins_losses_ties": tally(metrics),
        "kinds": ", ".join(kinds_of(metrics)) or "-",
        "pending": bool(row.get("pending")),
        "promoted": promoted or "-",
        "requires": _requires_of(requires),
        "proposer_seconds": metrics.get("proposer_seconds", "-"),
    }
    if not met:
        record["result"] = {**result, "installed": "-", "setup": "unmet"}
        _write_record(record_path, record)
        raise SystemExit(2)
    if verdict == "selected" or promoted:
        installed = install()
        if installed is None:
            record["result"] = {**result, "installed": "-", "setup": "refused at install"}
            _write_record(record_path, record)
            raise SystemExit(2)
    else:
        say(f"the head did not move: release {installed_before} stays installed")
        installed = installed_before
    # From the harness call to the end of the install, or to the verdict when nothing new installed.
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
            results.append({"request": text, "filed": False, "verdict": str(exc)})
            continue
        _, row = wait_for_step(client, text, before, started + STEP_TIMEOUT_S)
        seconds = round(time.monotonic() - started, 1)
        if row is None:
            results.append(
                {"request": text, "id": request_id, "filed": True, "verdict": "no step", "seconds": seconds}
            )
            continue
        metrics = row.get("metrics") or {}
        parts = tally_parts(metrics)
        results.append(
            {
                "request": text,
                "id": request_id,
                "filed": True,
                "kinds": ", ".join(kinds_of(metrics)) or "-",
                "verdict": verdict_of(row),
                "wins": parts[0] if parts else None,
                "losses": parts[1] if parts else None,
                "ties": parts[2] if parts else None,
                "seconds": seconds,
                "row": row,
            }
        )
        say(f"verdict: {verdict_of(row)}; W / L / T {tally(metrics)}; {seconds} s")
        # The record after every request, so a run stopped midway still leaves its rows.
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    totals = totals_of(results)
    record["totals"] = totals
    _write_record(record_path, record)
    print()
    _print_table(
        ("Request", "Kind proposed", "Verdict", "W / L / T", "Seconds"),
        [
            (
                r["request"],
                r.get("kinds", "-"),
                r.get("verdict", "-"),
                f"{r['wins']} / {r['losses']} / {r['ties']}" if r.get("wins") is not None else "- / - / -",
                r.get("seconds", "-"),
            )
            for r in results
        ],
    )
    print()
    _print_table(("Filed", "Answered", "Admitted", "Won", "Published", "Pending"), [tuple(totals.values())])


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Harness requests v1 on reef-pi: the two demos and the measurement; start it through ./run.sh",
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("install", help="install the served head into work/harness (run.sh does this first)")
    modes.add_parser("bugfix", help="the bug fix flow demo: demos/bugfix.md")
    modes.add_parser("research", help="the research loop demo: demos/research.md")
    measured = modes.add_parser("measure", help="file the fixed request list and count what won the gate")
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
