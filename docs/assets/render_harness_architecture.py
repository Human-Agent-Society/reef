#!/usr/bin/env python3
"""Rebuild the harness evolution architecture diagrams: python3 docs/assets/render_harness_architecture.py

Writes harness-architecture-light.svg and harness-architecture-dark.svg: what
runs on a person's machine, what runs in Reef, and the routes between them.
The drawing is static. It embeds IBM Plex Sans and IBM Plex Mono from the
font cache of render_architecture.py.
"""

from __future__ import annotations

import argparse
import base64
from html import escape
from pathlib import Path

from render_architecture import arrow, cap, font_files, text

ROOT = Path(__file__).resolve().parent
W, H = 1556, 1460

THEMES = {
    "light": {
        "card": "#ffffff",
        "ink": "#1c1917",
        "muted": "#6b645d",
        "line": "#57534e",
        "machine": "#c2620a",
        "reef": "#2f62c4",
        "store": "#8a847d",
        "tint": 0.06,
        "band": 0.11,
    },
    "dark": {
        "card": "#1c1a17",
        "ink": "#f5f3f0",
        "muted": "#aaa29a",
        "line": "#c9c2ba",
        "machine": "#f0a050",
        "reef": "#86a8f0",
        "store": "#9c958e",
        "tint": 0.10,
        "band": 0.16,
    },
}

# Five columns. A box takes one column or two.
COL_W, GAP = 260, 36
X = [56 + k * (COL_W + GAP) for k in range(5)]
WIDE = 2 * COL_W + GAP

# Each row's top and height: A on the person's machine, B in Reef.
ROWS = {
    "A1": (132, 118),
    "A2": (298, 140),
    "B1": (574, 118),
    "B2": (748, 96),
    "B3": (932, 118),
    "B4": (1106, 140),
    "B5": (1290, 122),
}
A1, A1_H = ROWS["A1"]
A2, A2_H = ROWS["A2"]
B1, B1_H = ROWS["B1"]
B2, B2_H = ROWS["B2"]
B3, B3_H = ROWS["B3"]
B4, B4_H = ROWS["B4"]
B5, B5_H = ROWS["B5"]
MACHINE_TOP, MACHINE_BOTTOM = 92, 454
REEF_TOP, REEF_BOTTOM = 534, 1430
BAND_TOP, BAND_BOTTOM = 900, 1266

TITLE, BODY, MONO = 18, 14, 13

# The boxes: (column, row, columns spanned, kind, step number or None, title, lines).
# A line that starts with a backquote is code, in the mono face.
BOXES = [
    (0, "A1", 1, "machine", None, "Agent command", ["a claude or pi command, first", "on PATH, that runs", "`reef-client harness run"]),
    (2, "A1", 1, "machine", None, "Layer install", ["`~/.local/share/reef-client/", "`harness/<scenario-name>", "the head, rendered for the agent"]),
    (3, "A1", 1, "machine", None, "Your own config", ["`~/.claude, ~/.pi, ~/.codex", "the agent reads it as always;", "Reef never reads or writes it"]),
    (0, "A2", 2, "machine", None, "reef-client harness", []),
    (
        2, "A2", 2, "machine", 1, "Your agent",
        [
            "`claude · pi · opencode · codex",
            "loads your own config first, then Reef's layer over it",
            "you work with it as always; its model calls go to Reef",
            "you score a session with a report",
        ],
    ),
    (4, "A2", 1, "store", None, "Model provider", ["`REEF_UPSTREAM_URL", "an OpenAI or Anthropic API,", "or a compatible one"]),
    (0, "B1", 1, "reef", None, "Client install", ["`GET /reef/harness", "layer_install: the head as", "files, args and env of the layer"]),
    (1, "B1", 1, "reef", None, "Client import", ["`POST /reef/harness/import", "merge_trees merges the changes", "onto the head, file by file"]),
    (2, "B1", 1, "reef", 3, "Feedback", ["`POST /reef/report", "a score and feedback for", "the calls of one session"]),
    (3, "B1", 1, "reef", 2, "Model proxy", ["`/reef/scenarios/<s>/v1/*", "forwards each call as sent,", "records it with its reply"]),
    (
        0, "B2", 2, "store", None, "Releases",
        [
            "every published tree, in a Git repository per scenario;",
            "ScenarioCommitter writes them, and the newest is the head",
        ],
    ),
    (
        2, "B2", 2, "store", None, "Records",
        ["each call with its reply, and each report, per scenario;", "a report names its calls by agent_record_id"],
    ),
    (4, "B3", 1, "reef", 4, "Batch", ["the pipeline selects reports", "and makes one job of", "`evolution.batch_size"]),
    (3, "B3", 1, "reef", None, "Workspace", ["a copy of the head in harness/,", "and the job's traces with", "their scores and feedback"]),
    (2, "B3", 1, "reef", 5, "Method", ["`step(context)", "edits the copy; its services:", "models, calls, check, evaluate"]),
    (1, "B3", 1, "reef", 6, "Check", ["`validate_tree", "refuses a file that no part", "takes, or that its format refuses"]),
    (
        2, "B4", 2, "reef", 7, "Evaluation",
        [
            "`HarnessEpisodeEvaluator",
            "renders the candidate and the head with the served model,",
            "runs each task as an episode, scores it with the EpisodeScorer;",
            "the history keeps each score, so an episode runs once",
        ],
    ),
    (
        4, "B4", 1, "reef", None, "Episodes",
        ["`run.episode", "the pinned agent in a root of", "its own; its model calls use", "the scenario's evaluation route"],
    ),
    (1, "B4", 1, "reef", None, "Policy", ["`ComparePolicy, FloorPolicy", "turns the scores into a", "SelectionDecision"]),
    (
        0, "B4", 1, "reef", 8, "Release rule",
        ["`ReleaseRules", "selected: published", "needs review: held", "rejected: not published"],
    ),
]  # fmt: skip

# reef-client's commands, one per row; a continued description has no command.
CLIENT_COMMANDS = [
    ("connect", "installs the agent at Reef's version, adds its command"),
    ("run", "installs the newest release and starts the agent on it;"),
    ("", "sends back what the session changed, in the background"),
    ("sync", "sends changes and installs the head, without the agent"),
]

# The adapter's sections, one per column: the key, then what it declares.
ADAPTER_SECTIONS = [
    ("evolve", "the kinds of file Reef may", "change, at the agent's paths"),
    ("dirs, sessions", "the agent's directories", "and its session files"),
    ("model", "how the agent reaches a", "model; episodes bind it"),
    ("run.episode", "one evaluation episode:", "its command and env"),
    ("run.layer, install", "the layer, and the agent's", "version, for the client"),
]


def write(x, y, s, size, fill, weight=500, anchor="start", mono=False):
    return text(x, y, escape(s), size, 400 if mono else weight, fill, anchor, mono=mono)


def note(p, x, y, s, anchor="start", mono=False, color=None):
    return write(x, y, s, MONO, color or p["muted"], anchor=anchor, mono=mono)


def zone(p, x0, y0, x1, y1, kind, opacity, dashed=False):
    dash = ' stroke-dasharray="5 5"' if dashed else ""
    return (
        f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" rx="14" fill="{p[kind]}" '
        f'fill-opacity="{opacity}" stroke="{p[kind]}" stroke-opacity="0.45" stroke-width="1"{dash}/>'
    )


def card(p, x, y, w, h, kind, title, lines, badge=None, dashed=False):
    """A box: an optional step number, a title, then its lines."""
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{p["card"]}" '
        f'stroke="{p[kind]}" stroke-width="1.4"{dash}/>'
    ]
    title_x = x + 18
    if badge is not None:
        out.append(f'<circle cx="{x + 29}" cy="{y + 28}" r="12" fill="{p[kind]}"/>')
        out.append(write(x + 29, y + 33, str(badge), 14, p["card"], 600, "middle"))
        title_x = x + 50
    out.append(write(title_x, y + 34, title, TITLE, p["ink"], 600))
    for k, line in enumerate(lines):
        if line.startswith("`"):
            out.append(write(x + 18, y + 58 + k * 21, line[1:], MONO, p["ink"], mono=True))
        else:
            out.append(write(x + 18, y + 58 + k * 21, line, BODY, p["muted"]))
    return "".join(out)


def path(p, points, color=None, width=1.3):
    """A line through `points`, with an arrowhead at the last point."""
    color = color or p["line"]
    (x0, y0), (x1, y1) = points[-2], points[-1]
    direction = ("down" if y1 > y0 else "up") if x1 == x0 else ("right" if x1 > x0 else "left")
    return (
        f'<path d="M{" L".join(f"{x} {y}" for x, y in points)}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'
    ) + arrow(x1, y1, direction, color)


def render(theme: str, fonts) -> str:
    p = THEMES[theme]
    faces = "".join(
        f"@font-face{{font-family:'{family}';font-weight:{weight};font-style:normal;"
        f"src:url(data:font/woff2;base64,{base64.b64encode(data).decode()}) format('woff2')}}\n"
        for family, weight, data in fonts
    )
    # No background rectangle: the drawing sits on the page's own light or dark ground.
    o = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
            f"aria-label=\"Harness evolution in Reef. On the person's machine, reef-client installs the head as a "
            f"layer over the person's own configuration and starts the agent. The agent's model calls go through "
            f"Reef's model proxy, which records them, and a report scores a session. In Reef, the pipeline batches "
            f"the reports, a method edits a copy of the head, the check and the evaluation test the candidate, and "
            f"the release rule publishes a selected candidate as the new head. After a session, the client sends "
            f'the changed files back, and Reef merges them onto the head.">'
        ),
        f"<style>\n{faces}</style>",
        write(40, 46, "Harness evolution: what runs where", 24, p["ink"], 600),
        write(40, 72, "Numbers 1 to 8 follow one session into one training step.", 15, p["muted"]),
    ]
    legend_x = W - 40
    for label, kind in reversed([("your machine", "machine"), ("Reef", "reef"), ("storage and outside", "store")]):
        legend_x -= len(label) * 7.2
        o.append(write(legend_x, 46, label, 14, p["muted"]))
        o.append(
            f'<rect x="{legend_x - 22}" y="35" width="14" height="14" rx="3" fill="{p[kind]}" '
            f'fill-opacity="0.18" stroke="{p[kind]}" stroke-width="1.2"/>'
        )
        legend_x -= 46

    # The zones, under the boxes: the person's machine, the provider outside it, Reef, and one training step.
    o.append(zone(p, 40, MACHINE_TOP, X[3] + COL_W + 14, MACHINE_BOTTOM, "machine", p["tint"]))
    o.append(cap(56, 118, "your machine", p["machine"]))
    o.append(zone(p, X[4] - 14, MACHINE_TOP, W - 40, MACHINE_BOTTOM, "store", p["tint"], dashed=True))
    o.append(cap(X[4], 118, "outside", p["store"]))
    o.append(zone(p, 40, REEF_TOP, W - 40, REEF_BOTTOM, "reef", p["tint"]))
    o.append(cap(56, REEF_TOP + 26, "reef service", p["reef"]))
    o.append(zone(p, 48, BAND_TOP, W - 48, BAND_BOTTOM, "reef", p["band"]))
    o.append(cap(72, B3 + 30, "one training step", p["reef"]))
    o.append(note(p, 72, B3 + 54, "the trainer runs it"))
    o.append(note(p, 72, B3 + 72, "through HarnessBackend"))

    for column, row, span, kind, badge, title, lines in BOXES:
        top, height = ROWS[row]
        o.append(card(p, X[column], top, WIDE if span == 2 else COL_W, height, kind, title, lines, badge))
    for k, (command, description) in enumerate(CLIENT_COMMANDS):
        o.append(write(X[0] + 18, A2 + 60 + k * 21, command, MONO, p["ink"], mono=True))
        o.append(write(X[0] + 104, A2 + 60 + k * 21, description, BODY, p["muted"]))
    o.append(card(p, X[0], B5, X[4] + COL_W - X[0], B5_H, "reef", "Adapter", [], dashed=True))
    o.append(write(X[0] + 104, B5 + 34, "reef/harness/adapters/<agent>/adapter.yaml", MONO, p["ink"], mono=True))
    o.append(
        write(
            X[2] + 120, B5 + 34,
            "Reef reads it. The client gets run.layer and install in the head's client form.", BODY, p["muted"],
        )
    )  # fmt: skip
    for k, (key, line1, line2) in enumerate(ADAPTER_SECTIONS):
        o.append(write(X[k] + 18, B5 + 66, key, MONO, p["ink"], mono=True))
        o.append(write(X[k] + 18, B5 + 87, line1, BODY, p["muted"]))
        o.append(write(X[k] + 18, B5 + 108, line2, BODY, p["muted"]))

    # On the machine: the command runs the client; the client writes the install and starts the agent.
    shim_x, install_x, layer_x, config_x = X[0] + 130, X[1] + 130, X[2] + 130, X[3] + 130
    o.append(path(p, [(shim_x, A1 + A1_H), (shim_x, A2)]))
    o.append(note(p, shim_x + 10, A1 + A1_H + 29, "runs"))
    o.append(path(p, [(install_x, A2), (install_x, A1 + 59), (X[2], A1 + 59)]))
    o.append(note(p, install_x + 12, A1 + 33, "writes the files"))
    o.append(note(p, install_x + 12, A1 + 50, "that differ"))
    o.append(path(p, [(X[0] + WIDE, A2 + 40), (X[2], A2 + 40)]))
    o.append(path(p, [(layer_x, A1 + A1_H), (layer_x, A2)]))
    o.append(path(p, [(layer_x + 12, A2), (layer_x + 12, A1 + A1_H)], p["muted"]))
    o.append(note(p, layer_x + 26, A1 + A1_H + 20, "loads it as a layer"))
    o.append(note(p, layer_x + 26, A1 + A1_H + 38, "the session's edits"))
    o.append(path(p, [(config_x, A1 + A1_H), (config_x, A2)]))
    o.append(note(p, config_x + 10, A1 + A1_H + 29, "loads first"))

    # The routes between the machine and Reef: each request down, with its answer up beside it.
    top, bottom = A2 + A2_H, B1
    request_y, answer_y = MACHINE_BOTTOM + 24, REEF_TOP - 14
    get_x = X[0] + COL_W - 30
    o.append(path(p, [(get_x, top), (get_x, bottom)]))
    o.append(path(p, [(get_x + 12, bottom), (get_x + 12, top)], p["muted"]))
    o.append(note(p, get_x - 10, request_y, "GET /reef/harness?form=client", "end", mono=True, color=p["ink"]))
    o.append(note(p, get_x - 10, answer_y, "files, args, env, install", "end", mono=True))
    for x, request, answer in [
        (X[1] + 36, "POST /reef/harness/import", "merged head, or conflicts"),
        (X[3] + 36, "POST /reef/scenarios/<s>/v1/*", "reply, x-reef-agent-record-id"),
    ]:
        o.append(path(p, [(x, top), (x, bottom)]))
        o.append(path(p, [(x + 12, bottom), (x + 12, top)], p["muted"]))
        o.append(note(p, x + 26, request_y, request, mono=True, color=p["ink"]))
        o.append(note(p, x + 26, answer_y, answer, mono=True))
    report_x = X[2] + 36
    o.append(path(p, [(report_x, top), (report_x, bottom)]))
    o.append(note(p, report_x + 14, request_y, "POST /reef/report", mono=True, color=p["ink"]))
    o.append(note(p, report_x + 14, answer_y, "references, score, feedback", mono=True))
    provider_x = X[4] + 130
    o.append(path(p, [(X[3] + COL_W, B1 + 40), (provider_x, B1 + 40), (provider_x, A2 + A2_H)]))
    o.append(note(p, provider_x + 10, MACHINE_BOTTOM + 30, "forwards each call"))
    o.append(note(p, provider_x + 10, MACHINE_BOTTOM + 48, "and returns its reply"))

    # In Reef: the routes read and write the storage.
    for x, s, up in [
        (X[0] + 130, "the head", True),
        (X[1] + 130, "commit: import", False),
        (X[2] + 130, "reports", False),
        (X[3] + 130, "calls", False),
    ]:
        o.append(path(p, [(x, B2), (x, B1 + B1_H)] if up else [(x, B1 + B1_H), (x, B2)]))
        o.append(note(p, x + 10, B1 + B1_H + 32, s))

    # One training step: batch, workspace, method, check, evaluation, policy, release rule.
    batch_x = X[4] + 130
    o.append(path(p, [(X[3] + COL_W, B2 + 48), (batch_x, B2 + 48), (batch_x, B3)]))
    o.append(note(p, batch_x + 10, BAND_TOP - 12, "reports and"))
    o.append(note(p, batch_x + 10, BAND_TOP + 6, "their calls"))
    head_route = [
        (X[1] + 200, B2 + B2_H),
        (X[1] + 200, B2 + B2_H + 32),
        (X[3] + 130, B2 + B2_H + 32),
        (X[3] + 130, B3),
    ]
    o.append(path(p, head_route))
    o.append(note(p, X[1] + 214, B2 + B2_H + 25, "the head"))
    o.extend(path(p, [(X[k], B3 + 59), (X[k - 1] + COL_W, B3 + 59)]) for k in (4, 3, 2))
    check_x, evaluation_x = X[1] + 130, X[2] + 72
    o.append(path(p, [(check_x, B3 + B3_H), (check_x, B4 - 28), (evaluation_x, B4 - 28), (evaluation_x, B4)]))
    o.append(note(p, check_x + 12, B4 - 34, "the candidate"))
    o.append(path(p, [(X[2] + WIDE, B4 + 50), (X[4], B4 + 50)]))
    o.append(path(p, [(X[4], B4 + 70), (X[2] + WIDE, B4 + 70)], p["muted"]))
    o.append(path(p, [(X[2], B4 + 70), (X[1] + COL_W, B4 + 70)]))
    o.append(path(p, [(X[1], B4 + 70), (X[0] + COL_W, B4 + 70)]))
    commit_x = X[0] + COL_W - 40
    o.append(path(p, [(commit_x, B4), (commit_x, B2 + B2_H)], p["reef"], width=1.8))
    o.append(note(p, commit_x - 10, B4 - 36, "published:", "end", color=p["reef"]))
    o.append(note(p, commit_x - 10, B4 - 18, "the new head", "end", color=p["reef"]))

    o.append("</svg>")
    return "\n".join(o)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font-cache", type=Path, default=Path.home() / ".cache" / "reef-fonts")
    args = parser.parse_args()
    fonts = font_files(args.font_cache)
    for theme in THEMES:
        target = ROOT / f"harness-architecture-{theme}.svg"
        target.write_text(render(theme, fonts))
        print(f"wrote {target.relative_to(ROOT.parent.parent)} ({target.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
