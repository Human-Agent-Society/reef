#!/usr/bin/env python3
"""Rebuild the harness evolution architecture diagrams: python3 docs/assets/render_harness_architecture.py

Writes harness-architecture-light.svg and harness-architecture-dark.svg: what
runs on a person's machine, what runs in Reef, and the routes between them, with
the two ways a step gets its traces: online from sessions, offline from tasks.
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
W, H = 1372, 1240

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

# Five columns. A box takes one column or two, and every box is one row high.
COL_W, GAP, BOX_H = 220, 40, 76
X = [56 + k * (COL_W + GAP) for k in range(5)]
WIDE = 2 * COL_W + GAP
# The workspace stops short of the last column's right edge, so the tasks reach the runs beside it.
WORKSPACE_W = X[4] + 150 - X[3]

# Each row's top: A on the person's machine, B in Reef.
ROWS = {"A1": 124, "A2": 244, "B1": 446, "B2": 574, "B3": 738, "B4": 870, "B5": 990}
A1, A2, B1, B2, B3, B4, B5 = ROWS.values()
MACHINE_TOP, MACHINE_BOTTOM = 84, 336
REEF_TOP, REEF_BOTTOM = 406, 1084
BAND_TOP, BAND_BOTTOM = 706, 966
# The two ways a step gets its traces, under Reef; the guide's table below the figure says the rest.
MODES_TOP = 1136
MODE_W = (X[4] + COL_W - X[0] - GAP) // 2

TITLE, BODY, MONO = 18, 14, 13

# The boxes: (column, row, width, kind, step number or None, title, line or "").
# A line that starts with a backquote is code, in the mono face.
BOXES = [
    (0, "A1", COL_W, "machine", None, "Agent command", "runs reef-client"),
    (2, "A1", COL_W, "machine", None, "Plugin install", "the head, as the agent's plugin"),
    (3, "A1", COL_W, "machine", None, "Your own config", "Reef never reads it"),
    (0, "A2", WIDE, "machine", None, "reef-client", "`connect · run · sync"),
    (2, "A2", WIDE, "machine", 1, "Your agent", "`claude · pi · opencode · codex"),
    (4, "A2", COL_W, "store", None, "Model provider", "OpenAI or Anthropic API"),
    (0, "B1", COL_W, "reef", None, "Client install", "the head as a plugin"),
    (1, "B1", COL_W, "reef", None, "Client sync", "merges onto the active release"),
    (2, "B1", COL_W, "reef", 3, "Feedback", "a score per session"),
    (3, "B1", COL_W, "reef", 2, "Model proxy", "forwards and records"),
    (0, "B2", WIDE, "store", None, "Releases", "every tree, in Git; the newest is the head"),
    (2, "B2", WIDE, "store", None, "Records", "raw traces: model calls and reports"),
    (4, "B2", COL_W, "store", None, "Tasks", "Harbor tasks: train · eval"),
    (3, "B3", WORKSPACE_W, "reef", None, "Workspace", "`harness/ · material/traces/ · state/"),
    (2, "B3", COL_W, "reef", 5, "Method", "`step(context)"),
    (1, "B3", COL_W, "reef", 6, "Check", "`validate_tree"),
    (2, "B4", WIDE, "reef", 7, "Evaluation", "runs each task as a run, scores it"),
    (4, "B4", COL_W, "reef", None, "Runs", "`run.task"),
    (1, "B4", COL_W, "reef", None, "Policy", "compare or floor"),
    (0, "B4", COL_W, "reef", 8, "Activation rule", "serve or hold"),
]


def write(x, y, s, size, fill, weight=500, anchor="start", mono=False):
    return text(x, y, escape(s), size, 400 if mono else weight, fill, anchor, mono=mono)


def zone(p, x0, y0, x1, y1, kind, opacity, dashed=False):
    dash = ' stroke-dasharray="5 5"' if dashed else ""
    return (
        f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" rx="14" fill="{p[kind]}" '
        f'fill-opacity="{opacity}" stroke="{p[kind]}" stroke-opacity="0.45" stroke-width="1"{dash}/>'
    )


def badge(p, x, y, number, kind):
    """A step number in a filled circle centered at (x, y)."""
    return f'<circle cx="{x}" cy="{y}" r="12" fill="{p[kind]}"/>' + write(
        x, y + 5, str(number), 14, p["card"], 600, "middle"
    )


def card(p, x, y, w, kind, title, line, number=None, dashed=False):
    """A box: an optional step number, a title, and one line."""
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{BOX_H}" rx="9" fill="{p["card"]}" '
        f'stroke="{p[kind]}" stroke-width="1.4"{dash}/>'
    ]
    title_x = x + 18
    if number is not None:
        out.append(badge(p, x + 29, y + 27, number, kind))
        title_x = x + 50
    out.append(write(title_x, y + 33, title, TITLE, p["ink"], 600))
    if line.startswith("`"):
        out.append(write(x + 18, y + 57, line[1:], MONO, p["ink"], mono=True))
    else:
        out.append(write(x + 18, y + 57, line, BODY, p["muted"]))
    return "".join(out)


def mode(p, x, kind, title, line):
    """One way a step gets its traces: a title in the mode's color, and where the traces come from."""
    return (
        f'<rect x="{x}" y="{MODES_TOP}" width="{MODE_W}" height="{BOX_H}" rx="9" fill="{p["card"]}" '
        f'stroke="{p[kind]}" stroke-width="1.4"/>'
        + write(x + 18, MODES_TOP + 33, title, TITLE, p[kind], 600)
        + write(x + 18, MODES_TOP + 57, line, BODY, p["muted"])
    )


def path(p, points, color=None, width=1.3):
    """A line through `points`, with an arrowhead at the last point."""
    color = color or p["line"]
    (x0, y0), (x1, y1) = points[-2], points[-1]
    direction = ("down" if y1 > y0 else "up") if x1 == x0 else ("right" if x1 > x0 else "left")
    return (
        f'<path d="M{" L".join(f"{x} {y}" for x, y in points)}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'
    ) + arrow(x1, y1, direction, color)


def pair(p, x, y0, y1):
    """A request from y0 to y1 at x, and its answer back beside it."""
    return path(p, [(x, y0), (x, y1)]) + path(p, [(x + 12, y1), (x + 12, y0)], p["muted"])


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
            f"aria-label=\"Harness evolution in Reef. On the person's machine, reef-client installs the head as the "
            f"agent's own plugin, beside the person's own configuration, and starts the agent. The agent's model calls go through "
            f"Reef's model proxy, which records them, and a report scores a session. In a training step, the "
            f"workspace gets the head and a batch of raw traces, a method edits the copy, the check and the "
            f"evaluation test the candidate, a selected candidate is a new release, and the activation rule serves it or holds it for a person. "
            f"After a session, the client sends the changed files back, and Reef merges them onto the active release. "
            f"An offline deployment gets its traces from uploaded Harbor tasks instead: the train tasks run on the "
            f'head as runs, and the eval tasks score the candidates.">'
        ),
        f"<style>\n{faces}</style>",
        write(40, 44, "Harness evolution: what runs where", 24, p["ink"], 600),
        write(
            40,
            68,
            "Numbers 1 to 8 follow one session into one training step.",
            15,
            p["muted"],
        ),
    ]
    legend_x = W - 40
    for label, kind in reversed([("your machine", "machine"), ("Reef", "reef"), ("storage and outside", "store")]):
        legend_x -= len(label) * 7.2
        o.append(write(legend_x, 44, label, 14, p["muted"]))
        o.append(
            f'<rect x="{legend_x - 22}" y="33" width="14" height="14" rx="3" fill="{p[kind]}" '
            f'fill-opacity="0.18" stroke="{p[kind]}" stroke-width="1.2"/>'
        )
        legend_x -= 46

    # The zones, under the boxes: the person's machine, the provider outside it, Reef, and one training step.
    o.append(zone(p, 40, MACHINE_TOP, X[3] + COL_W + 14, MACHINE_BOTTOM, "machine", p["tint"]))
    o.append(cap(56, MACHINE_TOP + 24, "your machine", p["machine"]))
    o.append(zone(p, X[4] - 14, MACHINE_TOP, W - 40, MACHINE_BOTTOM, "store", p["tint"], dashed=True))
    o.append(cap(X[4], MACHINE_TOP + 24, "outside", p["store"]))
    o.append(zone(p, 40, REEF_TOP, W - 40, REEF_BOTTOM, "reef", p["tint"]))
    o.append(cap(56, REEF_TOP + 24, "reef service", p["reef"]))
    o.append(zone(p, 48, BAND_TOP, W - 48, BAND_BOTTOM, "reef", p["band"]))
    o.append(cap(72, B3 + 44, "one training step", p["reef"]))

    for column, row, width, kind, number, title, line in BOXES:
        o.append(card(p, X[column], ROWS[row], width, kind, title, line, number))
    o.append(card(p, X[0], B5, X[4] + COL_W - X[0], "reef", "Adapter", "", dashed=True))
    o.append(write(X[0] + 104, B5 + 33, "reef/harness/adapters/<agent>/adapter.yaml", MONO, p["ink"], mono=True))
    o.append(
        write(
            X[0] + 18,
            B5 + 57,
            "evolve · dirs · sessions · model · run.task · run.layer",
            MONO,
            p["muted"],
            mono=True,
        )
    )

    # On the machine: the command runs the client; the client writes the install and starts the agent.
    middle_a1, middle_a2 = A1 + BOX_H // 2, A2 + BOX_H // 2
    o.append(path(p, [(X[0] + 110, A1 + BOX_H), (X[0] + 110, A2)]))
    o.append(path(p, [(X[1] + 110, A2), (X[1] + 110, middle_a1), (X[2], middle_a1)]))
    o.append(path(p, [(X[1] + COL_W, middle_a2), (X[2], middle_a2)]))
    o.append(path(p, [(X[2] + 110, A1 + BOX_H), (X[2] + 110, A2)]))
    o.append(path(p, [(X[2] + 122, A2), (X[2] + 122, A1 + BOX_H)], p["muted"]))
    o.append(path(p, [(X[3] + 110, A1 + BOX_H), (X[3] + 110, A2)]))

    # The routes between the machine and Reef: each request down, with its answer up beside it.
    top, bottom, route_y = A2 + BOX_H, B1, MACHINE_BOTTOM + 34
    o.append(pair(p, X[0] + COL_W - 24, top, bottom))
    o.append(write(X[0] + COL_W - 34, route_y, "GET /reef/harness", MONO, p["ink"], anchor="end", mono=True))
    o.append(pair(p, X[1] + 30, top, bottom))
    o.append(write(X[1] + 56, route_y, "POST /reef/harness/import", MONO, p["ink"], mono=True))
    o.append(path(p, [(X[2] + 30, top), (X[2] + 30, bottom)]))
    o.append(write(X[2] + 44, route_y, "POST /reef/report", MONO, p["ink"], mono=True))
    o.append(pair(p, X[3] + 30, top, bottom))
    o.append(write(X[3] + 56, route_y, "POST /reef/scenarios/<s>/v1/*", MONO, p["ink"], mono=True))
    provider_y = REEF_TOP - 14
    o.append(path(p, [(X[3] + 190, B1), (X[3] + 190, provider_y), (X[4] + 60, provider_y), (X[4] + 60, A2 + BOX_H)]))
    # The tasks come from outside the machine's session: whoever runs the deployment uploads them.
    upload_x = X[4] + 170
    o.append(write(upload_x, MACHINE_BOTTOM + 24, "your Harbor tasks", BODY, p["muted"], anchor="middle"))
    o.append(path(p, [(upload_x, MACHINE_BOTTOM + 32), (upload_x, B2)]))
    o.append(write(upload_x - 10, B1 + 30, "POST /reef/scenarios/", MONO, p["ink"], anchor="end", mono=True))
    o.append(write(upload_x - 10, B1 + 48, "<s>/tasks", MONO, p["ink"], anchor="end", mono=True))

    # In Reef: the routes read and write the storage.
    o.append(path(p, [(X[0] + 110, B2), (X[0] + 110, B1 + BOX_H)]))
    o.extend(path(p, [(x, B1 + BOX_H), (x, B2)]) for x in (X[1] + 110, X[2] + 110, X[3] + 110))

    # One training step: the workspace gets the head and a batch of raw traces; then the method, the check, the
    # evaluation, the policy and the activation rule.
    route_y = B2 + BOX_H + 32
    o.append(path(p, [(X[1] + 150, B2 + BOX_H), (X[1] + 150, route_y), (X[3] + 40, route_y), (X[3] + 40, B3)]))
    o.append(write(X[1] + 164, route_y - 8, "the head", MONO, p["muted"], mono=True))
    traces_x = X[3] + 170
    o.append(path(p, [(traces_x, B2 + BOX_H), (traces_x, B3)]))
    o.append(badge(p, traces_x + 26, route_y - 2, 4, "reef"))
    o.append(write(traces_x + 46, route_y + 3, "online: a batch of raw traces", BODY, p["ink"]))
    # Offline: the tasks run as runs, and the train tasks' runs are the workspace's traces.
    tasks_x, runs_x = X[4] + 190, X[4] + 60
    o.append(path(p, [(tasks_x, B2 + BOX_H), (tasks_x, B4)]))
    o.append(write(tasks_x - 8, B4 - 22, "tasks", MONO, p["muted"], anchor="end", mono=True))
    o.append(path(p, [(runs_x, B4), (runs_x, B3 + BOX_H)]))
    o.append(write(runs_x - 12, B4 - 22, "offline: train-task runs", BODY, p["ink"], anchor="end"))
    o.extend(path(p, [(X[k], B3 + BOX_H // 2), (X[k - 1] + COL_W, B3 + BOX_H // 2)]) for k in (3, 2))
    check_x, evaluation_x, elbow_y = X[1] + 110, X[2] + 60, B4 - 28
    o.append(path(p, [(check_x, B3 + BOX_H), (check_x, elbow_y), (evaluation_x, elbow_y), (evaluation_x, B4)]))
    o.append(write(check_x + 12, elbow_y - 7, "the candidate", MONO, p["muted"], mono=True))
    o.append(path(p, [(X[3] + COL_W, B4 + 30), (X[4], B4 + 30)]))
    o.append(path(p, [(X[4], B4 + 46), (X[3] + COL_W, B4 + 46)], p["muted"]))
    o.extend(path(p, [(X[k], B4 + BOX_H // 2), (X[k - 1] + COL_W, B4 + BOX_H // 2)]) for k in (2, 1))
    commit_x = X[0] + COL_W - 16
    o.append(path(p, [(commit_x, B4), (commit_x, B2 + BOX_H)], p["reef"], width=1.8))
    o.append(write(commit_x - 10, elbow_y - 7, "published", MONO, p["reef"], anchor="end", mono=True))

    # The two ways a step gets its traces.
    o.append(cap(56, MODES_TOP - 18, "two ways to evolve", p["muted"]))
    o.append(mode(p, X[0], "machine", "Online: learn from sessions", "the scored sessions are the traces"))
    o.append(
        mode(
            p,
            X[0] + MODE_W + GAP,
            "reef",
            "Offline: learn from uploaded tasks",
            "the train tasks' runs are the traces",
        )
    )

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
