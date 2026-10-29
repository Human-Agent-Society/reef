#!/usr/bin/env python3
"""Rebuild the figures of the harness developer guides: python3 docs/assets/render_harness_guides.py

Writes, each in a light and a dark file:

- harness-method-step-*.svg: one training step from the method's view, for
  "Write a harness method";
- harness-adapter-modes-*.svg: the three ways one adapter.yaml runs an agent,
  for "Harness adapters".

The boxes, colors and fonts are those of render_harness_architecture.py.
"""

from __future__ import annotations

import argparse
import base64
from itertools import pairwise
from pathlib import Path

from render_architecture import cap, font_files
from render_harness_architecture import BODY, BOX_H, MONO, THEMES, card, pair, path, write, zone

ROOT = Path(__file__).resolve().parent


def svg(p, fonts, width: int, height: int, label: str, title: str, subtitle: str, legend) -> list[str]:
    """The opening of one figure: the root element, the embedded fonts, the title, and the color legend."""
    faces = "".join(
        f"@font-face{{font-family:'{family}';font-weight:{weight};font-style:normal;"
        f"src:url(data:font/woff2;base64,{base64.b64encode(data).decode()}) format('woff2')}}\n"
        for family, weight, data in fonts
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{label}">',
        f"<style>\n{faces}</style>",
        write(40, 44, title, 24, p["ink"], 600),
        write(40, 68, subtitle, 15, p["muted"]),
    ]
    legend_x = width - 40
    for name, kind in reversed(legend):
        legend_x -= len(name) * 7.2
        out.append(write(legend_x, 44, name, 14, p["muted"]))
        out.append(
            f'<rect x="{legend_x - 22}" y="33" width="14" height="14" rx="3" fill="{p[kind]}" '
            f'fill-opacity="0.18" stroke="{p[kind]}" stroke-width="1.2"/>'
        )
        legend_x -= 46
    return out


# One training step, from the method's view: three columns over one band.
STEP_W, STEP_H = 1240, 628
STEP_X = (56, 452, 868)
STEP_COL = 316
STEP_ROWS = (132, 224, 316)
BAND_TOP = 452


def method_step(theme: str, fonts) -> str:
    p = THEMES[theme]
    o = svg(
        p,
        fonts,
        STEP_W,
        STEP_H,
        "One training step from the method's view. Reef writes the workspace: a copy of the head in harness, "
        "the step's material, and the method's state from the last step. Reef calls step(context) once; the "
        "method reads the workspace, calls the services, and writes the candidate tree, its state and its output. "
        "Reef then checks the candidate and evaluates it; a selected candidate is a release, which the activation "
        "rule serves or holds for a person.",
        "One training step, from the method's view",
        "Numbers 5 to 8 are the steps of the architecture figure in Evolve your harness.",
        [("you write", "machine"), ("Reef", "reef"), ("files", "store")],
    )
    left, middle, right = STEP_X
    zone_bottom = STEP_ROWS[-1] + BOX_H + 20
    o.append(zone(p, left - 16, 96, left + STEP_COL + 16, zone_bottom, "store", p["tint"]))
    o.append(cap(left, 120, "the workspace Reef writes", p["store"]))
    o.append(zone(p, middle - 16, 96, middle + STEP_COL + 16, zone_bottom, "machine", p["tint"]))
    o.append(cap(middle, 120, "your method", p["machine"]))
    o.append(zone(p, right - 16, 96, right + STEP_COL + 16, zone_bottom, "store", p["tint"]))
    o.append(cap(right, 120, "what Reef reads back", p["store"]))

    inputs = [
        ("harness/", "a copy of the head tree"),
        ("material/", "traces · tasks · host.json"),
        ("state/", "your files from the last step"),
    ]
    outputs = [
        ("harness/", "the tree you leave: the candidate"),
        ("state/", "kept for the next step"),
        ("output/", "summary.md · notes.json"),
    ]
    for row, (title, line) in zip(STEP_ROWS, inputs, strict=True):
        o.append(card(p, left, row + 8, STEP_COL, "store", title, line))
    for row, (title, line) in zip(STEP_ROWS, outputs, strict=True):
        o.append(card(p, right, row + 8, STEP_COL, "store", title, line))
    o.append(card(p, middle, STEP_ROWS[0] + 8, STEP_COL, "machine", "Method", "`step(context)", number=5))
    o.append(card(p, middle, STEP_ROWS[-1] + 8, STEP_COL, "reef", "Services", "`models · check · evaluate"))

    # The workspace goes into the method; what the method writes comes back out.
    middle_y = [row + 8 + BOX_H // 2 for row in STEP_ROWS]
    bus_in, bus_out = left + STEP_COL + 40, right - 40
    o.append(
        f'<path d="M{left + STEP_COL} {middle_y[1]} L{bus_in} {middle_y[1]} M{left + STEP_COL} {middle_y[2]} '
        f'L{bus_in} {middle_y[2]} L{bus_in} {middle_y[0]}" fill="none" stroke="{p["line"]}" stroke-width="1.3" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
    )
    o.append(path(p, [(left + STEP_COL, middle_y[0]), (middle, middle_y[0])]))
    o.append(write(left + STEP_COL + 14, middle_y[0] - 8, "reads", MONO, p["muted"], mono=True))
    o.append(path(p, [(middle + STEP_COL, middle_y[0]), (right, middle_y[0])]))
    o.append(write(middle + STEP_COL + 14, middle_y[0] - 8, "writes", MONO, p["muted"], mono=True))
    o.extend(path(p, [(bus_out, middle_y[0]), (bus_out, y), (right, y)]) for y in middle_y[1:])
    service_x = middle + 120
    o.append(pair(p, service_x, STEP_ROWS[0] + 8 + BOX_H, STEP_ROWS[-1] + 8))
    o.append(write(service_x + 26, STEP_ROWS[1] + 46, "context.services", MONO, p["muted"], mono=True))

    # After step returns: the check, the evaluation and the activation rule, right to left.
    o.append(zone(p, 40, BAND_TOP, STEP_W - 40, BAND_TOP + 140, "reef", p["band"]))
    o.append(cap(56, BAND_TOP + 24, "after step returns, Reef runs", p["reef"]))
    band_row = BAND_TOP + 44
    o.append(card(p, right, band_row, STEP_COL, "reef", "Check", "`validate_tree", number=6))
    o.append(card(p, middle, band_row, STEP_COL, "reef", "Evaluation", "`compare · floor · none", number=7))
    o.append(card(p, left, band_row, STEP_COL, "reef", "Activation rule", "serve or hold", number=8))
    o.append(path(p, [(right + STEP_COL // 2, zone_bottom), (right + STEP_COL // 2, band_row)]))
    o.append(write(right + STEP_COL // 2 + 12, zone_bottom + 26, "the candidate", MONO, p["muted"], mono=True))
    o.append(path(p, [(right, band_row + BOX_H // 2), (middle + STEP_COL, band_row + BOX_H // 2)]))
    o.append(path(p, [(middle, band_row + BOX_H // 2), (left + STEP_COL, band_row + BOX_H // 2)]))
    o.append(
        write(
            40,
            STEP_H - 20,
            "An unchanged tree, or one that fails the check, skips the step: no evaluation and no release.",
            BODY,
            p["muted"],
        )
    )
    o.append("</svg>")
    return "\n".join(o)


# One adapter.yaml, three ways the agent runs: three columns under the adapter.
MODES_W, MODES_H = 1300, 760
MODES_X = (56, 468, 880)
MODES_COL = 364
MODES_TOP = 252
# The tree box is narrower than a column, so the render arrow has room for its label.
TREE_W = 300


def adapter_modes(theme: str, fonts) -> str:
    p = THEMES[theme]
    o = svg(
        p,
        fonts,
        MODES_W,
        MODES_H,
        "One adapter.yaml, three ways an agent runs. Reef renders the harness tree with the adapter. For an "
        "evaluation run, the rendered tree is the agent's whole configuration in a fresh root, and run.task "
        "runs one prompt. For a Harbor task, Reef installs the pinned agent in the task's container, copies the "
        "rendered tree in, tasks run.task with the task's instruction, and the task's verifier scores it. On a "
        "person's machine, the agent's own command installs the rendered tree as its plugin, beside the person's own "
        "configuration, the client runs the agent with run.layer, and sends the changed files back.",
        "One adapter.yaml, three ways an agent runs",
        "Reef renders the same tree for each; the adapter's sections say how the agent starts.",
        [("your machine", "machine"), ("Reef", "reef"), ("files", "store")],
    )
    tree_x, adapter_x = MODES_X[0], MODES_X[1]
    o.append(card(p, tree_x, 104, TREE_W, "store", "Harness tree", "`rules.md · skills/ · config/"))
    o.append(card(p, adapter_x, 104, MODES_X[2] + MODES_COL - adapter_x, "reef", "Adapter", "", dashed=True))
    o.append(write(adapter_x + 104, 104 + 33, "reef/harness/adapters/<agent>/", MONO, p["ink"], mono=True))
    o.append(
        write(
            adapter_x + 18,
            104 + 57,
            "adapter.yaml: install · evolve · dirs · sessions · model · run · config;  render.py",
            MONO,
            p["muted"],
            mono=True,
        )
    )
    o.append(path(p, [(tree_x + TREE_W, 104 + BOX_H // 2), (adapter_x, 104 + BOX_H // 2)]))
    o.append(write(tree_x + TREE_W + 14, 104 + BOX_H // 2 - 8, "render", MONO, p["muted"], mono=True))

    modes = [
        (
            "reef",
            "evaluation run, in Reef",
            [
                ("Run root", "the rendered tree, nothing else"),
                ("Run", "`run.task, {prompt} = a prompt"),
                ("Trajectory", "read with the sessions reader"),
            ],
            "`dirs · model · run.task · sessions",
        ),
        (
            "reef",
            "Harbor task, in its container",
            [
                ("Task container", "Harbor starts it with Docker"),
                ("Install", "`npm <package>@<version>"),
                ("Run", "`run.task, {prompt} = instruction"),
                ("Verifier", "the task's tests give the score"),
            ],
            "`install · dirs · model · run.task · sessions",
        ),
        (
            "machine",
            "a person's machine",
            [
                ("Plugin install", "the agent's own command: run.layer register"),
                ("Run", "`binary + run.layer args and env"),
                ("Sync back", "changed files → the tree"),
            ],
            "`install · model · run.layer",
        ),
    ]
    rows = [MODES_TOP + 44 + index * (BOX_H + 18) for index in range(4)]
    zone_bottom = rows[-1] + BOX_H + 56
    for x, (kind, title, boxes, sections) in zip(MODES_X, modes, strict=True):
        o.append(zone(p, x - 16, MODES_TOP, x + MODES_COL + 16, zone_bottom, kind, p["tint"]))
        o.append(cap(x, MODES_TOP + 26, title, p[kind]))
        for row, (box_title, line) in zip(rows, boxes, strict=False):
            o.append(card(p, x, row, MODES_COL, kind, box_title, line))
        o.extend(path(p, [(x + 40, upper + BOX_H), (x + 40, lower)]) for upper, lower in pairwise(rows[: len(boxes)]))
        o.append(write(x, zone_bottom - 18, sections[1:], MONO, p["muted"], mono=True))
        # The adapter drives every mode; the first column sits under the tree, so its line bends.
        if x < adapter_x:
            elbow_y = 104 + BOX_H + 36
            o.append(
                path(
                    p,
                    [
                        (adapter_x + 40, 104 + BOX_H),
                        (adapter_x + 40, elbow_y),
                        (x + MODES_COL // 2, elbow_y),
                        (x + MODES_COL // 2, MODES_TOP),
                    ],
                )
            )
        else:
            o.append(path(p, [(x + MODES_COL // 2, 104 + BOX_H), (x + MODES_COL // 2, MODES_TOP)]))
    o.append(
        write(
            40,
            MODES_H - 20,
            "The middle column needs an npm install: Reef installs the agent in each task's container.",
            BODY,
            p["muted"],
        )
    )
    o.append("</svg>")
    return "\n".join(o)


FIGURES = {"harness-method-step": method_step, "harness-adapter-modes": adapter_modes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font-cache", type=Path, default=Path.home() / ".cache" / "reef-fonts")
    args = parser.parse_args()
    fonts = font_files(args.font_cache)
    for name, figure in FIGURES.items():
        for theme in THEMES:
            target = ROOT / f"{name}-{theme}.svg"
            target.write_text(figure(theme, fonts))
            print(f"wrote {target.relative_to(ROOT.parent.parent)} ({target.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
