#!/usr/bin/env python3
"""Rebuild the service diagrams: python3 docs/assets/render_service_architecture.py

Writes services-light.svg and services-dark.svg: the seven services of a
deployment (``reef serve --component NAME``) around the shared store that they
read and write, and the queues between them. The drawing is static. It embeds
IBM Plex Sans and IBM Plex Mono from the font cache of render_architecture.py.
"""

from __future__ import annotations

import argparse
import base64
from html import escape
from pathlib import Path

from render_architecture import arrow, cap, font_files, text

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1240, 790

THEMES = {
    "light": {
        "card": "#ffffff",
        "ink": "#1c1917",
        "muted": "#6b645d",
        "line": "#57534e",
        "tint": "#efedea",
        "control": "#0d6b62",
        "inference": "#2f62c4",
        "samples": "#9a5f0b",
        "training": "#8f3168",
        "evaluation": "#3a7428",
        "data": "#6a4fb0",
        "harness": "#a3452a",
        "store": "#8a847d",
    },
    "dark": {
        "card": "#1c1a17",
        "ink": "#f5f3f0",
        "muted": "#aaa29a",
        "line": "#c9c2ba",
        "tint": "#2a2723",
        "control": "#4fc0b2",
        "inference": "#86a8f0",
        "samples": "#e0a54b",
        "training": "#e07fb6",
        "evaluation": "#8fcb72",
        "data": "#b3a1f2",
        "harness": "#ec9475",
        "store": "#9c958e",
    },
}

TITLE, BODY, MONO = 18, 14, 12

# Three columns: control and inference, the shared store, the services that take work from the queues.
LEFT_X, STORE_X, RIGHT_X = 40, 440, 900
COLUMN_WIDTH, STORE_WIDTH = 300, 360
TOP, BOTTOM = 110, 750

# A service box: (kind, title, subtitle, lines, height). A line that starts with a backquote is muted.
CLIENTS = ("store", "Clients", "agents · apps · scripts", [], 64)
INFERENCE = (
    "inference",
    "inference",
    "N replicas · one engine owner",
    [
        "reads served_models per request,",
        "forwards it to the engine and",
        "appends its record",
        "the owner takes serving work:",
        "starts artifacts, publishes routes",
    ],
    200,
)
CONTROL = (
    "control",
    "control",
    "one active + N standby · control lease",
    [
        "selects artifacts → serving",
        "new records → record_selection",
        "samples → job data → jobs",
        "checkpoints → commits, evaluations",
        "activation rule · activate",
        "takes data_sync, harness_sync",
        "and training_requests",
        "`calls no engine and no service",
    ],
    268,
)
RIGHT = [
    (
        "data",
        "data",
        "N replicas · no lease",
        ["data routes · scenario reads", "record and artifact copies", "imports and exports"],
        132,
    ),
    ("harness", "harness", "N replicas · no lease", ["routes of harness clients", "control commits each import"], 112),
    ("samples", "samples", "N workers", ["takes record_selection", "runs record2sample"], 112),
    (
        "training",
        "training",
        "N workers · one job per scenario",
        ["Slime · Tinker · Spindle", "harness evolution steps"],
        112,
    ),
    ("evaluation", "evaluation", "N workers", ["takes evaluation_rollout", "serves the candidate itself"], 112),
]
# The label of each right-hand service's edge to the store.
RIGHT_EDGES = ["items", "imports", "samples", "checkpoints", "scores"]

# The store's rows: (title, who writes and reads).
STORE_ROWS = [
    ("served_models", "inference owner writes · replicas read"),
    ("records", "inference appends · samples, data read"),
    ("data items", "uploads · samples · record, artifact items"),
    ("commit history · control lease", "control writes"),
    ("serving · inference-owner", "control → inference"),
    ("samples · jobs · evaluations", "control → samples, training, evaluation"),
    ("data", "control, data → data"),
    ("data_sync · harness_sync · training_requests", "data, harness, control → control"),
    ("artifact repository", "Git LFS · S3 · Hugging Face"),
]
ROW_HEIGHT, ROW_GAP, ROWS_TOP = 54, 8, 174
# The rows from this index on are queues; a caption above them takes QUEUE_GAP.
FIRST_QUEUE, QUEUE_GAP = 4, 18


def write(
    x: float,
    y: float,
    content: str,
    size: int,
    fill: str,
    weight: int = 500,
    anchor: str = "start",
    mono: bool = False,
) -> str:
    return text(x, y, escape(content), size, 400 if mono else weight, fill, anchor, mono=mono)


def service(
    palette: dict[str, str],
    x: float,
    y: float,
    width: float,
    service_spec: tuple[str, str, str, list[str], int],
    dashed: bool = False,
) -> str:
    """A service box: a title in the service's color, a mono subtitle, and its lines."""
    kind, title, subtitle, lines, height = service_spec
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    stroke_width = 2.2 if kind == "control" else 1.5
    out = [
        (
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="9" fill="{palette["card"]}" '
            f'stroke="{palette[kind]}" stroke-width="{stroke_width}"{dash}/>'
        ),
        write(x + 18, y + 30, title, TITLE, palette[kind] if kind != "store" else palette["ink"], 600),
        write(x + 18, y + 50, subtitle, MONO, palette["muted"], mono=True),
    ]
    for index, line in enumerate(lines):
        muted = line.startswith("`")
        out.append(
            write(x + 18, y + 76 + 21 * index, line.lstrip("`"), BODY, palette["muted"] if muted else palette["ink"])
        )
    return "".join(out)


def both_ways(palette: dict[str, str], x0: float, x1: float, y: float, label: str) -> str:
    """A read and write edge between x0 and x1 at y, with arrowheads at both ends and a mono label above."""
    return (
        f'<path d="M{x0} {y} L{x1} {y}" fill="none" stroke="{palette["line"]}" stroke-width="1.3" stroke-linecap="round"/>'
        + arrow(x1, y, "right", palette["line"])
        + arrow(x0, y, "left", palette["line"])
        + write((x0 + x1) / 2, y - 8, label, MONO, palette["muted"], anchor="middle", mono=True)
    )


def render(theme: str, fonts: list[tuple[str, int, bytes]]) -> str:
    palette = THEMES[theme]
    faces = "".join(
        f"@font-face{{font-family:'{family}';font-weight:{weight};font-style:normal;"
        f"src:url(data:font/woff2;base64,{base64.b64encode(data).decode()}) format('woff2')}}\n"
        for family, weight, data in fonts
    )
    # No background rectangle: the drawing sits on the page's own light or dark ground.
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
            'aria-label="The seven services of a Reef deployment around the shared store. Inference reads '
            "served_models, forwards each request to the engine and appends its record; one inference worker owns "
            "the engines and takes the serving queue. Control, one active instance with standbys, commits and "
            "queues work for the other services, and takes data_sync, harness_sync and training requests. Data, "
            "harness, samples, training and evaluation take work from their queues and write their results back "
            'to the store. No service calls another.">'
        ),
        f"<style>\n{faces}</style>",
        write(40, 44, "The services of a deployment", 24, palette["ink"], 600),
        write(
            40,
            70,
            "Every edge reads or writes the shared store. No service calls another, and control calls no engine.",
            15,
            palette["muted"],
        ),
    ]

    # Left: the clients send model calls to inference; control below.
    clients_height, inference_y = CLIENTS[4], TOP + CLIENTS[4] + 44
    elements.append(service(palette, LEFT_X, TOP, COLUMN_WIDTH, CLIENTS, dashed=True))
    elements.append(
        f'<path d="M{LEFT_X + 60} {TOP + clients_height} L{LEFT_X + 60} {inference_y}" fill="none" '
        f'stroke="{palette["line"]}" stroke-width="1.3"/>' + arrow(LEFT_X + 60, inference_y, "down", palette["line"])
    )
    elements.append(write(LEFT_X + 72, TOP + clients_height + 27, "/v1/*", MONO, palette["muted"], mono=True))
    elements.append(service(palette, LEFT_X, inference_y, COLUMN_WIDTH, INFERENCE))
    control_y = BOTTOM - CONTROL[4]
    elements.append(service(palette, LEFT_X, control_y, COLUMN_WIDTH, CONTROL))
    elements.append(both_ways(palette, LEFT_X + COLUMN_WIDTH + 2, STORE_X - 2, inference_y + 90, "read · write"))
    elements.append(both_ways(palette, LEFT_X + COLUMN_WIDTH + 2, STORE_X - 2, control_y + 120, "read · write"))

    # Middle: the shared store and its tables and queues.
    elements.append(
        f'<rect x="{STORE_X}" y="{TOP}" width="{STORE_WIDTH}" height="{BOTTOM - TOP}" rx="12" fill="{palette["card"]}" '
        f'stroke="{palette["store"]}" stroke-width="2"/>'
    )
    elements.append(write(STORE_X + 18, TOP + 30, "Shared store", TITLE, palette["ink"], 600))
    elements.append(
        write(STORE_X + 18, TOP + 50, "SQLite on one host · PostgreSQL on several", MONO, palette["muted"], mono=True)
    )
    for index, (title, line) in enumerate(STORE_ROWS):
        y = ROWS_TOP + index * (ROW_HEIGHT + ROW_GAP) + (QUEUE_GAP if index >= FIRST_QUEUE else 0)
        elements.append(
            f'<rect x="{STORE_X + 16}" y="{y}" width="{STORE_WIDTH - 32}" height="{ROW_HEIGHT}" rx="5" fill="{palette["tint"]}"/>'
        )
        elements.append(write(STORE_X + 28, y + 22, title, BODY, palette["ink"], 600))
        elements.append(write(STORE_X + 28, y + 42, line, MONO, palette["muted"], mono=True))
    queues_y = ROWS_TOP + FIRST_QUEUE * (ROW_HEIGHT + ROW_GAP) + 10
    elements.append(cap(STORE_X + 28, queues_y, "queues", palette["muted"], size=11))

    # Right: the services that take work from the queues.
    y = TOP
    gap = (BOTTOM - TOP - sum(service_spec[4] for service_spec in RIGHT)) / (len(RIGHT) - 1)
    for service_spec, label in zip(RIGHT, RIGHT_EDGES, strict=True):
        elements.append(service(palette, RIGHT_X, y, COLUMN_WIDTH, service_spec))
        elements.append(both_ways(palette, STORE_X + STORE_WIDTH + 2, RIGHT_X - 2, y + 56, label))
        y += service_spec[4] + gap

    elements.append("</svg>")
    return "\n".join(elements)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font-cache", type=Path, default=Path.home() / ".cache" / "reef-fonts")
    args = parser.parse_args()
    fonts = font_files(args.font_cache)
    for theme in THEMES:
        target = ROOT / f"services-{theme}.svg"
        target.write_text(render(theme, fonts))
        print(f"wrote {target.relative_to(ROOT.parent.parent)} ({target.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
