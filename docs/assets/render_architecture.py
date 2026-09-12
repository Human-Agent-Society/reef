#!/usr/bin/env python3
"""Rebuild the README architecture diagrams: python3 docs/assets/render_architecture.py

Writes architecture-light.svg and architecture-dark.svg. Each file is
self-contained: IBM Plex Sans and IBM Plex Mono are embedded, box highlights
are CSS animations, and the moving dots are SMIL, so the drawing animates inside
an <img> tag on GitHub and stays a readable static diagram wherever animation
is off. The fonts are fetched from Google Fonts once and cached under
~/.cache/reef-fonts (override with --font-cache).
"""

from __future__ import annotations

import argparse
import base64
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
W, H = 1200, 550
CYCLE = 12  # seconds for one lap of serve, observe, grow, commit

THEMES = {
    "light": {
        "card": "#ffffff",
        "ink": "#14110e",
        "line": "#3c3630",
        "muted": "#7d766e",
        "faint": "#b8b0a6",
        "accent": "#a03729",
    },
    "dark": {
        "card": "#1c1a16",
        "ink": "#f7f4f0",
        "line": "#cfc8bf",
        "muted": "#a49c93",
        "faint": "#5d554c",
        "accent": "#d99183",
    },
}

FONTS = [("IBM Plex Sans", 500), ("IBM Plex Sans", 600), ("IBM Plex Mono", 400)]
FONTS_CSS = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@500;600&family=IBM+Plex+Mono:wght@400&display=swap"
)
USER_AGENT = "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/120 Safari/537.36"

# The two rows of the diagram: what serves a request, and what learns from it.
# Each box is a title and two lines of description.
SERVING = [
    ("Harness", "agent, prompts and tools", "receipt-linked feedback"),
    ("Scenario", "freezes the release", "stores the interaction"),
    ("Inference", "provider-native requests", "OpenAI, Anthropic compatible"),
]
LEARNING = [
    ("Records", "matches feedback to", "recorded interactions"),
    ("Trainer", "runs the recipe", "weights or harness"),
    ("Evaluation", "evaluates the candidate", "selects or rejects it"),
    ("Release", "accepted artifact with", "its parent history"),
]
STEPS = [
    ("Serve", "request served and recorded"),
    ("Observe", "feedback matched to records"),
    ("Grow", "recipe trains on records"),
    ("Commit", "evaluated, then published"),
]
# Which step lights each box up.
PHASE_OF = {"Harness": 1, "Scenario": 1, "Inference": 1, "Records": 2, "Trainer": 3, "Evaluation": 4, "Release": 4}

# Type scale.
TITLE, BODY, LABEL, NOTE, CAPS = 24, 17, 16, 15, 14
STEP_NAME, STEP_DESC, BADGE_R = 22, 17, 16

Y1, Y2, BH = 54, 286, 122
ROW1 = [(40, 344), (428, 344), (816, 344)]
ROW2 = [(40, 253), (329, 253), (618, 253), (907, 253)]
STRIP_Y = 500


def font_files(cache: Path) -> list[tuple[str, int, bytes]]:
    """Latin woff2 for each face, downloaded once through the Google Fonts CSS API."""
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    css = None
    for family, weight in FONTS:
        target = cache / f"{family.replace(' ', '')}-{weight}.woff2"
        if not target.exists():
            if css is None:
                req = urllib.request.Request(FONTS_CSS, headers={"User-Agent": USER_AGENT})
                css = urllib.request.urlopen(req, timeout=30).read().decode()
            block = next(
                b
                for b in re.findall(r"@font-face\s*{(.*?)}", css, re.S)
                if f"'{family}'" in b and f"font-weight: {weight}" in b and "U+0000-00FF" in b
            )
            url = re.search(r"url\((https://[^)]+)\)", block).group(1)
            target.write_bytes(urllib.request.urlopen(url, timeout=30).read())
        out.append((family, weight, target.read_bytes()))
    return out


def text(x, y, s, size, weight=500, fill="", anchor="start", cls="", mono=False, extra=""):
    fam = "IBM Plex Mono" if mono else "IBM Plex Sans"
    c = f' class="{cls}"' if cls else ""
    f = f' fill="{fill}"' if fill else ""
    return (
        f'<text{c} x="{x}" y="{y}" font-family="\'{fam}\', ui-sans-serif, sans-serif" '
        f'font-size="{size}" font-weight="{weight}"{f} text-anchor="{anchor}"{extra}>{s}</text>'
    )


def cap(x, y, s, fill, size=CAPS):
    return text(x, y, s.upper(), size, 500, fill, extra=f' letter-spacing="{size * 0.1}"')


def label(x, y, s, fill, anchor="middle", size=LABEL):
    return text(x, y, s, size, 400, fill, anchor, mono=True)


def arrow(x, y, direction, color, size=7):
    r = {"right": 0, "down": 90, "left": 180, "up": 270}[direction]
    return (
        f'<path d="M{-size} {-size * 0.55} L0 0 L{-size} {size * 0.55}" fill="none" stroke="{color}" '
        f'stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" '
        f'transform="translate({x} {y}) rotate({r})"/>'
    )


def box(p, x, y, w, name, line1, line2):
    phase = PHASE_OF[name]
    return "".join(
        [
            (
                f'<rect class="hi{phase}" x="{x}" y="{y}" width="{w}" height="{BH}" rx="10" fill="{p["card"]}" '
                f'stroke="{p["line"]}" stroke-width="1.3"/>'
            ),
            text(x + 18, y + 42, name, TITLE, 600, p["ink"]),
            text(x + 18, y + 72, line1, BODY, 500, p["muted"]),
            text(x + 18, y + 98, line2, BODY, 500, p["muted"]),
        ]
    )


def dot(p, path_id, t0, t1):
    """A dot that runs along a connector between two moments of the cycle (fractions of CYCLE)."""
    e = 0.004
    return (
        f'<circle r="4.5" fill="{p["accent"]}" opacity="0">'
        f'<animateMotion dur="{CYCLE}s" repeatCount="indefinite" calcMode="linear" '
        f'keyPoints="0;0;1;1" keyTimes="0;{t0};{t1};1"><mpath xlink:href="#{path_id}"/></animateMotion>'
        f'<animate attributeName="opacity" dur="{CYCLE}s" repeatCount="indefinite" '
        f'values="0;0;1;1;0;0" keyTimes="0;{t0};{t0 + e};{t1 - e};{t1};1"/></circle>'
    )


def keyframes(name: str, a: int, b: int, on: str, off: str) -> str:
    """One CSS animation that holds `on` between a% and b% of the cycle and `off` elsewhere."""
    stops = []
    if a > 0:
        stops.append(f"0%,{a - 1}%{{{off}}}")
    stops.append(f"{a}%,{b - 2}%{{{on}}}")
    stops.append(f"{b}%,100%{{{off}}}" if b < 100 else f"100%{{{off}}}")
    return f"@keyframes {name}{{{''.join(stops)}}}"


def styles(p, fonts):
    faces = "".join(
        f"@font-face{{font-family:'{fam}';font-weight:{w};font-style:normal;"
        f"src:url(data:font/woff2;base64,{base64.b64encode(data).decode()}) format('woff2')}}\n"
        for fam, w, data in fonts
    )
    # Each step owns a quarter of the cycle; the highlighted stroke fades over the last 2%.
    frames = []
    for n in range(1, 5):
        a, b = (n - 1) * 25, n * 25
        on, off = f"stroke:{p['accent']};stroke-width:1.8", f"stroke:{p['line']};stroke-width:1.3"
        fill_on, fill_off = f"fill:{p['accent']}", f"fill:{p['card']}"
        num_on, num_off = f"fill:{p['card']}", f"fill:{p['accent']}"
        name_on, name_off = f"fill:{p['ink']}", f"fill:{p['muted']}"
        frames += [
            keyframes(f"hi{n}", a, b, on, off),
            keyframes(f"bf{n}", a, b, fill_on, fill_off),
            keyframes(f"bn{n}", a, b, num_on, num_off),
            keyframes(f"nm{n}", a, b, name_on, name_off),
        ]
        frames += [
            f".hi{n}{{animation:hi{n} {CYCLE}s linear infinite}}",
            f".bf{n}{{animation:bf{n} {CYCLE}s linear infinite}}",
            f".bn{n}{{animation:bn{n} {CYCLE}s linear infinite}}",
            f".nm{n}{{animation:nm{n} {CYCLE}s linear infinite}}",
        ]
    return "<style>\n" + faces + "\n".join(frames) + "\n</style>"


def render(theme: str, fonts) -> str:
    p = THEMES[theme]
    # No background rectangle: the drawing sits on the README's own light or dark ground.
    o = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="Reef architecture: a harness sends requests through a scenario to inference; '
            f"records match feedback to the interactions, the trainer runs the recipe, evaluation accepts "
            f'or rejects the candidate, and the accepted release serves the next request.">'
        ),
        styles(p, fonts),
    ]

    # serving row; the request labels sit above the row and the return labels below it
    o.append(cap(40, 34, "serving", p["faint"]))
    for (x, w), (name, line1, line2) in zip(ROW1, SERVING, strict=True):
        o.append(box(p, x, Y1, w, name, line1, line2))
    for k in range(2):
        gx0, gx1 = ROW1[k][0] + ROW1[k][1], ROW1[k + 1][0]
        top, bot = Y1 + 42, Y1 + 80
        o.append(
            f'<path id="fwd{k}" d="M{gx0 + 2} {top} H{gx1 - 4}" stroke="{p["line"]}" stroke-width="1.3" fill="none"/>'
        )
        o.append(arrow(gx1 - 3, top, "right", p["line"]))
        o.append(
            f'<path id="back{k}" d="M{gx1 - 2} {bot} H{gx0 + 4}" stroke="{p["muted"]}" stroke-width="1.3" fill="none"/>'
        )
        o.append(arrow(gx0 + 3, bot, "left", p["muted"]))
        cx = (gx0 + gx1) / 2
        o.append(label(cx, Y1 - 14, ("request", "proxy")[k], p["muted"]))
        o.append(label(cx, Y1 + BH + 24, ("receipt", "reply")[k], p["muted"]))

    # learning row; the hand-off labels sit below the row
    o.append(cap(40, Y2 - 18, "learning", p["faint"]))
    for (x, w), (name, line1, line2) in zip(ROW2, LEARNING, strict=True):
        o.append(box(p, x, Y2, w, name, line1, line2))
    labels = ["batch", "candidate", "accepted"]
    for k in range(3):
        gx0, gx1 = ROW2[k][0] + ROW2[k][1], ROW2[k + 1][0]
        y = Y2 + 46
        o.append(
            f'<path id="learn{k}" d="M{gx0 + 2} {y} H{gx1 - 4}" stroke="{p["line"]}" stroke-width="1.3" fill="none"/>'
        )
        o.append(arrow(gx1 - 3, y, "right", p["line"]))
        o.append(label((gx0 + gx1) / 2, Y2 + BH + 24, labels[k], p["muted"]))
    ex, ew = ROW2[2]
    o.append(label(ex + ew / 2, Y2 + BH + 48, "rejected: keep the current release", p["faint"], size=NOTE))
    lx, lw = ROW2[3]
    o.append(label(lx + lw / 2, Y2 + BH + 48, "harness pulls the release", p["faint"], size=NOTE))

    # scenario -> records, and release -> scenario
    sx, sw = ROW1[1]
    rx, rw = ROW2[0]
    down_x, up_x = sx + 60, sx + sw - 60
    o.append(
        f'<path id="observe" d="M{down_x} {Y1 + BH + 2} V244 H{rx + rw / 2} V{Y2 - 4}" '
        f'stroke="{p["line"]}" stroke-width="1.3" fill="none"/>'
    )
    o.append(arrow(rx + rw / 2, Y2 - 3, "down", p["line"]))
    o.append(label((down_x + rx + rw / 2) / 2, 237, "records · feedback", p["muted"]))
    o.append(
        f'<path id="commit" d="M{lx + lw / 2} {Y2 - 2} V216 H{up_x} V{Y1 + BH + 4}" '
        f'stroke="{p["accent"]}" stroke-width="1.4" fill="none"/>'
    )
    o.append(arrow(up_x, Y1 + BH + 3, "up", p["accent"]))
    o.append(label(lx + lw / 2 - 90, 209, "served next", p["accent"]))

    # the step strip
    for k, (name, desc) in enumerate(STEPS):
        n = k + 1
        x = 40 + BADGE_R + k * 285
        o.append(
            f'<circle class="bf{n}" cx="{x}" cy="{STRIP_Y}" r="{BADGE_R}" fill="{p["card"]}" '
            f'stroke="{p["accent"]}" stroke-width="1.5"/>'
        )
        o.append(text(x, STRIP_Y + 6, str(n), 16, 600, p["accent"], "middle", cls=f"bn{n}"))
        o.append(text(x + BADGE_R + 14, STRIP_Y + 7, name, STEP_NAME, 600, p["muted"], cls=f"nm{n}"))
        o.append(text(x + BADGE_R + 14, STRIP_Y + 34, desc, STEP_DESC, 500, p["muted"]))

    # the moving dots, one lap per cycle
    for path_id, t0, t1 in [
        ("fwd0", 0.01, 0.05),
        ("fwd1", 0.06, 0.10),
        ("back1", 0.11, 0.15),
        ("back0", 0.16, 0.21),
        ("observe", 0.27, 0.46),
        ("learn0", 0.52, 0.58),
        ("learn1", 0.62, 0.70),
        ("learn2", 0.77, 0.83),
        ("commit", 0.85, 0.97),
    ]:
        o.append(dot(p, path_id, t0, t1))
    o.append("</svg>")
    return "\n".join(o)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font-cache", type=Path, default=Path.home() / ".cache" / "reef-fonts")
    args = parser.parse_args()
    fonts = font_files(args.font_cache)
    for theme in THEMES:
        target = ROOT / f"architecture-{theme}.svg"
        target.write_text(render(theme, fonts))
        print(f"wrote {target.relative_to(ROOT.parent.parent)} ({target.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
