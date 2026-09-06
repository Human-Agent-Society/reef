#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow==11.3.0"]
# ///
"""Rebuild the README architecture GIFs: uv run docs/assets/render_architecture.py.

Uses Arial on macOS or DejaVu Sans on Linux. Override with --font and --bold-font.
The diagram stays readable in every frame; motion only highlights the data flow.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT, SCALE = 1200, 780, 2
FRAME_MS, FRAMES = 80, 200
THEMES = {
    "light": {
        "bg": "#F7FAFB",
        "panel": "#FFFFFF",
        "ink": "#17343D",
        "muted": "#617B84",
        "border": "#DCE7EA",
        "wire": "#BDCED4",
        "teal": "#087F8C",
        "violet": "#7460BE",
        "soft": "#EAF4F5",
        "shadow": "#EAF0F2",
    },
    "dark": {
        "bg": "#0D171E",
        "panel": "#14232C",
        "ink": "#E5F0F3",
        "muted": "#91ABB5",
        "border": "#293F49",
        "wire": "#3D5661",
        "teal": "#57D4CE",
        "violet": "#B5A2F3",
        "soft": "#192F37",
        "shadow": "#0B141A",
    },
}
PHASES = [
    ("Serve", "Freeze the release. Return a response and receipt."),
    ("Observe", "Match feedback to receipts. Prepare eligible records."),
    ("Grow", "Run the recipe. Evaluate and select a candidate."),
    ("Commit", "Publish selected updates. Keep serving if rejected."),
]


def rgb(color):
    return tuple(bytes.fromhex(color.lstrip("#")))


def blend(a, b, amount):
    return tuple(round(x + (y - x) * amount) for x, y in zip(rgb(a), rgb(b), strict=False))


def font_path(bold=False):
    names = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold
        else ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for name in names:
        if Path(name).exists():
            return name
    raise FileNotFoundError("Pass --font and --bold-font with local TrueType fonts.")


class Diagram:
    def __init__(self, theme, regular, bold):
        self.c = THEMES[theme]
        self.fonts = {
            (size, weight): ImageFont.truetype(bold if weight else regular, size * SCALE)
            for size in (14, 15, 16, 17, 18, 20, 23, 26, 32)
            for weight in (False, True)
        }
        self.image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), self.c["bg"])
        self.draw = ImageDraw.Draw(self.image)
        self.paths = {}
        self.base()
        self.static = self.image.copy()

    def box(self, bounds, fill, outline=None, radius=16, width=1):
        self.draw.rounded_rectangle(
            tuple(v * SCALE for v in bounds), radius=radius * SCALE, fill=fill, outline=outline, width=width * SCALE
        )

    def text(self, x, y, value, size=18, color="ink", bold=False, anchor="la"):
        self.draw.text(
            (x * SCALE, y * SCALE), value, font=self.fonts[size, bold], fill=self.c.get(color, color), anchor=anchor
        )

    def line(self, points, color, width=2):
        self.draw.line([(x * SCALE, y * SCALE) for x, y in points], fill=color, width=width * SCALE, joint="curve")

    def dot(self, x, y, radius, color):
        self.draw.ellipse(
            ((x - radius) * SCALE, (y - radius) * SCALE, (x + radius) * SCALE, (y + radius) * SCALE), fill=color
        )

    def route(self, name, points, accent="teal"):
        # Round orthogonal bends so animated packets travel smoothly through them.
        smooth = [points[0]]
        for a, b, c in zip(points, points[1:], points[2:], strict=False):
            ab, bc = math.dist(a, b), math.dist(b, c)
            r = min(14, ab / 2, bc / 2)
            p = tuple(b[i] + (a[i] - b[i]) * r / ab for i in (0, 1))
            q = tuple(b[i] + (c[i] - b[i]) * r / bc for i in (0, 1))
            smooth.append(p)
            for step in range(1, 13):
                t = step / 12
                smooth.append(tuple((1 - t) ** 2 * p[i] + 2 * (1 - t) * t * b[i] + t * t * q[i] for i in (0, 1)))
        smooth.append(points[-1])
        self.paths[name] = (smooth, accent)
        self.line(smooth, self.c["wire"])
        a, b = smooth[-2:]
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        self.line(
            [
                (b[0] - 8 * math.cos(angle - 0.5), b[1] - 8 * math.sin(angle - 0.5)),
                b,
                (b[0] - 8 * math.cos(angle + 0.5), b[1] - 8 * math.sin(angle + 0.5)),
            ],
            self.c["wire"],
        )

    def icon(self, x, y, kind, accent):
        color = self.c[accent]
        self.box((x, y, x + 36, y + 36), blend(self.c["panel"], color, 0.10), radius=10)
        if kind == "harness":
            self.line([(x + 10, y + 12), (x + 16, y + 18), (x + 10, y + 24)], color)
            self.line([(x + 20, y + 24), (x + 27, y + 24)], color)
        elif kind == "scenario":
            for dx, dy in ((10, 10), (24, 10), (10, 24), (24, 24)):
                self.box((x + dx - 3, y + dy - 3, x + dx + 3, y + dy + 3), None, color, 2)
        elif kind == "inference":
            self.line(
                [
                    (x + 20, y + 7),
                    (x + 11, y + 20),
                    (x + 18, y + 20),
                    (x + 16, y + 29),
                    (x + 26, y + 15),
                    (x + 19, y + 15),
                ],
                color,
            )
        elif kind == "records":
            for dy in (10, 17, 24):
                self.line([(x + 11, y + dy), (x + 26, y + dy)], color)
                self.dot(x + 7, y + dy, 1, color)
        elif kind == "trainer":
            for dx, dy in ((10, 23), (18, 15), (26, 8)):
                self.line([(x + dx, y + 28), (x + dx, y + dy)], color, 3)
        elif kind == "evaluation":
            self.line([(x + 9, y + 18), (x + 15, y + 24), (x + 27, y + 11)], color, 3)

    def card(self, x, y, title, subtitle, detail, kind, accent="teal"):
        self.box((x, y + 4, x + 300, y + 130), self.c["shadow"])
        self.box((x, y, x + 300, y + 126), self.c["panel"], self.c["border"])
        self.icon(x + 20, y + 20, kind, accent)
        self.text(x + 68, y + 25, title, 23, bold=True)
        self.text(x + 20, y + 69, subtitle, 18, color="muted")
        self.text(x + 20, y + 96, detail, 16, color=accent)

    def base(self):
        c = self.c
        self.box((1, 1, 1199, 779), c["bg"], c["border"], 24)
        self.text(40, 30, "REEF  /  ARCHITECTURE", 15, "teal", True)
        self.text(40, 61, "Live requests. Continuous learning.", 32, bold=True)
        self.dot(998, 48, 4, c["teal"])
        self.text(1012, 37, "Always serving", 16, "muted")
        self.text(1158, 72, "Weights + harness", 16, "muted", anchor="ra")
        self.line([(40, 116), (1160, 116)], c["border"], 1)

        self.route("request", [(400, 211), (450, 211)])
        self.route("response", [(450, 255), (400, 255)])
        self.route("infer", [(750, 211), (800, 211)])
        self.route("answer", [(800, 255), (750, 255)])
        self.route("record", [(600, 300), (600, 344), (250, 344), (250, 407)])
        self.route("batch", [(400, 470), (450, 470)])
        self.route("candidate", [(750, 470), (800, 470)], "violet")
        self.route("publish", [(950, 533), (950, 622), (750, 622)], "violet")
        self.route("release", [(450, 622), (62, 622), (62, 143), (600, 143), (600, 174)])
        self.text(425, 181, "request", 14, "muted", anchor="ma")
        self.text(425, 270, "receipt", 14, "muted", anchor="ma")
        self.text(775, 181, "proxy", 14, "muted", anchor="ma")
        self.text(775, 270, "reply", 14, "muted", anchor="ma")
        self.text(406, 318, "records + feedback", 17, "muted", anchor="ma")
        self.text(425, 441, "batch", 14, "muted", anchor="ma")
        self.text(775, 441, "artifact", 14, "muted", anchor="ma")
        self.text(981, 571, "selected", 17, "violet")
        self.text(255, 595, "serve current release", 17, "teal", anchor="ma")

        self.card(100, 174, "Harness", "Agent, prompts and tools", "Requests + receipt-linked feedback", "harness")
        self.card(
            450, 174, "Scenario", "Freeze release · verify response", "Return receipt · store interaction", "scenario"
        )
        self.card(
            800, 174, "Inference", "Provider-native model requests", "OpenAI / Anthropic compatible", "inference"
        )
        self.text(100, 375, "LEARNING LOOP", 14, "muted", True)
        self.text(1100, 375, "Scoped to each scenario", 15, "muted", anchor="ra")
        self.card(100, 407, "Records", "Match feedback to interactions", "Filter eligible records", "records")
        self.card(
            450,
            407,
            "Trainer",
            "Prepare a batch · run training",
            "Recipe updates weights or harness",
            "trainer",
            "violet",
        )
        self.card(
            800,
            407,
            "Artifact evaluation",
            "Evaluate · select or reject",
            "Rejected? Keep the current release.",
            "evaluation",
            "violet",
        )

        self.box((450, 582, 750, 662), c["panel"], c["border"], 16)
        self.text(472, 595, "Versioned release", 20, bold=True)
        self.text(472, 626, "Accepted artifact + parent history", 16, "muted")
        for x in (681, 701, 721):
            self.dot(x, 609, 3, c["teal"])
        self.line([(684, 609), (718, 609)], c["teal"], 1)
        self.text(100, 672, "Harness recipes pull the served tree before running.", 15, "muted")

    def packet(self, name, progress):
        points, accent = self.paths[name]
        lengths = [math.dist(a, b) for a, b in itertools.pairwise(points)]
        total = sum(lengths)

        def point_at(t):
            distance = min(1, max(0, t)) * total
            for a, b, length in zip(points, points[1:], lengths, strict=False):
                if distance <= length:
                    ratio = distance / length if length else 0
                    return tuple(a[i] + (b[i] - a[i]) * ratio for i in (0, 1))
                distance -= length
            return points[-1]

        color = self.c[accent]
        for i in range(10, 0, -1):
            t = progress - i * 5 / total
            if t >= 0:
                x, y = point_at(t)
                self.dot(x, y, 2, blend(self.c["bg"], color, (11 - i) / 14))
        x, y = point_at(progress)
        self.dot(x, y, 9, blend(self.c["bg"], color, 0.12))
        self.dot(x, y, 5, color)
        self.dot(x - 1, y - 1, 1.5, self.c["panel"])

    def frame(self, index):
        self.image = self.static.copy()
        self.draw = ImageDraw.Draw(self.image)
        t = index * FRAME_MS / 1000
        phase = min(3, int(t / 4))
        # Inference traffic continues independently while learning progresses.
        traffic = (t % 3.2) / 3.2
        for name, start in (("request", 0), ("infer", 0.25), ("answer", 0.5), ("response", 0.75)):
            if start <= traffic < start + 0.25:
                self.packet(name, (traffic - start) * 4)
        for name, start, end in (
            ("record", 4, 6.4),
            ("batch", 6.4, 8),
            ("candidate", 9.2, 11.8),
            ("publish", 12, 13.6),
            ("release", 13.6, 16),
        ):
            if start <= t < end:
                self.packet(name, (t - start) / (end - start))

        # A restrained border pulse highlights the active learning stage.
        active = [(450, 174, 750, 300), (100, 407, 400, 533), (450, 407, 750, 533), (450, 582, 750, 662)][phase]
        accent = "violet" if phase == 2 else "teal"
        strength = 0.40 + 0.20 * math.sin(t * math.pi)
        self.box(active, None, blend(self.c["border"], self.c[accent], strength), 16, 2)
        self.box((40, 699, 1160, 758), self.c["soft"], radius=14)
        for i, (label, _) in enumerate(PHASES):
            x = 57 + i * 127
            if i == phase:
                self.box((x, 710, x + 116, 745), self.c["panel"], radius=9)
            self.text(x + 10, 718, f"0{i + 1}  {label}", 16, "teal" if i == phase else "muted", i == phase)
        self.text(595, 720, PHASES[phase][1], 17, "ink")
        return self.image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font")
    parser.add_argument("--bold-font")
    args = parser.parse_args()
    for theme in THEMES:
        diagram = Diagram(theme, args.font or font_path(), args.bold_font or font_path(True))
        # One shared palette avoids frame-to-frame color shimmer.
        samples = [diagram.frame(i).resize((600, 390)) for i in (12, 65, 120, 165, 190)]
        atlas = Image.new("RGB", (600, 390 * len(samples)))
        for i, sample in enumerate(samples):
            atlas.paste(sample, (0, i * 390))
        palette = atlas.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        frames = [diagram.frame(i).quantize(palette=palette, dither=Image.Dither.NONE) for i in range(FRAMES)]
        output = ROOT / f"architecture-{theme}.gif"
        frames[0].save(
            output, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=True, disposal=1
        )
        print(f"{output.name}: {output.stat().st_size / 1024:.0f} KiB, {FRAMES} frames, 16 seconds")


if __name__ == "__main__":
    main()
