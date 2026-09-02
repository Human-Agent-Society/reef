"use client";

import mermaid from "mermaid";
import { useEffect, useId, useRef, useState } from "react";

let renderQueue: Promise<void> = Promise.resolve();

const lightTheme = {
  background: "transparent",
  primaryColor: "#ffffff",
  primaryTextColor: "#18222e",
  primaryBorderColor: "#8aa49f",
  secondaryColor: "#e5f2ef",
  tertiaryColor: "#eef2f8",
  lineColor: "#647585",
  textColor: "#26323f",
  mainBkg: "#ffffff",
  nodeBorder: "#8aa49f",
  clusterBkg: "#eef5f3",
  clusterBorder: "#b8cbc7",
  edgeLabelBackground: "#f3f6fb",
  actorBkg: "#ffffff",
  actorBorder: "#3f8077",
  actorTextColor: "#18222e",
  actorLineColor: "#9aabb9",
  signalColor: "#526879",
  signalTextColor: "#26323f",
  sequenceNumberColor: "#ffffff",
  labelBoxBkgColor: "#e5f2ef",
  labelBoxBorderColor: "#8aa49f",
  labelTextColor: "#21433f",
  loopTextColor: "#21433f",
  noteBkgColor: "#e5f2ef",
  noteBorderColor: "#8aa49f",
  noteTextColor: "#21433f",
  activationBkgColor: "#cce4df",
  activationBorderColor: "#3f8077",
};

const darkTheme = {
  background: "transparent",
  primaryColor: "#1b2530",
  primaryTextColor: "#f0f3f7",
  primaryBorderColor: "#668d86",
  secondaryColor: "#19332f",
  tertiaryColor: "#222e3a",
  lineColor: "#94a2b1",
  textColor: "#d3dae2",
  mainBkg: "#1b2530",
  nodeBorder: "#668d86",
  clusterBkg: "#162a28",
  clusterBorder: "#436b65",
  edgeLabelBackground: "#111820",
  actorBkg: "#1b2530",
  actorBorder: "#68a79d",
  actorTextColor: "#f0f3f7",
  actorLineColor: "#617180",
  signalColor: "#aeb9c5",
  signalTextColor: "#dfe5eb",
  sequenceNumberColor: "#18222e",
  labelBoxBkgColor: "#19332f",
  labelBoxBorderColor: "#68a79d",
  labelTextColor: "#d9eee9",
  loopTextColor: "#d9eee9",
  noteBkgColor: "#19332f",
  noteBorderColor: "#68a79d",
  noteTextColor: "#d9eee9",
  activationBkgColor: "#28544d",
  activationBorderColor: "#68a79d",
};

// The hardcoded themes above are the fallback; the active palette is read
// from the CSS variables at draw time so diagrams follow globals.css instead
// of drifting when the palette changes.
const paletteRoles: Record<string, string> = {
  actorBkg: "--surface-raised",
  edgeLabelBackground: "--card",
  tertiaryColor: "--accent",
  lineColor: "--muted",
  textColor: "--text",
  primaryTextColor: "--foreground",
  actorTextColor: "--foreground",
  // Nodes sit on the figure's card, so they need their own raised surface and
  // the page's hairline border; the fallback themes' green borders belong to a
  // palette the site no longer uses.
  primaryColor: "--surface-raised",
  mainBkg: "--surface-raised",
  nodeBorder: "--diagram-line",
  primaryBorderColor: "--diagram-line",
  // The group bands carry their title and their contents; an outline around
  // them adds a third frame inside the figure's own card, and puts a rule
  // exactly where the title sits.
  clusterBkg: "--accent",
  clusterBorder: "--accent",
  // Sequence diagrams carried their own teal from the fallback theme, which
  // reads as a second design next to a flowchart on the same page.
  actorBorder: "--diagram-line",
  actorLineColor: "--diagram-line",
  signalColor: "--muted",
  signalTextColor: "--text",
  labelBoxBkgColor: "--accent",
  labelBoxBorderColor: "--diagram-line",
  labelTextColor: "--muted",
  loopTextColor: "--text",
  noteBkgColor: "--accent",
  noteBorderColor: "--diagram-line",
  noteTextColor: "--text",
  activationBkgColor: "--accent",
  activationBorderColor: "--diagram-line",
  sequenceNumberColor: "--card",
};

// Mermaid ships Trebuchet MS as its default face. Left alone, every diagram
// renders in a typeface the site uses nowhere else, which reads as a pasted-in
// artifact rather than part of the page.
const typographyRoles: Record<string, string> = {
  fontFamily: "--font-sans",
};

function cssOverrides(roles: Record<string, string>): Record<string, string> {
  const style = getComputedStyle(document.documentElement);
  const overrides: Record<string, string> = {};
  for (const [variable, cssVar] of Object.entries(roles)) {
    const value = style.getPropertyValue(cssVar).trim();
    if (value) overrides[variable] = value;
  }
  return overrides;
}

function renderDiagram(id: string, chart: string, dark: boolean) {
  const task = renderQueue.then(async () => {
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      themeVariables: {
        ...(dark ? darkTheme : lightTheme),
        fontSize: "13px",
        ...cssOverrides({ ...paletteRoles, ...typographyRoles }),
      },
      flowchart: {
        htmlLabels: true,
        useMaxWidth: true,
        curve: "basis",
        nodeSpacing: 24,
        rankSpacing: 34,
        padding: 12,
        diagramPadding: 4,
        subGraphTitleMargin: { top: 0, bottom: 26 },
      },
      sequence: {
        useMaxWidth: true,
        mirrorActors: false,
        diagramMarginX: 8,
        diagramMarginY: 8,
        actorMargin: 28,
        width: 104,
        height: 42,
        boxMargin: 8,
        boxTextMargin: 5,
        noteMargin: 8,
        messageMargin: 30,
        actorFontSize: 13,
        actorFontWeight: 600,
        noteFontSize: 12,
        noteFontWeight: 500,
        messageFontSize: 12,
      },
    });
    return mermaid.render(id, chart);
  });
  renderQueue = task.then(
    () => undefined,
    () => undefined,
  );
  return task;
}

export function MermaidDiagram({ chart }: { chart: string }) {
  const stableId = `mermaid-${useId().replaceAll(":", "")}`;
  const accTitle = /^\s*accTitle:\s*(.+)$/m.exec(chart)?.[1].trim() ?? "Architecture diagram";
  const version = useRef(0);
  const [svg, setSvg] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    function draw() {
      const current = ++version.current;
      const dark = document.documentElement.dataset.theme === "dark";
      setError("");
      void renderDiagram(`${stableId}-${current}`, chart, dark)
        .then((result) => {
          if (active && current === version.current) setSvg(result.svg);
        })
        .catch((cause: unknown) => {
          if (!active || current !== version.current) return;
          setSvg("");
          setError(cause instanceof Error ? cause.message : "Unable to render diagram");
        });
    }

    draw();
    // A cold direct load lays the diagram out with fallback-font metrics; when
    // the site webfont arrives its wider glyphs would otherwise clip the boxes.
    // Redraw once fonts settle. document.fonts is guarded for older engines.
    if (typeof document !== "undefined" && "fonts" in document) {
      void document.fonts.ready.then(() => {
        if (active) draw();
      });
    }
    const observer = new MutationObserver((records) => {
      if (records.some((record) => record.attributeName === "data-theme")) draw();
    });
    observer.observe(document.documentElement, { attributes: true });
    return () => {
      active = false;
      observer.disconnect();
    };
  }, [chart, stableId]);

  if (error) {
    return (
      <figure className="mermaid-diagram mermaid-error">
        <figcaption>Diagram rendering failed: {error}</figcaption>
        <pre><code>{chart}</code></pre>
      </figure>
    );
  }

  if (!svg) {
    return (
      <figure className="mermaid-diagram mermaid-loading" aria-label={accTitle} aria-busy="true">
        <span>Rendering diagram…</span>
      </figure>
    );
  }

  return (
    <figure
      className="mermaid-diagram"
      role="img"
      aria-label={accTitle}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
