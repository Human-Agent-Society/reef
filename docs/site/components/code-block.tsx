"use client";

import { Check, Copy } from "lucide-react";
import { useRef, useState } from "react";
import type { ReactNode } from "react";

const LABELS: Record<string, string> = {
  bash: "Shell",
  sh: "Shell",
  shell: "Shell",
  console: "Shell",
  python: "Python",
  py: "Python",
  yaml: "YAML",
  yml: "YAML",
  json: "JSON",
  toml: "TOML",
  text: "Text",
  ts: "TypeScript",
  typescript: "TypeScript",
  js: "JavaScript",
  javascript: "JavaScript",
  http: "HTTP",
  ini: "INI",
  diff: "Diff",
  rst: "reStructuredText",
};

function languageOf(className?: string) {
  const name = className?.split(" ").find((token) => token.startsWith("language-"))?.slice("language-".length);
  return name ? (LABELS[name] ?? name) : "";
}

// Every code block gets a head bar: the language on the left, the copy button
// on the right, always visible, so a phone needs no hover and the button never
// covers the first line of code. The text copied is the block's own
// textContent, which the shiki path stores with real newlines.
export function CodeBlock({ className, children }: { className?: string; children: ReactNode }) {
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);
  const language = languageOf(className);

  async function copy() {
    const text = ref.current?.textContent ?? "";
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // clipboard unavailable (insecure context, denied): leave the button as is
    }
  }

  return (
    <div className="code-block">
      <div className="code-head">
        <span>{language}</span>
        <button type="button" className="code-copy" onClick={copy} aria-label={copied ? "Copied" : "Copy code"} title={copied ? "Copied" : "Copy"}>
          {copied ? <Check size={14} /> : <Copy size={14} />}
          <span>{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>
      <pre ref={ref} className={className}>
        {children}
      </pre>
    </div>
  );
}
