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

// A head bar per block: the language, and a copy button that is always in view and never covers the code.
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
          <span aria-live="polite">{copied ? "Copied" : "Copy"}</span>
        </button>
      </div>
      <pre ref={ref} className={className}>
        {children}
      </pre>
    </div>
  );
}
