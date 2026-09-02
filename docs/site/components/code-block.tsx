"use client";

import { Check, Copy } from "lucide-react";
import { useRef, useState } from "react";
import type { ReactNode } from "react";

// A copy button in the top-right of every code block. The rendered <pre> is
// wrapped so the button positions against it; the text copied is the block's
// own textContent, which the shiki path stores with real newlines. On a
// phone the button is always visible (no hover), so it sits out of the way in
// the padding and grows to a 44px touch target on coarse pointers.
export function CodeBlock({ className, children }: { className?: string; children: ReactNode }) {
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);

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
      <pre ref={ref} className={className}>
        {children}
      </pre>
      <button
        type="button"
        className="code-copy"
        onClick={copy}
        aria-label={copied ? "Copied" : "Copy code"}
        title={copied ? "Copied" : "Copy"}
      >
        {copied ? <Check size={15} /> : <Copy size={15} />}
      </button>
    </div>
  );
}
