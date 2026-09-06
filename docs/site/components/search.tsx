"use client";

import Link from "next/link";
import { ArrowRight, FileText, Search as SearchIcon, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

type SearchDocument = {
  slug: string;
  title: string;
  description: string;
  text: string;
};

export function Search({ documents }: { documents: SearchDocument[] }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(0);
  // Reset the highlight when the query changes, in render, without an effect.
  const [lastQuery, setLastQuery] = useState(query);
  if (query !== lastQuery) {
    setLastQuery(query);
    setSelected(0);
  }
  const inputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const restoreFocus = useRef<HTMLElement | null>(null);

  const closeSearch = useCallback(() => {
    setOpen(false);
    setQuery("");
  }, []);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        // One dialog at a time: the shortcut waits while the drawer is open.
        if (!document.querySelector('[role="dialog"]:not(.search-dialog)')) setOpen(true);
      }
      if (event.key === "Escape") closeSearch();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [closeSearch]);

  useEffect(() => {
    if (!open) return;
    // Remember what opened the dialog so focus returns there on close, and
    // put focus in the input to start.
    restoreFocus.current = document.activeElement as HTMLElement | null;
    requestAnimationFrame(() => inputRef.current?.focus());
    // Scroll lock, so result-list scrolling does not chain into the page.
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
      restoreFocus.current?.focus();
    };
  }, [open]);

  const results = useMemo(() => {
    const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    if (!terms.length) return documents.slice(0, 6);

    return documents
      .map((doc) => {
        const title = doc.title.toLowerCase();
        const description = doc.description.toLowerCase();
        const text = doc.text.toLowerCase();
        // Accumulate only positive matches: a page that matches one term is
        // kept and ranked by its strongest field; an absent term never
        // subtracts, so a multi-word query cannot cancel a real match out.
        const score = terms.reduce((total, term) => {
          if (title.includes(term)) return total + 12;
          if (description.includes(term)) return total + 5;
          if (text.includes(term)) return total + 1;
          return total;
        }, 0);
        return { doc, score };
      })
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 8)
      .map(({ doc }) => doc);
  }, [documents, query]);

  const hasQuery = query.trim().length > 0;

  function onInputKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (!results.length) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setSelected((index) => (index + 1) % results.length);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setSelected((index) => (index - 1 + results.length) % results.length);
    } else if (event.key === "Enter") {
      event.preventDefault();
      const doc = results[selected];
      if (doc) {
        closeSearch();
        router.push(`/docs/${doc.slug}`);
      }
    }
  }

  function onDialogKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (event.key !== "Tab" || !dialogRef.current) return;
    // Trap Tab inside the dialog so focus cannot fall behind the modal.
    const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input, [tabindex]:not([tabindex="-1"])',
    );
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  return (
    <>
      <button className="search-trigger" type="button" onClick={() => setOpen(true)} aria-label="Search documentation" title="Search">
        <SearchIcon size={16} />
        <span>Search</span>
        <kbd>⌘ K</kbd>
      </button>

      {/* Portal: the fixed overlay must escape the header, whose
          backdrop-filter makes it the containing block for fixed children;
          inside it, the backdrop is header-sized and outside clicks miss it. */}
      {open && createPortal(
        <div className="search-overlay" role="presentation" onMouseDown={closeSearch}>
          <div
            ref={dialogRef}
            className="search-dialog"
            role="dialog"
            aria-modal="true"
            aria-label="Search documentation"
            onMouseDown={(event) => event.stopPropagation()}
            onKeyDown={onDialogKeyDown}
          >
            <div className="search-input-row">
              <SearchIcon size={19} />
              <input
                ref={inputRef}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={onInputKeyDown}
                placeholder="Search concepts, APIs, and guides…"
                aria-label="Search query"
              />
              <button type="button" onClick={closeSearch} aria-label="Close search">
                <X size={18} />
              </button>
            </div>
            <div className="search-results">
              <p className="search-label">
                {hasQuery
                  ? `${results.length} ${results.length === 1 ? "result" : "results"}`
                  : "Suggested pages"}
              </p>
              {results.length ? (
                results.map((doc, index) => (
                  <Link
                    key={doc.slug}
                    href={`/docs/${doc.slug}`}
                    onClick={closeSearch}
                    aria-selected={index === selected}
                    className={index === selected ? "is-selected" : undefined}
                  >
                    <FileText size={18} />
                    <span>
                      <strong>{doc.title}</strong>
                      <small>{doc.description}</small>
                    </span>
                    <ArrowRight size={16} />
                  </Link>
                ))
              ) : (
                <div className="search-empty">No matching documentation found.</div>
              )}
            </div>
            <div className="search-footer"><span><kbd>↵</kbd> Open result</span><span><kbd>esc</kbd> Close</span></div>
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}
