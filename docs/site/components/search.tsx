"use client";

import Link from "next/link";
import { ArrowRight, FileText, Search as SearchIcon, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type SearchDocument = {
  slug: string;
  title: string;
  description: string;
  text: string;
};

export function Search({ documents }: { documents: SearchDocument[] }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const closeSearch = useCallback(() => {
    setOpen(false);
    setQuery("");
  }, []);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen(true);
      }
      if (event.key === "Escape") closeSearch();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [closeSearch]);

  useEffect(() => {
    if (open) requestAnimationFrame(() => inputRef.current?.focus());
  }, [open]);

  const results = useMemo(() => {
    const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    if (!terms.length) return documents.slice(0, 6);

    return documents
      .map((doc) => {
        const title = doc.title.toLowerCase();
        const description = doc.description.toLowerCase();
        const text = doc.text.toLowerCase();
        const score = terms.reduce((total, term) => {
          if (title.includes(term)) return total + 12;
          if (description.includes(term)) return total + 5;
          if (text.includes(term)) return total + 1;
          return total - 20;
        }, 0);
        return { doc, score };
      })
      .filter(({ score }) => score >= 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 8)
      .map(({ doc }) => doc);
  }, [documents, query]);

  return (
    <>
      <button className="search-trigger" type="button" onClick={() => setOpen(true)}>
        <SearchIcon size={16} />
        <span>Search</span>
        <kbd>⌘ K</kbd>
      </button>

      {open && (
        <div className="search-overlay" role="presentation" onMouseDown={closeSearch}>
          <div className="search-dialog" role="dialog" aria-modal="true" aria-label="Search documentation" onMouseDown={(event) => event.stopPropagation()}>
            <div className="search-input-row">
              <SearchIcon size={19} />
              <input
                ref={inputRef}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search concepts, APIs, and guides…"
                aria-label="Search query"
              />
              <button type="button" onClick={closeSearch} aria-label="Close search">
                <X size={18} />
              </button>
            </div>
            <div className="search-results">
              <p className="search-label">{query ? `${results.length} results` : "Suggested pages"}</p>
              {results.length ? (
                results.map((doc) => (
                  <Link key={doc.slug} href={`/docs/${doc.slug}`} onClick={closeSearch}>
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
        </div>
      )}
    </>
  );
}
