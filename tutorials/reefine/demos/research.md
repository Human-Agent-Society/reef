# Research loop

The request `run.py research` files, the text of the fenced block below, one line:

```text
when I ask a research question, first search for the relevant papers, download and read them, then answer with citations
```

A working answer is a `rules` entry or a skill that says to search before answering and to cite what was read, or a `code_extension` that searches (with `fetch`) and downloads papers into the workspace before the model answers. An extension that needs a search service names it under `requires`, and `reef-pi setup` checks it off before the install.

The show session runs `reef-pi -p "what is the best known lower bound for sorting by comparisons, with a source"` in an empty directory under `work/research-<timestamp>/workspace/`. What it should do under the change: search, fetch at least one paper or reference, then answer with a citation (the bound is n log2 n comparisons in the worst case, up to lower order terms, from the decision tree argument). `run.py` prints the tool calls in order and the final answer.
