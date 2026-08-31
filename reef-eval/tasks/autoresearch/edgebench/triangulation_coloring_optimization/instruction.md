## Triangulation Coloring - Minimize Ugliness (CodeChef TRICOL)

Write `solution.py` in the project root (stdin → stdout). All modifications must be made only in `solution.py`.

Read `README.md`, `tools/README.md`, and **`sample_data/README.md`**. Official scoring: flips change the triangulation; **U** (ugly triangles) is counted on the **final** triangulation after all flips.

---

## Evaluation

- **10 hidden judge cases**, each **N=512**
- **G = X·C + Y·F + U²** per case; **total score = sum** (lower is better)
---

## Local Testing

Use the **3 pre-built N=512 files** in `sample_data/`. Use only these pre-built cases; do **not** call `tools/src/gen.py` or generate random local cases.

```bash
./eval_sample_data.sh
# or per file:
python3 solution.py < sample_data/0000.txt > output.txt
./tools/bin/tester sample_data/0000.txt output.txt
```

Also try `sample_data/0001.txt` and `sample_data/0002.txt`.

When `./eval_sample_data.sh` improves, **submit** for all 10 hidden judge cases.

---

## Strategy (algorithm exploration)

Focus on improving **`solution.py`** using `sample_data/` for feedback:

1. **Joint optimization** — recolor and flip interact; optimize them together, not in tiny fixed phases.
2. **U² dominates** on hard cases — reducing ugly triangles by even a few often beats tweaking C/F.
3. **Iterate**: change algorithm → `./eval_sample_data.sh` → submit when promising.
4. Do **not** loop local tests endlessly; move to submit after a few sample checks.

---

## Rules

- Only modify `solution.py`; do **not** change `tools/` or `sample_data/`
- Output: line 1 `C F`, line 2 color string (length N), then `F` flip lines (1-indexed vertex pairs)
- The local `tester` applies flips exactly like the judge
