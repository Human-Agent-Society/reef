2D Rectangular Knapsack with (Optional) 90° Rotations — Official Problem Statement (Hard Set)
============================================================================================

Summary
-------
You are given a single rectangular **bin** and several **item types**. Each item type t has
dimensions (w_t, h_t), profit v_t, and an availability limit L_t. You may place zero or more
axis‑aligned rectangles (items) of these types into the bin, without overlap and fully inside
its boundary. In some tests, 90° rotation of items is allowed; in others it is not. Your goal is
to maximize the **total profit** of the placed items.

This is an **optimization** problem with **partial scoring**. Your program prints a **feasible packing**
(i.e., a list of placements). The judge validates feasibility and computes your score relative to
reference values for that test case.


1) Formal Model
---------------
• The bin is a rectangle of integer width W and height H, with its bottom‑left corner at (0, 0).
  Coordinates increase to the right (x) and up (y).

• There are M item types. Each type t has:
  – width  w_t  (integer, ≥ 1)
  – height h_t  (integer, ≥ 1)
  – profit v_t  (integer, ≥ 0)
  – limit  L_t  (integer, ≥ 0), the maximum number of copies you may place.

• A **placement** is a tuple (type_id, x, y, rot) where:
  – type_id is the string id of some item type t,
  – (x, y) are **integers** giving the bottom‑left corner of the placed rectangle,
  – rot ∈ {0, 1}. If rot = 1 and rotation is allowed in this test, the item is rotated by 90°
    so its realized size is (w′, h′) = (h_t, w_t). If rot = 0, size is (w′, h′) = (w_t, h_t).
    If rotation is **not** allowed in this test, then rot must be 0 for every placement.

• **Feasibility constraints**:
  – **Inside the bin**: 0 ≤ x and 0 ≤ y and x + w′ ≤ W and y + h′ ≤ H.
  – **Non‑overlap**: we use the **half‑open rectangle** convention. A placement occupies the set
    [x, x + w′) × [y, y + h′). Two placements are considered non‑overlapping iff their intervals
    are disjoint on **at least one** axis. In particular, **touching at edges or corners is allowed**.
  – **Limits**: for each type t, the number of placements with type_id = t must be ≤ L_t.

• Objective: maximize the sum of profits v_t over all placed items.


2) Input Format (read from stdin)
---------------------------------
The input is a single JSON object with exactly two keys: "bin" and "items".

• "bin" is a JSON object:
  { "W": <int>, "H": <int>, "allow_rotate": <true|false> }

• "items" is a JSON array of item‑type objects. Each item object has **exactly** the following keys:
  {
    "type":  "<id-string>",
    "w":     <int>,
    "h":     <int>,
    "v":     <int>,
    "limit": <int>
  }

Additional or missing keys are considered invalid.

Example (abridged):
{
  "bin":   {"W": 1399, "H": 1699, "allow_rotate": true},
  "items": [
    {"type":"A","w":33,"h":89,"v":4237,"limit":120},
    {"type":"B","w":91,"h":55,"v":8121,"limit":60}
    // ... more types ...
  ]
}


3) Output Format (write to stdout)
----------------------------------
Your program must print a single JSON object with exactly one key: "placements".

• "placements" is a JSON array. Each element is a placement object with **exactly** these keys:
  { "type": "<id-string>", "x": <int>, "y": <int>, "rot": <0|1> }

Notes:
• The order of placements does not matter.
• Do **not** add extra keys; the checker will reject unknown keys.
• It is valid to output an empty array if you choose to place nothing.

Example (feasible output):
{
  "placements": [
    {"type":"B","x":0,"y":0,"rot":1},
    {"type":"A","x":91,"y":0,"rot":0},
    {"type":"A","x":124,"y":0,"rot":0}
  ]
}


4) Feasibility Details and Edge Cases
-------------------------------------
• **Half‑open geometry**: A rectangle occupies [x, x+w′)×[y, y+h′). Therefore:
  – Two rectangles with x + w′ = x′ (or y + h′ = y′) **do not overlap** (touching is allowed).
  – The checker’s sweep‑line treats rectangles that **end** at x before inserting rectangles that
    **start** at the same x. This matches the half‑open convention.

• **Rotation permission**: If "allow_rotate" is false in the input, every placement must have rot = 0
  (no rotation). If "allow_rotate" is true, you may choose rot ∈ {0, 1} per placement.

• **Limits**: The checker counts how many times each type appears in your placements and rejects outputs
  that exceed the per‑type limit L_t.

• **Coordinate system**: (0,0) is the bin’s bottom‑left corner. x increases to the right, y upward.
  Coordinates and sizes are integers throughout.

• **Validation**: The checker rejects outputs that are not valid JSON or that violate any of the rules
  above (unknown types, out‑of‑bounds, overlap, invalid rot, extra/missing keys).


5) Constraints (Hard Set for this Round)
----------------------------------------
The official hidden tests in this round follow these ranges:

• Bin:
  – 900 ≤ W, H ≤ 2000
  – Some tests use near‑prime/co‑prime‑like dimensions to discourage trivial tilings.

• Items:
  – Number of types M: 8 ≤ M ≤ 12
  – Dimensions: 7 ≤ w_t, h_t ≤ ⌊0.6 · max(W, H)⌋ and w_t ≤ W, h_t ≤ H
  – Profit: 1 ≤ v_t ≤ 10^9
  – Limit: 1 ≤ L_t ≤ 2000
  – The distribution mixes high‑density but supply‑limited types with awkward aspect ratios
    and strip‑like pieces. Rotation is disabled in some tests.

• Output size: Near‑optimal solutions typically use O(10^2–10^3) placements per test.
  (There is no explicit hard cap, but extremely large outputs may risk time limits.)

• Time & memory limits: see Section 8.


6) Scoring
----------
For each test, the judge computes:
• V = your total profit (sum of v_t for all placements).
• B = a **lower bound** (baseline value) — a simple shelf heuristic without rotation, in input order.
• K = an **upper bound** (“best” for this round) — a fractional area fill that respects per‑type limits
      but ignores geometry (optimistically packs by value density).

Your per‑test ratio is:
  ratio = clamp( (V − B) / (K − B), 0, 1 )

Corner case: if K ≤ B, then ratio = 1 if V ≥ K, else 0.

Your problem score is the **average** ratio over all tests. Future rounds may tighten how B and/or K
are computed; you do not need to output B or K.


7) Common Mistakes That Cause Wrong Answer
------------------------------------------
• Using rot = 1 when "allow_rotate" is false.
• Overlapping rectangles (especially at shared edges): remember we use **half‑open** geometry; touching
  is allowed, overlap is not.
• Out‑of‑bounds placements: x + w′ must be ≤ W and y + h′ ≤ H.
• Exceeding per‑type limits L_t.
• Output JSON not exactly following the schema (wrong top‑level key, extra keys in placements,
  missing keys, non‑integer x/y/rot).


8) Limits
---------
• Time limit: 1 second
• Memory limit: 512 MB
• Number of test cases: 15
(These values correspond to the current hard set and match the contest configuration.)


9) Reference Implementations and Hints (Non‑binding)
----------------------------------------------------
You may find success with:
• Maximal‑rectangles or skyline‑based packers, combined with multiple item orderings (by value density,
  by height/width/area), and local repair.
• Consider that the highest value density items are **supply‑limited**; good solutions typically mix
  several types and use strips/slenders to close narrow leftovers.
• When rotation is disabled, favor orderings that reduce fragmentation in one orientation.


10) Small Worked Example
------------------------
Input:
{
  "bin": {"W": 10, "H": 6, "allow_rotate": true},
  "items": [
    {"type":"a","w":4,"h":3,"v":10,"limit":3},
    {"type":"b","w":3,"h":2,"v":6,"limit":10}
  ]
}

One feasible output (not necessarily optimal):
{
  "placements": [
    {"type":"a","x":0,"y":0,"rot":0},
    {"type":"a","x":4,"y":0,"rot":0},
    {"type":"b","x":8,"y":0,"rot":1},
    {"type":"b","x":8,"y":2,"rot":1},
    {"type":"b","x":0,"y":3,"rot":0},
    {"type":"b","x":3,"y":3,"rot":0},
    {"type":"b","x":6,"y":3,"rot":0}
  ]
}

These placements are inside the bin, respect limits, and do not overlap under the half‑open convention.


11) Compliance Checklist (Before You Submit)
--------------------------------------------
[ ] Output exactly one top‑level key: "placements".
[ ] Each placement contains exactly the keys "type", "x", "y", "rot".
[ ] All coordinates and rot are **integers**; rot ∈ {0,1}.
[ ] When rotation is disabled in the test input, use rot = 0 everywhere.
[ ] No placement exceeds the bin boundary.
[ ] No two placements overlap (touching at edges/corners is OK).
[ ] Per‑type counts do not exceed L_t.
[ ] Your JSON is syntactically valid (commas, quotes, etc.).


12) Clarifications
------------------
• “Half‑open”: a rectangle [x, x+w′)×[y, y+h′) includes all integer points with x ≤ X < x+w′ and
  y ≤ Y < y+h′. Two rectangles that meet at a vertical or horizontal line **do not** overlap.
• There is **no requirement** to fill the bin or to use all limits; you may place any multiset
  that satisfies the constraints.
• Rotations are exactly 90°; no other angles are allowed. Rotating swaps width and height.
• Profits, coordinates, and counts are within 64‑bit signed integer ranges in all official tests.


End of statement.
