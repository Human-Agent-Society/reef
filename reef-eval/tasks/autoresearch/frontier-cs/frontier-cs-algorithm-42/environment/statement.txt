Problem
You are given an integer n. Place n unit squares (side length 1) inside an axis-parallel square container of side length L so that:
- Every unit square lies entirely inside the container.
- Unit squares have no common interior points (touching edges/corners is allowed).
- Each unit square may be rotated by an arbitrary angle.

Your goal is to minimize L.

Input
A single integer n (1 ≤ n ≤ 100000).

Output
- First line: a real number L (the claimed container side length).
- Next n lines: three real numbers xi, yi, ai for i = 1..n:
  - (xi, yi) is the center of the i-th unit square.
  - ai is its rotation angle in degrees counterclockwise, 0 ≤ ai < 180.

All numbers must be finite reals. Any reasonable precision is allowed; at least 6 decimal places is recommended.

Validity
Your output is valid if and only if:
- Containment: Every point of every unit square lies inside [0, L] × [0, L].
- Non-overlap (interiors): The interiors of any two unit squares are disjoint. Touching along edges or at corners is allowed.
- Angles: 0 ≤ ai < 180 for all i.

The judge verifies geometry with epsilon = 1e-7:
- Containment: A square is accepted if its maximum outward violation beyond the container is ≤ 1e-7.
- Non-overlap: Two squares are accepted as interior-disjoint if the minimum signed distance between their boundaries is ≥ −1e-7.

Invalid submissions score 0 for that test.

Goal
Minimize L subject to the validity constraints.

Baseline
A simple baseline packs all squares unrotated (ai = 0) on the unit grid, with centers at (0.5 + x, 0.5 + y), using the smallest integer-sided container that fits all n. The baseline side length is
- L0(n) = ceil(sqrt(n)).

For example, for n = 11, the baseline uses L0 = 4.000000.

Scoring
Let L be your reported container side length (validated by the judge). Define:
- LB = sqrt(n)  (trivial area lower bound; no packing can have L < LB),
- L0 = ceil(sqrt(n))  (baseline side length),
- s = s(n)  (reference scale defined below; s satisfies LB ≤ s ≤ L0).

The score is computed as follows:
- If invalid: 0 points.
- If L ≥ L0: 1 points.
- If L = LB: 100 points.
- If LB < L ≤ s:
  - Let p2 = (s − L) / (s − LB) ∈ (0, 1].
  - Score = 95 + 5 × min(1.0, 1.1 × p2).
- If s < L < L0:
  - Let p1 = (L0 − L) / (L0 − s) ∈ (0, 1).
  - Score = 94 × min(1.0, 1.1 × p1) + 1.

This scheme:
- Gives 0 points at the baseline (L = L0).
- Reaches at least 95 points once you meet the reference s(n) (i.e., L ≤ s(n)).
- Reaches 100 points at the area bound (L = LB).
- Applies a +10% amplification to progress within each band, capped at the band’s ceiling (95 in the upper band, 100 in the lower band), to enhance differentiation while keeping anchors fixed.

Reference scale s(n), we define s(n) as the best human score for n <= 100, and then s(n) = 2 * s(ceil(n / 4)) for n > 100. 

Notes on scoring
- Baseline (L = L0): 1 points.
- Meeting s(n): at least 95 points.
- Area bound (L = LB): 100 points.
- Scores vary smoothly between these anchors.
- The +10% amplification is applied within each band and capped at that band’s ceiling (95 or 100) to increase separation among close solutions without exceeding the anchors.

Time limit
1 second

Memory limit
512 MB

Sample
Input
11

Output
4.000000
0.500000 0.500000 0.000000
1.500000 0.500000 0.000000
2.500000 0.500000 0.000000
3.500000 0.500000 0.000000
0.500000 1.500000 0.000000
1.500000 1.500000 0.000000
2.500000 1.500000 0.000000
3.500000 1.500000 0.000000
0.500000 2.500000 0.000000
1.500000 2.500000 0.000000
2.500000 2.500000 0.000000

(The sample is a valid baseline packing for n = 11 with L = 4.000000.)

Additional clarifications
- Unit squares: Side length exactly 1, centered at (xi, yi), rotated by ai degrees around the center.
- Ordering: Squares may be listed in any order.
- Precision: Inputs are read as doubles; small rounding differences are tolerated per epsilon.
- Touching: Squares may touch each other and the container boundary; only interior overlap is forbidden.

Design notes (for organizers and contestants)
- Why this scoring? It anchors 0 at the simple baseline L0, assigns ≥95 once a high-quality reference s(n) is matched (using best-known/curated targets for n ≤ 100 and a recursion for larger n), and caps at 100 at the area bound LB. A +10% amplification within each band enhances differentiation while keeping the anchor points fixed.
- Existence: For fixed n and L, the feasible set (positions and angles satisfying containment and non-overlap) is closed and bounded. Minimizers exist by compactness; hence a minimal container side length exists for each n.
- Implementation tip: The baseline generator achieves L = ceil(sqrt(n)) with unrotated squares on a grid. Heuristics (local search, small rotations, nonlinear optimization with overlap penalties) often reduce L below the baseline for many n.