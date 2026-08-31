Sphere Packing in a Cube (Optimization)

You are given an integer n. Place n pairwise-disjoint congruent solid spheres inside the unit cube [0, 1]^3 so that the common radius is as large as possible. You do not need to output the radius explicitly; the checker infers the largest valid common radius from your centers.

Input
The input consists of a single line with one integer:
n  (2 ≤ n ≤ 10^6 in principle; in the official tests n ≤ 4096.)

Output
Output exactly n lines. Each line must contain three real numbers x y z (in any standard real-number format parseable by C/C++, e.g., 0.0, 1, 1e-3), giving the coordinates of a sphere center. All coordinates must satisfy 0 ≤ x, y, z ≤ 1. Whitespace is free; no additional text is allowed. Trailing whitespace is ignored.

Feasibility & how the checker interprets your output
Given your centers C = {c_i}, the checker computes the largest common radius that makes the spheres non-overlapping and contained in the cube, completely ignoring any radius you might have assumed. Formally, your geometric radius is
    r(C) = min( ½ * min_{i≠j} ||c_i - c_j||_2 ,  min_i dist(c_i, ∂[0,1]^3) ),
where dist(c_i, ∂[0,1]^3) is the minimum distance from c_i to any cube face. If any coordinate lies outside [0,1] (up to an absolute tolerance of 1e−12), the output is rejected.

Scoring
This is an optimization problem. For each test case we normalize your geometric radius r(C) between a baseline lower bound and a general upper bound:
    score = clamp( (r(C) − baseline) / (best − baseline), 0, 1 ).
Here:
• baseline is the radius achieved by a balanced m×k×ℓ cubic grid with m·k·ℓ ≥ n (equally spaced centers with margin r = 1/(2·max(m,k,ℓ))). This is a constructive packing every team can implement quickly.
• best is an a priori upper bound that no packing can exceed: best = min( ½ , ((δ·3)/(4πn))^{1/3} ), where δ = π/√18 is the Kepler density upper bound for sphere packings in 3D.
Your total score is the average of per-test-case scores.

Validation & precision
• The checker reads doubles and computes in IEEE-754 double precision. It accepts coordinates on the faces of the cube.
• The checker rejects outputs that do not contain exactly n triples, that contain non-finite numbers, or that place any center outside [0,1].
• Distances are computed exactly as written above; there is no relative/absolute tolerance applied to r(C) beyond floating-point rounding.
• The checker runs in O(n^2) on your centers (only 8.4 million pairs at n=4096), so keep n modest if you test locally.

Example
Input
5

Valid Output (one of many)
0.1 0.1 0.1
0.9 0.1 0.1
0.1 0.9 0.1
0.1 0.1 0.9
0.9 0.9 0.9

(Your solution is free to produce any arrangement.)

Notes
• You only need to output centers. The checker automatically determines the maximum common radius supported by your centers.
• Greedy / local-improvement heuristics, lattice-based constructions, simulated annealing, or nonlinear optimization often yield good packings.
• Touching spheres and touching the cube faces are allowed; overlaps are not.
