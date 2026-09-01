# Polyomino case study: ordering pieces against the current skyline

The Polyomino run started at `27.8105` and reached `89.7965` after 30 Reef
updates. At update 3, Guidance-TTT selected a parent scoring `80.1328` and
produced the following guidance:

> Implement a dynamic polyomino ordering strategy during skyline packing, where
> the processing order of polyominoes is determined based on their geometric
> compatibility with the current skyline state. For each candidate width, sort
> polyominoes using a heuristic that prioritizes pieces with bounding boxes that
> closely match the skyline's column height variations, such as those with
> lower aspect ratios or that fit within narrow vertical gaps. This could
> improve snugness and reduce wasted space compared to static orderings like
> largest-first.

The executor turned that idea into an additional skyline phase. At every
placement, it scored every remaining combination of piece, orientation, and
horizontal position against the current column heights. The choice favored a
low resulting height, little empty space below the piece, and a close match
between the piece's lower contour and the skyline. Existing static skyline and
grid-based packing paths remained available.

The child scored `86.7646`, a gain of `6.6318` points over its direct parent.
This transition is useful because the guidance names a decision that the
current state should change: piece order. The implementation exposes that
decision inside the packing loop instead of adding another fixed ordering.
The verifier then measures the complete program, including the retained
fallbacks, on the same 70-case suite.

This pair does not isolate dynamic ordering as an ablation. The executor wrote
the child as a complete program, and the score applies to that program. It does
show that a state-dependent search instruction survived the handoff from the
guidance model to the executor and produced a verified child that scored
`6.6318` points higher.

Archive identifiers:

```text
entry  7223721c-9237-4f03-b4d9-a39ae7ff0d36
node   c4e1a74c-31a3-47e4-9450-c15cf79baa91
update 3
```
