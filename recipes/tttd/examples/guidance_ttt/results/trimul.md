# TriMul case study: reuse the gated tile across output blocks

The TriMul run reduced the search-time latency from `10,177.40 µs` to
`1,110.85 µs`. At update 17, Guidance-TTT selected a parent measured at
`1,180.62 µs` and proposed:

> Optimize the C=384 output pipeline by introducing a shared-memory caching
> layer for the contraction output. Reorganize the [B,H,N,N] contraction result
> into a [B,N,N,H] layout with H split into 3 blocks of 128, and cache each
> H-block in shared memory during the LayerNorm + gate application. This reduces
> redundant global memory reads for the three C-blocks by overlapping
> computation with cached data, while maintaining FP16 intermediates and fusing
> the final linear projection with the output transpose. Target the H100's high
> shared memory bandwidth (96 KB per SM) to amortize the contraction output's
> memory footprint.

The executor preserved the optimization target but changed the mechanism. It
replaced a two-kernel output path with one fused Triton kernel, kept the gated
tile live in registers, and reused it in a static loop over the three C blocks.
This removed the full intermediate tensor's global write and reload. The
implementation did not literally add the proposed shared-memory layer; it
found a more direct way to realize the requested data reuse.

The child measured `1,128.03 µs`, 4.45% below its direct parent. The connection
is specific: the guidance identifies repeated movement of the contraction
output, and the child removes that movement at the named C=384 output stage.

The run's eventual search-time best was `1,110.85 µs` at update 25. For the
fixed final kernel, three sequential repeats on one H100 under CUDA 12.8,
PyTorch 2.7.1, and Triton 3.3.1 measured `1,155.29`, `1,157.47`, and
`1,162.62 µs`, for `1,158.46 ± 3.76 µs`. All correctness checks passed. The
repeat result is reported separately because GPU timing noise makes the
single best search observation optimistic.

Archive identifiers:

```text
entry  446edddc-af97-4883-8cf6-520d64a641c4
node   650fa7bf-4005-4079-aa95-716f6399ff01
update 17
```
