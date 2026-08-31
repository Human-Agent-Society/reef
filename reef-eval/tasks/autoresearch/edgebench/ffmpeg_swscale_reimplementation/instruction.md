# FFmpeg libswscale Re-implementation

You are a systems engineer. Your goal is to re-implement FFmpeg's **libswscale**
image scaling and pixel-format conversion library in **Zig or Rust**, producing
a C-compatible shared library that matches or exceeds FFmpeg's C scalar
reference performance using portable SIMD.

## Setup

1. FFmpeg source (libswscale + libavutil) is at `/reference/ffmpeg-src/`.
   Study the scalar C implementation — this is what you are reimplementing.
2. A full FFmpeg binary (with ASM optimisations) is at `/reference/ffmpeg`.
   Use it to generate test inputs/outputs and verify your understanding.
3. The public performance baseline library is at `/home/workspace/libswscale_public_baseline.so`.
   This wraps FFmpeg's C-only code (compiled with `--disable-asm`). Your
   implementation is benchmarked against this baseline.
4. Your workspace is `/home/workspace/swscale-impl/`. Scaffold templates for both Zig
   and Rust are provided — pick one and build from there.
5. The C API you must implement is defined in `/home/workspace/swscale_api.h`.

## Deliverable

Source code at `/home/workspace/swscale-impl/` that compiles to a shared library named
`libswscale_candidate.so`, exporting these three C-linkage functions:

```c
void *swscale_create(int src_w, int src_h, int src_fmt,
                     int dst_w, int dst_h, int dst_fmt, int algo);
int   swscale_process(void *ctx,
                      const uint8_t *const src[4], const int src_stride[4],
                      uint8_t *const dst[4], const int dst_stride[4]);
void  swscale_destroy(void *ctx);
```

The verifier will **rebuild your library from source** before testing.
Pre-built binaries without source will be rejected.

### Build commands the verifier tries (in order):

1. If `build.zig` exists: `zig build -Doptimize=ReleaseFast`
2. If `Cargo.toml` exists: `cargo build --release`
3. If `Makefile` exists: `make release`

The output library must be discoverable at one of:
- `zig-out/lib/libswscale_candidate.so`
- `target/release/libswscale_candidate.so`
- `./libswscale_candidate.so`

## Supported Pixel Formats

Your library must handle conversions between any pair of these formats:

| ID | Name     | Layout                          |
|----|----------|---------------------------------|
| 0  | YUV420P  | Planar YUV 4:2:0 (3 planes)    |
| 1  | YUV422P  | Planar YUV 4:2:2 (3 planes)    |
| 2  | YUV444P  | Planar YUV 4:4:4 (3 planes)    |
| 3  | NV12     | Semi-planar Y + UV (2 planes)   |
| 4  | NV21     | Semi-planar Y + VU (2 planes)   |
| 5  | RGB24    | Packed R,G,B (1 plane)          |
| 6  | BGR24    | Packed B,G,R (1 plane)          |
| 7  | RGBA     | Packed R,G,B,A (1 plane)        |
| 8  | BGRA     | Packed B,G,R,A (1 plane)        |
| 9  | GRAY8    | Single-plane grayscale          |

## Supported Scaling Algorithms

| ID | Name      | Description                |
|----|-----------|----------------------------|
| 0  | Nearest   | Nearest-neighbour sampling |
| 1  | Bilinear  | Bilinear interpolation     |
| 2  | Bicubic   | Bicubic interpolation      |

When `src_w == dst_w` and `src_h == dst_h`, only pixel-format conversion is
needed (no scaling). This is the most common fast path.

**Buffer alignment**: The verifier allocates all source and destination plane
buffers with 32-byte alignment. You may assume aligned loads/stores in your
SIMD code. All dimensions for subsampled formats (YUV420P, YUV422P, NV12,
NV21) will have even width and height.

## What Has to Stay Correct

The verifier checks output quality on hidden workloads:

- **Format-only conversion** (same dimensions): PSNR >= 60 dB per plane,
  or exact byte match for mathematically lossless paths (e.g. RGB<->BGR channel swap).
- **Scaling conversion** (different dimensions): PSNR >= 40 dB per plane.
- If correctness fails on any hidden workload, the score is **zero**.

## Scoring

```
if not build_ok or not correctness_ok or anti_cheat_violated:
    reward = 0.0
else:
    reward = geometric_mean(baseline_time / candidate_time)
```

A reward of 1.0 means you match FFmpeg's C scalar speed exactly.
Above 1.0 means you are faster. The SIMD opportunity is significant —
FFmpeg's own recent swscale rewrite achieved 2.6x overall speedup through
x86 SIMD backends.

## How to Work

### Starting Codebase Guidance

The Rust scaffold is already pre-populated in `/home/workspace/swscale-impl/` with a working baseline implementation. It includes exact stride-safe copy paths, packed RGB/BGR/RGBA/BGRA conversion helpers with alpha preservation, GRAY8 replication, YUV420P<->NV12/NV21 split/interleave helpers, format metadata, and a generic conversion/scaling framework.

Do not restart from the empty stub and do not switch to Zig. The Rust scaffold now starts from the verified best run-005 candidate, which reached 26/30 hidden correctness. It already fixes the broad conversion coverage: YUV420P/YUV422P/NV12/NV21 -> RGB/BGRA, RGB/RGBA/BGR -> YUV420P/YUV422P/NV12/NV21, YUV444P -> YUV420P, YUV420P -> GRAY8, bicubic upscale, RGBA bilinear resize, and exact copy/channel paths. Preserve those working paths unless a byte-level comparison proves a local change is safe.

Focus only on these four remaining hidden failures from the 26/30 candidate:

- `yuv444p -> rgb24 352x288` same-size conversion: current PSNR is about 31.86. YUV444P does not use the same unscaled fast path as YUV420P/YUV422P in FFmpeg; compare against the generic `output.c`/`yuv2rgb.c` path, including YUV table lookup, limited-range rounding, vertical filter selection, and RGB24 packed write behavior. Do not blindly add YUV444P to the YUV420P/YUV422P table fast path; that was tested and did not improve the failure.
- `rgb24 -> rgb24 1280x720 -> 640x360 bilinear`: current PSNR is about 28.91. The remaining gap is FFmpeg's default `SWS_BILINEAR` downscale filter construction from `utils.c::initFilter`, not a naive center bilinear formula. Reproduce filter positions, filter size, coefficient normalization, and integer rounding for 2:1 downscale before optimizing.
- `yuv420p -> rgb24 1280x720 -> 640x360 bilinear`: current PSNR is about 16.04. Do not convert each sampled YUV pixel directly to RGB and then scale with an RGB approximation. FFmpeg scales luma and chroma planes through separate horizontal/vertical filter chains and then performs YUV->RGB table output. Implement this as separate Y/U/V scaling to the destination geometry or a dedicated YUV420P-to-RGB24 downscale path.
- `rgb24 -> rgb24 720x576 -> 360x288 nearest`: current PSNR is about 14.92. A coordinate-only encoded probe can match `(1,1),(3,1),...`, but random inputs still fail, so the issue is not just `floor(out*src/dst)`. Compare against FFmpeg's `SWS_POINT` path with real random data and preserve its internal pipeline/filter/rounding behavior.

Your first action should be `cargo build --release && python3 /home/workspace/verify_correctness.py`, then inspect `/app/results/correctness.json` and fix one of the four workload families above at a time. Submit after a meaningful correctness improvement.

Important scaffold details already applied before you start:
- `scale_nearest_pos` is center-positioned and matches a coordinate-encoded 2:1 probe; do not revert it to floor(out*src/dst) without real random-data evidence.
- YUV420P/YUV422P/NV12/NV21 -> RGB uses a FFmpeg-like table path and currently passes hidden conversion checks exactly; avoid changing those paths globally.
- YUV->GRAY8 uses limited-range luma expansion through `yuv_luma_to_gray`; do not copy raw Y for gray output.
- RGB->YUV chroma handling has special cases that make hidden RGB/BGR/RGBA-to-YUV checks pass; validate U/V planes before changing it.

### Correctness-first implementation plan

Correctness is the hard gate. If any verifier workload fails correctness, the final score is exactly zero no matter how fast the library is. Do not spend time on SIMD or benchmark tuning until `verify_correctness.py` is clean for all visible workloads and every failure has been reduced with byte/plane diffs.

Work in this order:

1. Keep the project buildable after every edit. Prefer the scaffold/language you can finish fastest.
2. Implement exact lossless paths first:
   - same-format copy, plane by plane, respecting strides
   - RGB24<->BGR24 channel swap
   - RGB24/BGR24 to RGBA/BGRA with A=255 when alpha is introduced
   - RGBA<->BGRA and RGBA/BGRA to RGB24/BGR24 while preserving existing alpha when the destination has alpha
   - GRAY8 to RGB/BGR/RGBA/BGRA by byte replication
   - YUV420P<->NV12 and YUV420P<->NV21 split/interleave with exact UV/VU order
3. Then make same-size YUV/RGB conversions match FFmpeg scalar behavior. Avoid broad approximate BT.601 formulas unless byte-level validation proves they meet PSNR >= 60 dB. Pay attention to limited-range coefficients, rounding, saturation, chroma sample position, and planar vs semi-planar chroma addressing.
4. For RGB/RGBA/BGRA to YUV420P/YUV422P/NV12/NV21, validate chroma planes separately. Most near-miss failures come from U/V downsampling, not the Y plane. Do not use nearest chroma sampling when the reference averages or filters a block.
5. For the remaining scaling failures, follow FFmpeg's actual filter generation. For `SWS_POINT`, verify random-data output, not only coordinate-coded probes. For `SWS_BILINEAR` 2:1 downscale, implement the `initFilter`-style wider downscale kernel and fixed-point normalization/rounding rather than a two-tap center bilinear approximation.
6. For `yuv420p -> rgb24` downscale, keep color conversion and scaling isolated in the FFmpeg order: scale Y, U, and V planes with their own luma/chroma filters, then run the YUV->RGB table output.
7. Only after correctness passes should you run `run_dev_bench.py` and optimize hot loops.

### Debugging recipe for each failing workload

Use the development tools to compare against the reference before changing formulas globally:

```bash
cd /home/workspace/swscale-impl
cargo build --release || zig build -Doptimize=ReleaseFast
python3 /home/workspace/verify_correctness.py
```

For each failure, inspect `/app/results/correctness.json` and compare per-plane PSNR. A failure with Y high but U/V low means chroma subsampling/interleaving is wrong. A failure with packed RGB PSNR around 30-45 usually means range, matrix, channel order, or rounding is wrong. Scaling PSNR near 5-10 means the sampling coordinate map/filter is wrong, not just a coefficient off by one.

Prefer tiny focused probes over guessing: generate a small raw frame with `/reference/ffmpeg`, run your candidate through `/home/workspace/pixel_formats.py`, and print first differing bytes plus per-plane max/mean error. Fix one workload class at a time and rerun correctness before adding performance code.

### Build and test cycle:

```bash
# Pick your language and start from the scaffold
cp -r /home/workspace/scaffold/zig/* /home/workspace/swscale-impl/   # or /home/workspace/scaffold/rust/*

# Build
cd /home/workspace/swscale-impl
zig build -Doptimize=ReleaseFast               # or: cargo build --release

# Test correctness against FFmpeg reference
python3 /home/workspace/verify_correctness.py

# Benchmark against the public baseline
python3 /home/workspace/run_dev_bench.py
```

### Pre-generated test media:

Test images are pre-generated at `/home/workspace/media/` in various formats and sizes
(gradients, colour bars, noise). Use these for quick iteration:

```bash
ls /home/workspace/media/             # See available test images
cat /home/workspace/media/manifest.json  # Format, size, path metadata
```

### Use the reference FFmpeg binary for experiments:

```bash
# Generate a test pattern
/reference/ffmpeg -f lavfi -i testsrc=duration=1:size=1920x1080:rate=1 \
    -frames:v 1 -f rawvideo -pix_fmt yuv420p /tmp/test_yuv420p.raw

# Convert between formats (use pre-generated media or your own)
/reference/ffmpeg -f rawvideo -pix_fmt yuv420p -s 640x480 \
    -i /home/workspace/media/gradient_640x480_yuv420p.raw \
    -f rawvideo -pix_fmt rgb24 /tmp/gradient_rgb24.raw
```

**Note:** `verify_correctness.py` is a development-only tool that uses
`/reference/ffmpeg` to generate golden outputs. The reference binary is
deleted before final scoring — the actual verifier compares your output
against the baseline library instead.

### Study the FFmpeg source:

```bash
# The scalar C implementation you are competing against
ls /reference/ffmpeg-src/libswscale/
# Start with: swscale.c (entry point), swscale_internal.h (structures)
# Look for the pixel conversion and scaling functions in the directory
```

### Key reference files:

- `/home/workspace/swscale_api.h` — The C API your library must export
- `/home/workspace/pixel_formats.py` — Pixel format metadata, plane geometry helpers,
  and the ctypes loading code the verifier uses to call your library

## Constraints

You CAN:
- Use Zig's `@Vector` SIMD or Rust's `std::simd` / `std::arch` intrinsics
- Use any algorithm or data structure for the conversion
- Create helper files and modules
- Pre-compute lookup tables and filter coefficients in `swscale_create`

You CANNOT:
- Wrap, exec, or dlopen the reference FFmpeg binary or its libraries
  (the reference is **deleted before verification**)
- Access `/tests/` or any hidden verifier files
- Use inline assembly (the task tests portable SIMD, not hand-tuned ASM)
- Download external code (no internet access)

### Suggested phases:
- Study FFmpeg scalar source, understand YUV<->RGB maths and scaling filter
  generation, and set up your project scaffold.
- Implement core format conversions (YUV420P<->RGB24 first) and get
  `verify_correctness.py` passing for basic cases.
- Add scaling (bilinear at minimum), then SIMD optimisation of hot conversion
  loops.
- Expand format coverage and benchmark against the baseline.
- Finish with a final correctness sweep, edge cases, and cleanup.

Keep a **building and working** library at all times. A library that handles
60% of conversions correctly at 1.2x speed is much better than one that
doesn't compile.

## Behavioral Rules

- Never stop to ask. Work autonomously until interrupted.
- Check time regularly before starting large refactors.
- Keep your library buildable at all times.
- Test against the reference FFmpeg frequently.
- Optimise for breadth of format coverage first, then depth of SIMD optimisation.

## Current Scaffold Baseline

The Rust scaffold already matches the hidden correctness suite at 30/30 when built and judged against the current verifier. It scored 0.506321 in local hidden-judge validation. Do not restart from scratch and do not replace the fixed-point compatibility paths unless you preserve their behavior.

Known correctness-critical paths already covered:
- `yuv444p -> rgb24` same-size conversion uses FFmpeg's full-chroma `yuv2rgb_write_full` style fixed-point coefficients and signed 30-bit clipping behavior.
- `rgb24 -> rgb24` 2:1 nearest and bilinear downscale use FFmpeg's internal RGB-to-Y/UV pipeline, including half-width RGB chroma input and the observed libswscale h/v filter coefficients.
- `yuv420p -> rgb24` 2:1 bilinear downscale scales Y and chroma planes separately with the observed libswscale 4-tap filters before RGB output.

Your priority is performance improvement while keeping all public and hidden correctness passing. Before optimizing a path, preserve the fixed-point rounding, border coefficients, chroma subsampling decisions, and full-chroma output behavior in the scaffold.
