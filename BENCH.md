# Video pipeline benchmarks

Benchmarks for the browser video → anaglyph pipeline (`js/video-pipeline.js`,
`video.html`) against the native `anaglyph-rs` CLI, plus per-stage bounds.

## Running

```sh
npm run build:wasm                 # current wasm for the browser side (SIMD)
npm run build:wasm:scalar          # baseline build without wasm SIMD (A/B)
node bench/run-web-bench.mjs       # browser matrix + native CLI
node bench/run-web-bench.mjs --iters 5 --gpu
node bench/run-web-bench.mjs --browser firefox --skip-native
node bench/run-web-bench.mjs --webm --codec avc   # same clips, one codec family
node bench/run-web-bench.mjs --browser firefox --latency realtime --hw off
```

- `--browser firefox` benchmarks Firefox headless (Playwright); it swaps in
  the open-codec WebM clip set (Playwright's Firefox carries no proprietary
  H.264 codecs). `--webm` forces the same WebM set on any browser for
  apples-to-apples cross-browser runs.
- `--codec avc|vp9|vp8|av1` pins every run's encoder preference order to one
  family; `--latency quality|realtime` sets the encoder latency mode;
  `--hw off` disables both hardware codec preferences. All three are also
  selectable in `bench-video.html`.
- Results land in `bench/results/web-<timestamp>.json` (raw, per-iteration)
  and `bench/results/latest.md` (summary table). `bench/results/` is
  git-ignored: numbers are machine-specific by nature.
- Diff two runs (before/after a change):
  `node bench/compare.mjs bench/results/web-A.json bench/results/web-B.json`
- Everything browser-side can also be driven interactively via
  `bench-video.html` (clip/direction/backend/hardware/latency/codec pickers,
  results table + JSON) after `npm run serve`.

The native rows need the CLI built:
`cd ../anaglyph-rs && cargo build --release --features video`.

### Headless caveat (important)

The default run uses Playwright's headless Chromium. In containers or on
machines without a usable GPU it has **no WebGPU adapter and no H.264
hardware codecs**, so browser rows measure software decode + CPU/SwiftShader
blend + software encode, while the native rows usually use VAAPI/hardware
encode — i.e. not an apples-to-apples browser-vs-native comparison. For real
hardware numbers run `--gpu` (full chromium, `headless=new`) on a desktop GPU
with WebGPU, or drive `bench-video.html` in a normal browser window; the JSON
`webMeta` block records the user agent, WebGPU adapter, and CPU info so runs
can be attributed.

## What the columns mean

- **median pairs/s** — full conversions (demux → decode → pair → blend →
  encode → mux), median over `--iters` iterations. Iteration 1 includes
  warm-up (codec init, shader compile), which the median mostly absorbs.
- **decode-only fps** — demux + `VideoDecoder`, frames discarded: the ceiling
  any pipeline stage other than blend can reach.
- **encode-only fps** — synthetic RGBA frames straight into the planned
  encoder: the encoder's own ceiling (synthetic content compresses easier
  than video, so read it as an upper bound). Distinguishes a slow encoder
  from slow submission plumbing.
- **blend-only fps** — wasm `VideoBlender.blend` on synthetic frames with no
  codecs: the blend stage's standalone throughput.
- **stage columns** (`copy/blend/enc-wait/dec-wait ms`) — summed time inside
  each step per run. Stages overlap asynchronously, so these are attribution,
  not a wall-clock split.

## Baseline (2026-11-07, wasm SIMD enabled)

Software-only container: headless browsers (no WebGPU adapter, no hardware
codecs), CPU blend both sides, 480x270 clips from the **WebM set** so both
browsers chew the same input; median of 3. Native CLI rows use VAAPI h264
encode. This is a *stage-shape* baseline, not hardware-vs-hardware.

| browser | clip | direction | encoder chosen | median pairs/s | decode-only fps | encode-only fps | blend-only fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chromium | booth2s-vp9 | clockwise | avc1.640029 | 1232 | 5633 | 6416 | 2350 |
| Chromium | booth2s-vp9 | auto | avc1.640029 | 882 | 5472 | 6550 | 2350 |
| Chromium | boomerang-vp9 | clockwise | avc1.640029 | 1268 | 7301 | 6673 | 2349 |
| Chromium | boomerang-vp9 | auto | avc1.640029 | 881 | 7174 | 6805 | 2388 |
| Firefox | booth2s-vp9 | clockwise | avc1.640029 | 330 | 3706 | 940 | 2648 |
| Firefox | booth2s-vp9 | auto | avc1.640029 | 320 | 3780 | 908 | 2984 |
| Firefox | boomerang-vp9 | clockwise | avc1.640029 | 339 | 3979 | 979 | 2480 |
| Firefox | boomerang-vp9 | auto | avc1.640029 | 333 | 4021 | 972 | 3016 |

### SIMD blend A/B (same container, Chromium, WebM set)

`npm run build:wasm:scalar` rebuilds without `simd128` (`blend_is_simd()`
reports false and the e2e equivalence case flags it; the SIMD loop is
bit-identical to the scalar reference, checked byte-for-byte in
`bench/e2e-video.mjs` for all anaglyph types):

| metric | scalar wasm | SIMD wasm | gain |
| --- | --- | --- | --- |
| blend-only fps | 951 | 2350 | 2.5x |
| pipeline clockwise (booth2s) | 864 | 1232 | +43% |
| pipeline auto (booth2s) | 633 | 882 | +39% |
| pipeline clockwise (boomerang) | 892 | 1268 | +42% |

Reading the current baseline:

- **Chromium is producer-bound** (~1230/s sits below its 6400/s encode-only
  ceiling; stage time goes to copy + blend + decode-wait). Auto mode pays
  the motion estimate (~0.4 ms/pair).
- **Firefox is encoder-bound**: its pipeline (~330/s) runs at the software
  H.264 encoder's real-content rate even though synthetic encode-only hits
  ~940/s. All four codec families were measured head-to-head on Firefox
  software: avc ~335/s > vp8 ~180/s > vp9 ~58/s > av1 ~47/s, so the default
  H.264-first order is also the fast order; `latencyMode: realtime` made no
  measurable difference. On machines where Firefox exposes a hardware
  encoder (Media Foundation / VideoToolbox) `prefer-hardware` grabs it and
  the gap to Chrome largely closes — the report line `encoder:` shows which
  one was picked; the Acceleration checkboxes in `video.html` force software
  for comparison.
- **WebGPU blend is off by default — CPU SIMD wins end to end.** Every pair
  crosses CPU memory twice regardless (decoder `copyTo` out, `VideoFrame`
  into the encoder), so GPU blending adds an upload + `mapAsync` readback
  round-trip per pair without saving the copies it was meant to replace.
  That holds on SwiftShader in this container **and on real desktop GPUs**
  (user-reported: pipeline fps drops with the WebGPU checkbox on). The
  toggle remains to A/B on your hardware: run one conversion with it off,
  one on, and compare the report's fps.

### Copy/emit overlap + raw NV12 frame path (2026-11-08, same container)

Two pipeline changes: decoder-frame copying runs on its own promise chain
*ahead* of pair emission (bounded by `COPY_AHEAD_LIMIT = 3` frame buffers),
so each frame's frame extraction overlaps the previous frames' blend and
encode-wait instead of serializing behind them; and on engines whose
`copyTo` only yields native planes (WebKit), the planes now go through the
wasm `YuvFrameConverter` (`frameCopy: "NV12 → RGBA (wasm)"`) instead of a
canvas raster. Motion analysis also gained a direct-RGBA downsample (no
per-frame RGB24 packing pass), bit-identical drift values (tested):

| browser | clip | direction | median pairs/s | was |
| --- | --- | --- | --- | --- |
| Chromium | booth2s-vp9 | clockwise | 1238 | 1232 |
| Chromium | booth2s-vp9 | auto | 926 | 882 |
| Chromium | boomerang-vp9 | clockwise | 1256 | 1268 |
| Chromium | boomerang-vp9 | auto | 943 | 881 |
| Chromium | booth2s (h264) | clockwise | 1642 | 1185 |
| Chromium | booth2s (h264) | auto | 1120 | 810 |
| Firefox | booth2s-vp9 | clockwise | 337 | 330 |
| Firefox | booth2s-vp9 | auto | 328 | 320 |
| Firefox | boomerang-vp9 | clockwise | 343 | 339 |
| Firefox | boomerang-vp9 | auto | 333 | 333 |
| WebKitGTK | booth2s-vp9 | clockwise | 269 | 259–285 (canvas) |
| WebKitGTK | booth2s-vp9 | auto | 237 | 249 (canvas) |
| WebKitGTK | boomerang-vp9 | clockwise | 274 | — |
| WebKitGTK | boomerang-vp9 | auto | 240 | — |

Median of 3 (Chromium/Firefox) or 5 (WebKit, whose run-to-run variance on
this box runs ±10% — never compare single WebKitGTK iterations). Motion-only
improved to ~2590 fps (from ~2260); in-pipeline motion cost fell to
~0.29 ms/pair.

Reading the overlap results:

- **The overlap pays exactly where the pipeline is producer-bound.** On
  Chromium with H.264 clips it is worth +38% (1185 → 1642 clockwise); on
  VP9 clips the same pipeline barely moves (1232 → 1238) because software
  VP9 decoding saturates the CPU's *cores* — overlapping stages on one
  main thread cannot add compute that the decoder is already eating. The
  H.264 `decode-only` ceiling with idle cores is where short per-stage
  round trips decide the rate, and that is what serialization was
  wasting.
- **The auto-mode penalty is uniform wasm work, not an engine gap.**
  Motion costs ~0.29 ms/pair in every browser (motion-only: ~2590 fps on
  all three engines). Because Chromium's fixed-pairing wall is ~0.6
  ms/pair, that constant shows up as a −30% relative hit there but ~−3%
  on Firefox (337 → 328) — the "auto widens the gap" effect is ratio
  arithmetic over engines whose ceilings differ 5x.
- **Firefox stays encoder-bound** (enc-wait ~1.7 ms/pair): the overlap
  absorbs frame copying behind encode-wait, worth a few percent; the real
  ceiling is the software H.264 encoder.
- **WebKitGTK is pinned to its copyTo latency floor** (see engine notes):
  copy ≈ 2.8 ms/pair dominates the serial copy chain. The raw wasm plane
  path and the canvas raster measure statistically tied on WebKitGTK
  (~269 vs 259–285); the raw path's wins are correctness (deterministic
  BT.601/709 matrix instead of engine-default raster; e2e pins 38.4 dB
  agreement with Chromium) and removing the canvas allocation/raster hop
  — a real Safari/iOS GPU-memory win is plausible but unmeasurable here.

Native CLI (`-b cpu`, VAAPI h264 encode, wall-clock incl. process start):

| clip | direction | frames/s |
| --- | --- | --- |
| booth2s | clockwise | 451 |
| booth2s | auto | 886 |
| booth2s-aac | clockwise | 903 |
| booth2s-aac | auto | 666 |
| boomerang | clockwise | 1260 |
| boomerang | auto | 1184 |

Native runs on 5-second clips are dominated by process/encoder startup;
the 11-second boomerang shows the steady state (~1200 frames/s).

## Engine notes (WebKit / Safari / iOS)

Verified against WebKitGTK 2.54.2 (GNOME Web, driven by
`bench/e2e-webkit.mjs` — same engine family as iOS Safari):

- **`VideoFrame.copyTo` ignores `format`**: an "RGBA" copy returns the
  frame's native NV12/I420 planes (whole coded frame, whatever was asked
  for) plus a non-spec `[{offset, stride}]` return value. Reading that as
  RGBA produced the garbled gray output on iOS. The pipeline probes one
  frame (`detectFrameCopy`): browsers with spec conversion use plain
  `copyTo`; browsers that only yield native planes feed them through the
  wasm `YuvFrameConverter` (both buffers in wasm memory — `copyTo` writes
  planes straight in, one pass converts YUV→RGBA and crops to the visible
  rect; `frameCopy: "NV12 → RGBA (wasm)"`), falling back to canvas raster
  only if the planes or the wasm scratch are unusable. The e2e suite pins
  the raw path on WebKit and checks its output against the Chromium
  reference at ≥30 dB PSNR (measured 38.4 dB).
- **`copyTo` is a fixed ~1.4 ms round trip regardless of destination
  size** (4×2-rect copies cost the same as full-frame; concurrent copies
  do not overlap — measured in `wk-copybench`-style probes). That is a
  latency floor, not throughput: at 480p270, ~2.8 ms/pair of in-pipeline
  copy (~1.4 ms round trip + wasm convert, contended with the decoder)
  caps WebKitGTK near ~270 pairs/s no matter what the blend does. On real
  Safari/iOS the planes may sit in GPU memory and behave differently —
  the raw path is still the right shape there (no CPU raster), but treat
  the absolute numbers as WebKitGTK's.
- **Color space**: WebKit reports `primaries: "unspecified"` (non-spec) and
  echoes it into the encoder's first-chunk metadata; mediabunny rejects it.
  `colorSpaceInit`/`sanitizeChunkMeta` gate both paths to muxable values.
  The raw path picks the YUV matrix from `colorSpace.matrix` and falls back
  to the resolution convention (≥720 → BT.709) for "unspecified"
  (`chooseYuvMatrix`).
- **No WebGPU**: the blender always reports CPU fallback there, consistent
  with the new off-by-default.

Cancellation is raced (`withAbort`) against every codec await
(`flush()`, `dequeue` waits, the copy and emit chains, `muxChain`) because
engines may never settle these after teardown; the copy-ahead gate races
the abort signal too, so a cancelled run cannot strand the copy chain on a
capacity slot that will never be freed. `video.html` additionally
force-resets the worker 1.5 s after a cancel that produces no reply, so the
page can never stay busy until reload. `bench/e2e-video.mjs` (Chromium) and
`bench/e2e-webkit.mjs` both cover cancel-then-convert-again.

## Clips

| file | content | why |
| --- | --- | --- |
| `bench/booth2s.mp4` | 2 s 360-booth clip, 480x270 h264, no audio | steady drift |
| `bench/booth2s-aac.mp4` | same + AAC 48k | audio stream-copy |
| `bench/boomerang.mp4` | booth2s + its reverse, 378 frames | one reversal (`auto`) |
| `bench/booth2s-vp9.webm` | booth2s re-encoded VP9 (`-b:v 1200k`) | Firefox-safe set |
| `bench/boomerang-vp9.webm` | boomerang re-encoded VP9 | Firefox-safe set |
| `bench/long-boomerang.mp4` | 5× concat, ~1700 frames | cancellation e2e |

Regenerate the derived clips with the ffmpeg lines in
[the e2e header](bench/e2e-video.mjs) if the base clip changes.
