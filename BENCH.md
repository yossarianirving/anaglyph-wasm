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
- **WebGPU blend can lose to CPU in the full pipeline**: the CPU↔GPU copies
  (YUV→RGBA decode copies, upload, `mapAsync` readback, RGBA→YUV encode
  input) are attributed to `copy`/`blend`, and on SwiftShader (`--gpu` in
  this container) they measure slower than the SIMD CPU loop. Only compare
  `backend=cpu` vs `backend=auto` runs on real GPUs (`--gpu` on desktop
  Chromium, or `video.html` on hardware); toggle WebGPU blending off in the
  UI if the report shows WebGPU blend with lower fps than a CPU-blend run.

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
