# Video pipeline benchmarks

Benchmarks for the browser video → anaglyph pipeline (`js/video-pipeline.js`,
`video.html`) against the native `anaglyph-rs` CLI, plus per-stage bounds.

## Running

```sh
npm run build:wasm                 # current wasm for the browser side
node bench/run-web-bench.mjs       # browser matrix + native CLI
node bench/run-web-bench.mjs --iters 5 --gpu
```

- Results land in `bench/results/web-<timestamp>.json` (raw, per-iteration)
  and `bench/results/latest.md` (summary table). `bench/results/` is
  git-ignored: numbers are machine-specific by nature.
- Diff two runs (before/after a change):
  `node bench/compare.mjs bench/results/web-A.json bench/results/web-B.json`
- Everything browser-side can also be driven interactively via
  `bench-video.html` (clip/direction/backend/iterations pickers, results
  table + JSON) after `npm run serve`.

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
- **blend-only fps** — wasm `VideoBlender.blend` on synthetic frames with no
  codecs: the blend stage's standalone throughput.
- **stage columns** (`copy/blend/enc-wait/dec-wait ms`) — summed time inside
  each step per run. Stages overlap asynchronously, so these are attribution,
  not a wall-clock split.

## Baseline (2026-11-07)

Software-only environment: headless Chromium (no WebGPU adapter, software
h264 codec), native CLI on VAAPI encode, CPU blend both sides; 480x270 clips;
median of 3. This is a *stage-shape* baseline, not hardware-vs-hardware.

| clip | direction | blend backend | median pairs/s | decode-only fps | blend-only fps |
| --- | --- | --- | --- | --- | --- |
| booth2s (5.7 s) | clockwise | CPU | 1153 | 5672 | 1088 |
| booth2s (5.7 s) | auto | CPU | 793 | 5588 | 1109 |
| booth2s-aac (5.7 s) | clockwise | CPU | 1167 | 5621 | 1103 |
| boomerang (11.3 s) | clockwise | CPU | 1223 | 7186 | 1097 |
| boomerang (11.3 s) | auto | CPU | 805 | 7485 | 1114 |

Native CLI (`-b cpu`, VAAPI h264 encode, wall-clock incl. process start):

| clip | direction | frames/s |
| --- | --- | --- |
| booth2s | clockwise | 451 |
| booth2s | auto | 886 |
| booth2s-aac | clockwise | 903 |
| booth2s-aac | auto | 666 |
| boomerang | clockwise | 1260 |
| boomerang | auto | 1184 |

Reading the baseline:

- **Blend is the bottleneck** on the CPU path: pipeline throughput
  (~1150–1220 pairs/s) sits at the blend-only rate (~1100 fps) while decode
  alone does 5–7× that. WebGPU blending (real adapter) should push the
  pipeline toward decode/encode speed; the `--gpu` matrix runs both backends
  per row for exactly this comparison.
- **Auto direction costs ~30 %** on the CPU path (≈0.35 ms/pair for the
  motion estimate + tracker); negligible on GPU-bound runs, worth watching
  otherwise.
- Native runs on 5-second clips are dominated by process/encoder startup;
  the 11-second boomerang shows the steady state (~1200 frames/s).

## Clips

| file | content | why |
| --- | --- | --- |
| `bench/booth2s.mp4` | 2 s 360-booth clip, 480x270 h264, no audio | steady drift |
| `bench/booth2s-aac.mp4` | same + AAC 48k | audio stream-copy |
| `bench/boomerang.mp4` | booth2s + its reverse, 378 frames | one reversal (`auto`) |
| `bench/long-boomerang.mp4` | 5× concat, ~1700 frames | cancellation e2e |

Regenerate the derived clips with the ffmpeg lines in
[the e2e header](bench/e2e-video.mjs) if the base clip changes.
