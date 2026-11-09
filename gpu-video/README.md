# gpu-video — the all-GPU video → anaglyph pipeline

A from-scratch implementation of the video converter that uses **only
JavaScript, WebGPU and mediabunny** — no wasm, and no CPU copy of a single
video pixel. It exists because profiling the wasm pipeline showed the
`VideoFrame → RGBA` copy (`copyTo`) dominating wall time on mobile: on
WebKitGTK it is a fixed ~1.4 ms round trip per call, ~2.8 ms per pair before
any blending happens (see [BENCH.md](../BENCH.md)).

The idea: keep every decoded `VideoFrame` on the GPU from decoder to
encoder. Two frame-source modes share everything else:

- **external** — frames imported as `texture_external` for the blend and
  motion passes (Chrome, Safari/iOS; the iOS-verified path).
- **canvas** — for engines without `importExternalTexture` (Firefox):
  each frame is staged through a 2D `drawImage` +
  `copyExternalImageToTexture` into `rgba8unorm` textures sampled as
  plain `texture_2d` by the same passes.

The encoder input is captured from a WebGPU canvas (or a
`copyTextureToBuffer` readback). The only things that ever touch CPU
memory are the ~8 KB per-frame motion result and a 256-byte completion
gate.

## Run it

Open `gpu-video/video.html` — same controls as `video.html` (direction,
anaglyph type, HW decode/encode toggles, encode quality tier) plus:

- **Canvas capture** — zero-copy encoder input (`new VideoFrame(canvas)`).
  Uncheck it to force the `copyTextureToBuffer → VideoFrame(BGRA bytes)`
  path. The page probes canvas capture once per run and falls back
  automatically if the engine rejects `new VideoFrame(canvas)`.
- **Encode quality** — Standard/High/Maximum bitrate tiers from the shared
  planner `js/encoder-plan.js` (used by both pipelines; measured costs and
  benefits in [BENCH.md](../BENCH.md) § Encode quality tiers).
- **Strict requirement** — there is no CPU fallback here on purpose. If
  `navigator.gpu` or an adapter is missing, the page says so and refuses to
  convert (use `video.html` on such browsers).

On an **iPhone/iPad** serve the repo over HTTPS (e.g. `npx http-server -S`
with a local cert, or any static host) and open the page in Safari 18+.
WebGPU shipped in Safari 18; older iOS has no `navigator.gpu`.

## How it works

```
demux (mediabunny) ──► VideoDecoder ──► VideoFrame (held open, GPU-resident)
       │                                   │
       │              ┌────────────────────┴───────────────────┐
       │              │ motion (auto only, per frame):         │
       │              │  luma fragment → r32float (full size)  │
       │              │  box downsample → 256-wide analysis    │
       │              │  SAD compute (±12 px, 8 px blocks)     │
       │              │  → ~8 KB readback → {dx, reliable}     │
       │              └────────────────────┬───────────────────┘
       │                                   ▼
       │                    pairing (js/pairing.js + tracker.js)
       │                                   │
       │              blend fragment (two frame sources +
       │              eye-matrix uniform) → BGRA target texture
       │                                   │
       │              capture (default): copyToBuffer → packed BGRA
       │              → VideoFrame(bytes); canvas mode blits to an
       │              OffscreenCanvas → VideoFrame(canvas) instead
       ▼                                   ▼
mux (mediabunny) ◄── VideoEncoder ◄────────┘        (+ audio stream-copy)
```

Files:

| file | contents |
| --- | --- |
| `gpu-pipeline.js` | the whole pipeline: `convertVideoToAnaglyphGpu`, GPU context, audio copy, mux, cancel — encoder planning comes from the shared `js/encoder-plan.js` — mirrors the engine-quirk handling of `js/video-pipeline.js` (which stays the reference) |
| `shaders.js` | all WGSL: blend, luma, blit, box downsample, SAD compute (`makeSadCompute(tileW)`) |
| `motion.js` | analysis geometry (ports `motion.rs`), CPU aggregation (`aggregateBlocks` reproduces `match_frames`' median + quality gate), `GpuMotion` pass driver |
| `tracker.js` | line-for-line JS port of the Rust `DirectionTracker` |
| `gpu-worker.js` | worker wrapper (protocol identical to `video-worker.js`) |
| `video.html` | the UI page |
| `bench.html` | benchmark page, same `window.__benchResults` contract as `bench-video.html` |
| `validate-shaders.mjs` | naga-based WGSL validation + workgroup-memory accounting |
| `tracker.test.mjs`, `motion.test.mjs` | `node --test` unit tests |

### Design notes

- **No parallax shift.** The wasm video pipeline blends with `offset = 0`
  — motion estimation only decides the eye order (`auto`), it never shifts
  pixels — so the blend shader needs no shift LUT either. The motion
  estimate feeds `pairing.js` through its synchronous `observe` contract.
- **Color math matches Rust.** Samples come back display-encoded
  (sRGB-ish) on all engines measured, so the shaders do the matrix math
  and the luma weights directly on those values — the same gamma-encoded
  0–255 domain as `combine_slices` / `motion.rs`, no transfer-function
  conversion. Output stores raw into `bgra8unorm`. Verified against the
  native CLI by PSNR, not theory.
- **Frame-source modes, one shader family.** `makeBlendShader(source)` /
  `makeLumaShader(source)` emit the `texture_external` or `texture_2d`
  variant around the same sampling/blend math (see `shaders.js`). The
  uv convention is uv.y = 0 at the TOP row everywhere — imported
  external textures were measured (probe-dims/probe-verify vs ffmpeg)
  to be the frame's visible rect stored top-left, so identity sampling
  preserves orientation through blend → capture → encode and through
  the bytes readback alike.
- **Motion on the GPU.** Three passes per frame in one command buffer,
  submitted fire-and-forget from the decode-output callback (the GPU leads
  the CPU encode chain by a frame). The kernel reproduces `motion.rs`
  exactly: 8 px blocks with `MAX_SHIFT` image margins, ±12 search,
  first-minimum tie-break, variance ≥ 6.0 flat reject, `|d| = 12`
  saturation reject, parabolic subpixel refinement; the CPU keeps only the
  upper-median + quality-ratio reduction. The workgroup tile size is
  derived from `maxComputeWorkgroupStorageSize` (29 at the 16 KB spec
  minimum, 61 at 32 KB).
- **Frame-close safety.** On the serial emit (iOS, canvas capture) a
  decoded frame closes after the emit (blend submit → capture →
  `encoder.encode`) that referenced it ran. On the pipelined emit the
  pairing window and the capture FIFO overlap, so entries are refcounted:
  each frame counts the sliding-window pairs referencing it (`livePairs`)
  and closes when the count hits zero *and* the pairing watermark has
  passed it (`released`) — the last owner to drain closes it. This is
  required for external sources, whose imported views keep the
  `VideoFrame` alive until the blend pass has run.
- **Pipelined capture.** Firefox resolves `mapAsync` on a ~10 Hz GPU-
  process tick (~100 ms; see `bench/probe-mapasync.mjs`), so a per-pair
  completion gate would cap throughput at ~10 pairs/s. The pipelined
  path issues up to `CAPTURE_DEPTH` (16) blends ahead and drains the
  FIFO in pair order into the encoder (~16 pairs per tick). Canvas mode
  owns a ring slot per in-flight pair (frame stagers + capture canvas);
  bytes mode imports the external textures inside the pass and drains
  each slot's readback buffer into the shared packed staging array.
  iOS keeps the serial emit (the device-verified behaviour).
- **Auto mode holds frames, not copies.** Held-back (unconfirmed) frames
  stay as `VideoFrame` objects pinned by the decoder — bounded by the
  pairing window's pending cap (~20 frames) — instead of being rasterized
  into RGBA buffers the way the wasm pipeline must.

## Engine status

| engine | mode | notes |
| --- | --- | --- |
| Chrome 151 (Linux/Mesa) | external | full e2e suite green; needs a windowed browser for WebGPU on this host (`headless` → null adapter) |
| iOS Safari 18 (device) | external | verified working on device at tag `ios-safari-working-2026-11-08`; still the serial emit path (device-verified) — canvas capture now defaults off, so the iOS default is serial **bytes** emit: re-verify on device |
| Firefox 159 | canvas | `importExternalTexture` absent → staged `texture_2d` sources; full e2e suite green; ~10 → ~310 pairs/s via the pipelined capture FIFO + batched motion readback |
| WebKitGTK (Playwright) | — | shim exposes `navigator.gpu` but `requestAdapter()` returns null; parked |

Benchmarks on this host (480×270 clips, median pairs/s, headed browsers,
same AMD RDNA-3 GPU; `node bench/run-web-bench.mjs --headed
--page gpu-video/bench.html --skip-native [--browser firefox]
--capture-bytes`):

| pipeline | clockwise | auto |
| --- | --- | --- |
| wasm CPU (Chrome) | 1427–1694 | 1084–1130 |
| all-GPU (Chrome, bytes capture) | 1069–1150 | 682–732 |
| all-GPU (Chrome, canvas capture) | 513–570 | 635–636 |
| wasm CPU (Firefox) | 317–339 | 309–335 |
| all-GPU (Firefox, bytes capture) | 286–312 | 249–290 |
| all-GPU (Firefox, canvas capture) | 245–294 | 231–249 |

Bytes capture (packed BGRA readback → `new VideoFrame(bytes)`, pipelined
16 deep) is the default on desktop: it roughly doubles Chrome's clockwise
rate and lifts Firefox `auto` from 82–111 to 249–290 (batched motion
readback — K frames share one `mapAsync`). Clockwise on Chrome now runs
at ~65% of wasm; the residual is the GPU cost of YUV→RGB conversion,
which every `texture_external` sample pays (~0.88 ms per sampling draw
at 480×270 — `bench/probe-motion.html` isolates it: plain-texture motion
passes hit 5.2k f/s vs 1.1k f/s with real frames). Converting once per
frame into a `texture_2d` ("staged" source) was measured and rejected:
`copyExternalImageToTexture` from `VideoFrame` costs 863 f/s vs 1134
for draw-time sampling — the shader converting only what it reads wins.
`auto` on Chrome (~700) sits at that sample-throughput ceiling: its
motion passes add a second conversion per frame, and the remaining
direction-tracking wait is ~K-invariant GPU time, not map latency
(`bench/probe-depth.mjs`: depth 4 ≈ 16 ≈ 64, batch K 2…32 all noise).

At 480×270 on this host wasm still leads; the GPU pipeline's edge is
mobile (the original motivation: the `copyTo` frame copy that dominated
iOS profiling) and zero-CPU pixel movement. The per-pair GPU cost is
resolution-sensitive while the wasm CPU blend scales faster, so larger
frames close the gap (see Follow-up).

## Follow-up — closing the desktop gap

From the earlier A/B runs; status as of the desktop-performance pass:

- **Pipeline Chrome's external-mode capture.** ✅ Done — the pipelined
  FIFO (`issuePairPipelined`) now runs for external sources with bytes
  capture too (non-iOS): each in-flight pair's decoded frames are closed
  by refcount (`livePairs`/`released`) when the pair's gate resolves,
  since the external view keeps the `VideoFrame` alive until the blend
  pass has run. A depth sweep at 480×270 found depth irrelevant once
  motion readbacks are batched (4 ≈ 16 ≈ 64), so one `CAPTURE_DEPTH = 16`
  serves both modes.
- **Feed the encoder CPU bytes, not canvas snapshots.** ✅ Done — bytes
  capture is the default everywhere (`canvasCapture = false` in the
  pipeline; the UI checkbox keeps canvas available). Chrome encode-only
  ingest is ~9-10k fps through `new VideoFrame(bytes)` vs ~1.5k through
  canvas, and clockwise went 467–580 → 1069–1150 pairs/s.
- **Firefox `auto` direction tracking.** ✅ Done — motion samples are
  read back in batches (`motion.js`: K=8 frames share one MAP_READ
  buffer + one `mapAsync` armed at batch close; `flush()` closes the
  tail). Firefox `auto` went 82–111 → 249–290 pairs/s; Chrome `auto`
  320 → 682–732. The serial emit path (iOS, canvas capture) keeps the
  per-frame await order.
- **Conversion ceiling (what is left).** The GPU path converts YUV→RGB
  on every external sample: the motion luma pass converts each frame
  once, and the blend pass converts its two frames again. All measured
  alternatives are equal or worse (staged `copyExternalImageToTexture`
  per frame: 863 vs 1134 f/s; see the numbers above). Beating wasm at
  small resolutions would mean sampling the YUV planes directly
  (`texture_2d` views of the decoder surface — not exposed to WebGPU)
  or sharing the conversion between motion and blend, neither of which
  the platform currently allows. Accept ~65% of wasm at 480×270 on
  desktop.
- **Resolution crossover.** The per-pair GPU cost scales with pixels
  but sublinearly vs the wasm CPU blend; at 1080p the wasm CPU path
  drops toward the GPU path's rate and the GPU path overtakes it
  (measured directionally; worth pinning with `run-web-bench.mjs` at
  1080p before claiming a number).
- **WebKitGTK.** `requestAdapter()` returns null under the Playwright
  shim; parked.

## Validation

- `node gpu-video/validate-shaders.mjs` — every WGSL source type-checked
  with naga v30 (both frame-source variants; external textures
  substituted) + workgroup limits.
- `node --test gpu-video/ js/` — tracker port vs the Rust test suite,
  pairing integration through the real `AutoFramePairer`, motion
  geometry + aggregation gates vs `motion.rs`, and `blend-uniform.test.mjs`
  (eye-matrix uniform bytes → WGSL `mat3x3` product vs `combine_slices`
  for all five anaglyph types — guards the transposition that made
  non-`color` types render heavily green).
- `node bench/e2e-gpu.mjs [--browser=chrome|firefox] [--bytes]` — 16-case
  browser suite (silent/HW, aac + opus stream-copy, one case per anaglyph
  type checked against the native anaglyph-rs CLI, boomerang auto with
  reversal-time check, odd dims, rotation, software decode, one-frame
  and corrupt-upload errors, cancel-then-recover). Outputs are PSNR'd
  against the native CLI ground truth: silent 41.6–42.3 dB, auto 41.0–
  41.3 dB, aac 41.2–41.8 dB — encoder-noise level, verified on Chrome
  and Firefox. Blend types land at 40.8–44.6 dB on Chrome; on Firefox
  the `true` type floors at 18 dB because the engine's own RGB→YUV
  encoder path lifts the green channel ~34/255 on boundary-chroma
  pixels (green=0) — `bench/probe-fx-greenspill.mjs` measures the same
  lift through the wasm pipeline, so the GPU pipeline matches the
  reference within the engine's fidelity.

Known unknowns on device (all spec-supported, but engine-dependent):
`new VideoFrame(OffscreenCanvas)` capture semantics on Safari (falls back
automatically), `importExternalTexture` dict-vs-positional form (probed at
runtime), and video-frames-into-canvas capture for hardware decoders.
