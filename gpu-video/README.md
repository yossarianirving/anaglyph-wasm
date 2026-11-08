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
anaglyph type, HW decode/encode toggles) plus:

- **Canvas capture** — zero-copy encoder input (`new VideoFrame(canvas)`).
  Uncheck it to force the `copyTextureToBuffer → VideoFrame(BGRA bytes)`
  path. The page probes canvas capture once per run and falls back
  automatically if the engine rejects `new VideoFrame(canvas)`.
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
       │              blit → WebGPU OffscreenCanvas ──► VideoFrame(canvas)
       ▼                                   ▼
mux (mediabunny) ◄── VideoEncoder ◄────────┘        (+ audio stream-copy)
```

Files:

| file | contents |
| --- | --- |
| `gpu-pipeline.js` | the whole pipeline: `convertVideoToAnaglyphGpu`, GPU context, encoder planning, audio copy, mux, cancel — mirrors the engine-quirk handling of `js/video-pipeline.js` (which stays the reference) |
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
- **Frame-close safety.** In external mode a decoded frame is only closed
  after the serial emit (blend submit → capture → `encoder.encode`) that
  referenced it ran. In canvas mode the pipelined path may still have the
  pair's blend in flight when the pairing window releases the frame —
  safe because `drawImage` copies the frame's pixels into the slot's
  scratch canvas synchronously at issue time.
- **Pipelined capture (canvas mode).** Firefox resolves `mapAsync` on a
  ~10 Hz GPU-process tick (~100 ms; see `bench/probe-mapasync.mjs`), so
  a per-pair completion gate would cap throughput at ~10 pairs/s. Canvas
  mode issues up to 16 blends ahead — each in-flight pair owns a ring
  slot (frame stagers + capture canvas) — and drains the FIFO in pair
  order into the encoder (~16 pairs per tick). The external path keeps
  the serial emit (2 ms gates on Chrome, iOS-verified behaviour).
- **Auto mode holds frames, not copies.** Held-back (unconfirmed) frames
  stay as `VideoFrame` objects pinned by the decoder — bounded by the
  pairing window's pending cap (~20 frames) — instead of being rasterized
  into RGBA buffers the way the wasm pipeline must.

## Engine status

| engine | mode | notes |
| --- | --- | --- |
| Chrome 151 (Linux/Mesa) | external | full e2e suite green; needs a windowed browser for WebGPU on this host (`headless` → null adapter) |
| iOS Safari 18 (device) | external | verified working on device at tag `ios-safari-working-2026-11-08`; the pipelined-canvas work keeps this path byte-identical |
| Firefox 159 | canvas | `importExternalTexture` absent → staged `texture_2d` sources; full e2e suite green; ~10 → ~220 pairs/s via the pipelined capture FIFO |
| WebKitGTK (Playwright) | — | shim exposes `navigator.gpu` but `requestAdapter()` returns null; parked |

Benchmarks on this host (480×270 clips, median pairs/s, headed browsers,
same AMD RDNA-3 GPU; `node bench/run-web-bench.mjs --headed
--page gpu-video/bench.html --skip-native`):

| pipeline | clockwise | auto |
| --- | --- | --- |
| wasm CPU (Chrome) | 1427–1694 | 1084–1130 |
| all-GPU (Chrome, canvas capture) | 467–580 | 507–659 |
| all-GPU (Firefox, canvas capture) | 238–308 | 82–111 |
| wasm CPU (Firefox) | 317–339 | 309–335 |

Desktop throughput is not where the GPU pipeline wins — the wasm path's
CPU-memory encoder feed beats the per-frame GPU→encoder handoff. Its
edge is mobile (the original motivation: the `copyTo` frame copy that
dominated iOS profiling) and zero-CPU pixel movement; Firefox `auto` is
rate-limited by motion-sample readback latency on the GPU tick and
would benefit from deeper readback pipelining.

## Follow-up — closing the desktop gap

Ideas parked while the mobile path was the priority (numbers from the
A/B runs above, 480×270 on this host):

- **Pipeline Chrome's external-mode capture.** The depth-16 capture FIFO
  (`issuePairPipelined`) only runs in canvas source mode; the external
  path still waits on each pair's readback gate before issuing the next.
  Chrome's `mapAsync` is ~2 ms serial (Firefox's is ~100 ms), so the win
  is smaller but real — the gate wait is pure dead time between blends.
- **Feed the encoder CPU bytes, not canvas snapshots.** Encode-only
  ingest is 6429–9545 fps with `new VideoFrame(bytes)` vs ~1543 fps for
  `new VideoFrame(canvas)` on the same machine. The `--bytes` readback
  path already produces packed BGRA; using it as the primary encoder
  feed on desktop (keeping canvas capture where the copy is expensive,
  e.g. mobile) could lift the GPU path's 467–580 pairs/s substantially.
- **Stage costs to attack.** Per pair the wasm path pays ~0.063 ms copy
  + ~0.47 ms blend before an encoder feed that never stalls below
  ~6400 fps; the GPU path pays a ~1.58 ms GPU→encoder crossing per pair.
  That crossing, not GPU compute, is the gap — batching VideoFrames
  through it (ring of pre-created frames, encoder-side queueing) is the
  lever.
- **Resolution crossover.** The per-pair crossing cost is roughly fixed
  while the wasm copy+blend scales with pixels; at 1080p the wasm CPU
  path drops toward the GPU path's rate and the GPU path overtakes it
  (measured directionally; worth pinning with `run-web-bench.mjs` at
  1080p before claiming a number).
- **Firefox `auto` direction tracking.** 82–111 pairs/s is rate-limited
  by motion-sample readback latency on the GPU tick; batching those
  readbacks the way pair captures are batched should bring `auto` in
  line with `clockwise` (238–308 pairs/s).

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
