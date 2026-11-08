# gpu-video — the all-GPU video → anaglyph pipeline

A from-scratch implementation of the video converter that uses **only
JavaScript, WebGPU and mediabunny** — no wasm, and no CPU copy of a single
video pixel. It exists because profiling the wasm pipeline showed the
`VideoFrame → RGBA` copy (`copyTo`) dominating wall time on mobile: on
WebKitGTK it is a fixed ~1.4 ms round trip per call, ~2.8 ms per pair before
any blending happens (see [BENCH.md](../BENCH.md)).

The idea: keep every decoded `VideoFrame` on the GPU from decoder to
encoder. Frames are imported as external textures for the blend pass and
for the motion passes; the encoder input is captured from a WebGPU canvas.
The only things that ever touch CPU memory are the ~8 KB per-frame motion
result and a 256-byte completion gate.

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
       │              blend fragment (two external textures +
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
- **Color math matches Rust.** `texture_external` samples sRGB-decode to
  linear; the shaders re-encode with the sRGB ETRF so the matrix math and
  the luma weights operate on the same gamma-encoded 0–255 domain as
  `combine_slices` / `motion.rs`. Output stores raw into `bgra8unorm`.
- **Motion on the GPU.** Three passes per frame in one command buffer,
  submitted fire-and-forget from the decode-output callback (the GPU leads
  the CPU encode chain by a frame). The kernel reproduces `motion.rs`
  exactly: 8 px blocks with `MAX_SHIFT` image margins, ±12 search,
  first-minimum tie-break, variance ≥ 6.0 flat reject, `|d| = 12`
  saturation reject, parabolic subpixel refinement; the CPU keeps only the
  upper-median + quality-ratio reduction. The workgroup tile size is
  derived from `maxComputeWorkgroupStorageSize` (29 at the 16 KB spec
  minimum, 61 at 32 KB).
- **Encode drain gate.** Pairs are released back to the pool only after the
  emit chain (blend submit → capture → `encoder.encode`) that referenced
  them has run, so a `frame.close()` can never yank pixels from a blend
  that is still queued.
- **Auto mode holds frames, not copies.** Held-back (unconfirmed) frames
  stay as `VideoFrame` objects pinned by the decoder — bounded by the
  pairing window's pending cap (~20 frames) — instead of being rasterized
  into RGBA buffers the way the wasm pipeline must.

## Validation status

No browser in the development container exposes WebGPU (probed: Chromium
1254 + system Chrome 151 with every flag, Firefox, WebKitGTK 2.54.2), so
functional GPU testing happens on the target device (iPhone Safari 18+):

- `node gpu-video/validate-shaders.mjs` — every WGSL source type-checked
  with naga v30 (external textures substituted) + workgroup limits.
- `node --test gpu-video/` — tracker port vs the Rust test suite, pairing
  integration through the real `AutoFramePairer`, motion geometry +
  aggregation gates vs `motion.rs`.
- The `video.html` "Canvas capture" toggle exists so a device test can
  A/B the two encoder-input paths if one misbehaves.

Known unknowns on device (all spec-supported, but engine-dependent):
`new VideoFrame(OffscreenCanvas)` capture semantics on Safari (falls back
automatically), `importExternalTexture` dict-vs-positional form (probed at
runtime), and video-frames-into-canvas capture for hardware decoders.
