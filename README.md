# anaglyph-wasm

Browser video → anaglyph 3-D conversion with two interchangeable pipelines:

- **`webgpu`** — the all-GPU pipeline: JavaScript + WebGPU +
  [mediabunny](https://mediabunny.dev) only. Decoded frames stay on the GPU
  from the decoder to the encoder (blending, motion analysis and the encoder
  capture are all GPU passes).
- **`wasm`** — WebCodecs hardware decode/encode around a Rust/wasm blender
  (SIMD; optional experimental WebGPU blending).

Both are driven by the same promise API, run inside a dedicated worker by
default, and share the feature set: fixed or auto-detected orbit direction
(with eye-swap on booth reversals), five anaglyph color types, AAC audio
stream-copied from the source, MP4/H.264 primary with WebM fallback,
rotation/odd-dimension handling, progress callbacks and cancel.

## Install

GitHub-only distribution (no npm publish). Install a tagged release:

```sh
npm install github:yossarianirving/anaglyph-wasm#v0.1.0
```

The built wasm (`pkg/`) is committed at each release tag, so no Rust
toolchain is needed to consume the package — plain clones, git dependencies
and jsDelivr (`https://cdn.jsdelivr.net/gh/yossarianirving/anaglyph-wasm@v0.1.0/…`)
all work as-is.

## Usage

### Worker-based (recommended)

```js
import { createVideoConverter } from "anaglyph-wasm";

const converter = createVideoConverter({
  engine: "auto", // "auto" | "webgpu" | "wasm"
});

try {
  const { blob, url, report, engine } = await converter.convert({
    file, // File | Blob — stereo "360 booth" clip; MP4/H.264 works best
    direction: "auto-reverse", // clockwise | counter-clockwise | auto | auto-reverse
    anaglyphType: "color", // color | half-color | grayscale | optimized | true
    encoderQuality: "standard", // standard | high | maximum
    onProgress: (s) => console.log(s.frames, s.fps),
  });
  videoEl.src = url; // object URL, ready to play
  aEl.href = url; // …and to download
} catch (err) {
  if (err.name === "AbortError") { /* cancelled */ }
  else console.error("failed:", err.message);
}

converter.cancel(); // cooperative, with a 1.5 s force-reset watchdog
converter.dispose();
```

### Main thread

```js
import { convertVideo } from "anaglyph-wasm";

const { blob, report, engine } = await convertVideo({
  file,
  direction: "auto",
  signal: controller.signal,
});
```

### Engine selection

`engine: "auto"` resolves to `"webgpu"` when `navigator.gpu` answers with an
adapter, else `"wasm"`. A `webgpu` run that fails for adapter reasons (the
GPU got reclaimed after page load, say) is retried **once** on the wasm
pipeline; the result reports `engine: "wasm", fellBack: true`, and the
`onEngine(engine, note)` callback fires with `note.fallback` carrying the
original failure message.

`checkSupport()` (cached) exposes the probe behind this decision:
`{ webcodecs, h264Encode, webgpu, engine }`.

### Subpath imports

```js
import { convertVideoToAnaglyph } from "anaglyph-wasm/wasm";   // pipeline only
import { convertVideoToAnaglyphGpu } from "anaglyph-wasm/gpu"; // pipeline only
import { planEncoder } from "anaglyph-wasm/encoder-plan";      // bitrate tiers
import "anaglyph-wasm/workers/video";                          // worker module
import "anaglyph-wasm/workers/gpu";                            // worker module
```

The workers speak a plain message protocol (`convert-video` /
`cancel-video` in; `ready`, `video-progress`, `video-result`,
`video-cancelled`, `anaglyph-error` out), so they can also be wired by hand:

```js
const worker = new Worker(
  new URL("anaglyph-wasm/workers/video", import.meta.url),
  { type: "module" },
);
```

### Bundler notes

- **mediabunny is bundled, not a dependency.** The pipelines import it from
  `vendor/mediabunny.mjs` by relative path because browsers do not apply
  page import maps inside module workers — a bare specifier would only
  resolve on the main thread. `npm run vendor` refreshes the copy from the
  devDependency.
- **wasm URL.** By default `anaglyph_wasm_bg.wasm` is fetched relative to the
  wasm-bindgen glue via `import.meta.url`. If your bundler cannot resolve
  that inside a worker, pass an explicit URL:
  `convert({ …, wasmUrl: myWasmUrl })`.

## Browser support

| feature | Chrome/Edge | Firefox | Safari (macOS/iOS) |
| --- | --- | --- | --- |
| `wasm` pipeline | 94+ | 130+ | 26+ (WebKit build) |
| `webgpu` pipeline | 113+ | 141+ (`dom.webgpu.enabled`) | 18+ |
| H.264 encode (MP4 out) | ✓ | ✓ (platform dependent) | ✓ |

With no WebGPU adapter the unified entry simply starts on the wasm
pipeline; with no H.264 encoder the output falls back to WebM/VP9 (reported
in `report.container`).

## Development

```sh
npm ci
npm run build:wasm   # needs rust (wasm32-unknown-unknown) + matching wasm-bindgen CLI
npm test             # unit suites (node --test js/ gpu-video/)
npm run serve        # http://localhost:8000/video-unified.html
```

- The Rust core builds on [anaglyph-rs](https://github.com/yossarianirving/anaglyph-rs)
  pinned by git tag; for local co-development use the sibling checkout with a
  cargo `paths = ["../anaglyph-rs"]` override (uncommitted).
- End-to-end suites (Playwright; drivers must live inside the repo):
  `node bench/e2e-video.mjs`, `node bench/e2e-gpu.mjs --browser=chrome|firefox`.
- Benchmarks: see [BENCH.md](BENCH.md).

### Repository layout

| path | contents |
| --- | --- |
| `index.js` / `index.d.ts` | public entry: `convertVideo`, `createVideoConverter`, `checkSupport` |
| `workers/` | worker wrappers for both pipelines |
| `js/` | wasm-pipeline modules (planner, pairing, pipeline, wasm factories) |
| `gpu-video/` | all-GPU pipeline modules |
| `pkg/` | generated wasm-bindgen output (committed per release) |
| `vendor/` | vendored mediabunny bundle (runtime asset) |
| `video-unified.html`, `video.html`, `gpu-video/video.html` | demo pages |
| `bench/` | benchmark + e2e drivers and fixtures (not published) |

## License

No license has been attached to this repository yet.
