# Changelog

## Unreleased

- `video-unified.html`: engine default is now the wasm pipeline everywhere
  except iPhone/iPad, which boot on the all-GPU WebGPU pipeline (all iOS
  browsers are WebKit, so the device alone decides; a missing adapter still
  self-heals through the wasm fallback retry). Previously the adapter picked
  the default, so Chrome/Firefox with WebGPU started on the GPU pipeline;
  `?engine=webgpu|wasm` now overrides in both directions.
- `video-unified.html`: the result button says "Save" and opens the system
  share sheet via the Web Share API where file sharing is supported
  (`navigator.canShare`); other browsers keep the plain download.

## 0.1.0 — 2026-11-09

First packaged release.

- Public entry `index.js`: `convertVideo()` (main thread),
  `createVideoConverter()` (worker wrapper with progress, cancel +
  force-reset watchdog and a one-shot WebGPU→wasm fallback) and
  `checkSupport()`; TypeScript types in `index.d.ts`.
- Two interchangeable pipelines behind one API: the all-GPU pipeline
  (`engine: "webgpu"`) and the WebCodecs + wasm pipeline (`engine:
  "wasm"`), both with auto orbit detection (incl. eye-swap on reversals),
  five anaglyph color types, AAC stream-copy, MP4 primary / WebM
  fallback, encode-quality tiers (standard / high / maximum).
- Workers consolidated under `workers/`; wasm-bindgen factory wiring
  shared in `js/wasm-factories.js`.
- Reproducible builds: `anaglyph-rs` pinned to git tag `0.1.0`,
  `wasm-bindgen` pinned to `0.2.130` with a CLI-version guard in
  `scripts/build-wasm.sh`; the built `pkg/` is committed per release so
  git installs need no Rust toolchain.
- Distribution: GitHub only (git-tag dependency, plain clone, jsDelivr);
  mediabunny ships as a vendored runtime bundle (`vendor/`).
