/**
 * wasm-bindgen wiring for the wasm video pipeline.
 *
 * `convertVideoToAnaglyph` (js/video-pipeline.js) never imports wasm itself —
 * the blender, tracker and YUV-converter factories are injected so the
 * pipeline stays pure JS. This module supplies those factories from the
 * wasm-bindgen glue, so the worker (workers/video-worker.js) and the package
 * entry (index.js) share one definition.
 */

import init, {
  MotionAnalyzer,
  Tracker,
  VideoBlender,
  YuvFrameConverter,
} from "../pkg/anaglyph_wasm.js";

/** @type {Promise<unknown> | null} */
let initPromise = null;

/**
 * Initialize the wasm module once. `wasmUrl` (optional) overrides where
 * `anaglyph_wasm_bg.wasm` is fetched from — bundlers that cannot resolve the
 * default `import.meta.url` fetch inside a worker need this escape hatch.
 * Later calls ignore the URL of subsequent callers (first wins).
 * @param {string | URL | null} [wasmUrl]
 * @returns {Promise<unknown>}
 */
export function ensureWasm(wasmUrl) {
  if (!initPromise) {
    initPromise = init(wasmUrl ? { module_or_path: wasmUrl } : undefined);
  }
  return initPromise;
}

/**
 * @param {number} width
 * @param {number} height
 * @param {string} anaglyphType
 * @param {"gpu" | null} [backend] `null`/undefined selects CPU; the pipeline
 *   passes "gpu" only when the WebGPU blend option is on.
 */
export function createBlender(width, height, anaglyphType, backend) {
  // wasm-bindgen static async factory (async ctors are deprecated).
  return VideoBlender.new(width, height, anaglyphType, backend ?? null);
}

/**
 * WebKit's copyTo hands back native NV12/I420 planes; this wasm converter
 * turns them (plus the visible-crop geometry) into dense RGBA in place of a
 * canvas raster. Both buffers live in wasm memory so the ~1.4 ms/frame copyTo
 * carries the pixels straight in, and `convert` speaks a plain geometry
 * object on this side of the glue.
 * @param {number} planeBytes
 * @param {number} width
 * @param {number} height
 */
export function createYuvConverter(planeBytes, width, height) {
  const c = new YuvFrameConverter(planeBytes, width, height);
  return {
    planesView: () => c.planesView(),
    convert: (g) =>
      c.convert(
        g.yOffset,
        g.yStride,
        g.uvOffset,
        g.uvStride,
        g.vOffset,
        g.x,
        g.y,
        g.width,
        g.height,
        g.planar,
        g.bt709,
      ),
    dstView: () => c.dstView(),
  };
}

/**
 * wasm-bindgen keeps Rust snake_case on the JS surface; pairing.js speaks
 * camelCase, so adapt here (the glue layer owns wasm shapes).
 * @param {number} width
 * @param {number} height
 * @param {boolean} reverse
 */
export async function createTracker(width, height, reverse) {
  const analyzer = MotionAnalyzer.new(width, height);
  const tracker = new Tracker(reverse);
  return {
    tracker: {
      push: (index, timeSecs, dx, reliable) =>
        tracker.push(index, timeSecs, dx, reliable),
      confirmedThrough: () => tracker.confirmed_through(),
      currentDirection: () => tracker.current_direction(),
      finish: () => tracker.finish(),
      takeReversals: () => tracker.take_reversals(),
    },
    observe: (frame) => ({
      dx: analyzer.observe_rgba(new Uint8Array(frame.rgba)),
      reliable: analyzer.is_reliable(),
    }),
  };
}
