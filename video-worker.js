import init, { MotionAnalyzer, Tracker, VideoBlender, YuvFrameConverter } from "./pkg/anaglyph_wasm.js";
import { convertVideoToAnaglyph } from "./js/video-pipeline.js";

/** @type {AbortController | null} */
let controller = null;

async function run() {
  await init();
  console.log("Video worker initialized");
  self.postMessage({ type: "ready" });
}
run();

self.onmessage = function (e) {
  switch (e.data.type) {
    case "convert-video":
      startConversion(e.data);
      break;
    case "cancel-video":
      controller?.abort();
      break;
    default:
      break;
  }
};

async function startConversion(data) {
  if (controller) {
    postMessage({
      type: "anaglyph-error",
      message: "a conversion is already running",
    });
    return;
  }
  controller = new AbortController();
  try {
    const { blob, report } = await convertVideoToAnaglyph({
      file: data.file,
      direction: data.direction,
      anaglyphType: data.anaglyphType,
      acceleration: data.acceleration,
      // wasm-bindgen static async factory (async ctors are deprecated).
      // `backend` is "cpu" when the WebGPU blend toggle is off.
      createBlender: (width, height, anaglyphType, backend) =>
        VideoBlender.new(width, height, anaglyphType, backend ?? null),
      // WebKit's copyTo hands back native NV12/I420 planes; this wasm
      // converter turns them (plus the visible-crop geometry) into dense
      // RGBA in place of a canvas raster. Both buffers live in wasm memory
      // so the ~1.4 ms/frame copyTo carries the pixels straight in, and
      // `convert` speaks a plain geometry object on this side of the glue.
      createYuvConverter: (planeBytes, width, height) => {
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
      },
      createTracker: async (width, height, reverse) => {
        // wasm-bindgen keeps Rust snake_case on the JS surface; pairing.js
        // speaks camelCase, so adapt here (the glue layer owns wasm shapes).
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
      },
      onProgress: (status) => postMessage({ type: "video-progress", ...status }),
      signal: controller.signal,
    });
    const url = URL.createObjectURL(blob);
    postMessage({ type: "video-result", url: url, report: report });
  } catch (err) {
    const message = String(err && err.message ? err.message : err);
    postMessage({
      type: err && err.name === "AbortError" ? "video-cancelled" : "anaglyph-error",
      message: message,
    });
  } finally {
    controller = null;
  }
}
