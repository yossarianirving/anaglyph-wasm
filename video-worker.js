import init, { MotionAnalyzer, Tracker, VideoBlender } from "./pkg/anaglyph_wasm.js";
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
      // wasm-bindgen static async factory (async ctors are deprecated).
      createBlender: (width, height, anaglyphType) =>
        VideoBlender.new(width, height, anaglyphType),
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
