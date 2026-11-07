import init, { VideoBlender } from "./pkg/anaglyph_wasm.js";
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
