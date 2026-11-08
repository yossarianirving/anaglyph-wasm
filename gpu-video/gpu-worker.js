/**
 * Worker around the WebGPU-only pipeline: no wasm, no CPU frame copies.
 * Protocol mirrors video-worker.js (convert-video / cancel-video,
 * video-progress / video-result / video-cancelled / anaglyph-error).
 */

import {
  convertVideoToAnaglyphGpu,
  createGpuContext,
} from "./gpu-pipeline.js";

/** @type {AbortController | null} */
let controller = null;

self.postMessage({ type: "ready" });

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
    // The adapter is requested in the worker (WebGPU works there) and
    // handed to the pipeline, which owns its lifetime for the run.
    const gpu = await createGpuContext();
    const { blob, report } = await convertVideoToAnaglyphGpu({
      file: data.file,
      direction: data.direction,
      anaglyphType: data.anaglyphType,
      acceleration: data.acceleration,
      canvasCapture: data.canvasCapture !== false,
      gpu,
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
