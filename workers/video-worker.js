/**
 * Worker around the wasm pipeline (WebCodecs decode/encode + wasm blending).
 * Message protocol: convert-video / cancel-video in; ready, video-progress,
 * video-result, video-cancelled, anaglyph-error out.
 *
 * wasm init is lazy (first convert-video) rather than at script load so a
 * caller-provided `wasmUrl` can steer where the .wasm is fetched from; the
 * `ready` handshake therefore means "script loaded", and callers should queue
 * jobs until it arrives either way (index.js does).
 */

import { convertVideoToAnaglyph } from "../js/video-pipeline.js";
import {
  createBlender,
  createTracker,
  createYuvConverter,
  ensureWasm,
} from "../js/wasm-factories.js";

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
    await ensureWasm(data.wasmUrl ?? null);
    const { blob, report } = await convertVideoToAnaglyph({
      file: data.file,
      direction: data.direction,
      anaglyphType: data.anaglyphType,
      acceleration: data.acceleration,
      encoderQuality: data.encoderQuality,
      createBlender,
      createYuvConverter,
      createTracker,
      onProgress: (status) => postMessage({ type: "video-progress", ...status }),
      signal: controller.signal,
    });
    // `url` serves the plain-page consumers; `blob` (structured-clone moves
    // the handle, not the bytes) serves the library API in index.js.
    postMessage({
      type: "video-result",
      url: URL.createObjectURL(blob),
      blob: blob,
      report: report,
    });
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
