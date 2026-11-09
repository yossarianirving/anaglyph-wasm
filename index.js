/**
 * anaglyph-wasm public entry: browser video → anaglyph 3-D video.
 *
 * Two interchangeable pipelines, picked per conversion:
 *  - "webgpu": all-GPU pipeline (JS + WebGPU + mediabunny, no wasm);
 *  - "wasm":  WebCodecs decode/encode around the wasm blender.
 *
 * Use from the main thread with convertVideo(), or off the UI thread with
 * createVideoConverter() (a worker wrapper; same promise semantics). Both
 * resolve engine "auto" to "webgpu" when an adapter is available and fall
 * back to "wasm" once when a webgpu run fails for adapter reasons.
 *
 * The pipelines import mediabunny from ../vendor/ (a bundled copy kept in
 * sync with the devDependency by `npm run vendor`) rather than the bare
 * specifier: browsers do not apply page import maps inside module workers,
 * and every real conversion runs in a worker — see vendor/README.md.
 */

/** Cached capability probe (checkSupport). @type {Promise<object> | null} */
let supportPromise = null;

/**
 * Probe the browser once (cached). Resolves to
 * `{ webcodecs, h264Encode, webgpu, engine }` — `engine` is what "auto"
 * would pick here ("webgpu" if an adapter answers, else "wasm").
 * @param {boolean} [force] re-probe instead of using the cached result
 */
export function checkSupport(force = false) {
  if (force || !supportPromise) {
    supportPromise = (async () => {
      const webcodecs =
        typeof VideoDecoder !== "undefined" &&
        typeof VideoEncoder !== "undefined";
      let h264Encode = false;
      if (webcodecs) {
        h264Encode = (
          await VideoEncoder.isConfigSupported({
            codec: "avc1.640028",
            width: 1920,
            height: 1080,
          }).catch(() => ({ supported: false }))
        ).supported;
      }
      let webgpu = false;
      if (typeof navigator !== "undefined" && navigator.gpu) {
        webgpu = !!(await navigator.gpu
          .requestAdapter()
          .catch(() => null));
      }
      return { webcodecs, h264Encode, webgpu, engine: webgpu ? "webgpu" : "wasm" };
    })().catch((err) => {
      supportPromise = null; // a failed probe must not poison the cache
      throw err;
    });
  }
  return supportPromise;
}

/**
 * Run one conversion on the calling thread.
 *
 * @param {object} options
 * @param {File | Blob} options.file stereo source video
 * @param {"clockwise" | "counter-clockwise" | "auto" | "auto-reverse"} [options.direction]
 * @param {"color" | "half-color" | "grayscale" | "optimized" | "true"} [options.anaglyphType]
 * @param {"auto" | "webgpu" | "wasm"} [options.engine]
 * @param {{ decode?: boolean, encode?: boolean, webgpu?: boolean }} [options.acceleration]
 *   hw decode/encode (default on); `webgpu` opts into GPU blending on the
 *   wasm pipeline (usually slower, off by default).
 * @param {boolean} [options.canvasCapture] zero-copy encoder capture for the
 *   webgpu pipeline (usually slower on desktop). Ignored by the wasm pipeline.
 * @param {"standard" | "high" | "maximum"} [options.encoderQuality]
 * @param {string | URL | null} [options.wasmUrl] override for fetching
 *   `anaglyph_wasm_bg.wasm` (bundler escape hatch).
 * @param {(engine: string, note?: { fallback?: string }) => void} [options.onEngine]
 *   called with the engine that runs; `note.fallback` carries the webgpu
 *   failure message when the wasm retry was triggered.
 * @param {AbortSignal} [options.signal]
 * @returns {Promise<{ blob: Blob, report: object, engine: string, fellBack?: boolean }>}
 *   rejects with an `AbortError`-named error on cancel.
 */
export async function convertVideo(options) {
  const {
    engine = "auto",
    wasmUrl = null,
    onEngine = () => {},
    signal,
    ...job
  } = options;
  if (!["auto", "webgpu", "wasm"].includes(engine)) {
    throw new Error(`unknown engine "${engine}" (use "auto", "webgpu" or "wasm")`);
  }
  const chosen = engine === "auto" ? (await checkSupport()).engine : engine;
  onEngine(chosen);
  try {
    const result = await runOn(chosen, job, wasmUrl, signal);
    return { ...result, engine: chosen };
  } catch (err) {
    if (isGpuAvailabilityFailure(err) && chosen === "webgpu") {
      onEngine("wasm", { fallback: errMessage(err) });
      const result = await runOn("wasm", job, wasmUrl, signal);
      return { ...result, engine: "wasm", fellBack: true };
    }
    throw err;
  }
}

/**
 * Worker-based converter: runs each conversion inside a dedicated worker
 * with the same promise semantics as convertVideo, plus cancel() with a
 * force-reset watchdog (some engines can strand a codec await past the
 * abort; terminating the worker is the one recovery they cannot ignore).
 *
 * @param {object} [options]
 * @param {"auto" | "webgpu" | "wasm"} [options.engine] default per-run engine
 * @param {{ wasm?: string, webgpu?: string }} [options.workerUrls] override
 *   the worker script URLs (defaults resolve next to this module).
 * @param {(engine: string, note?: { fallback?: string }) => void} [options.onEngine]
 *   called when a job is dispatched and on fallback switches.
 */
export function createVideoConverter(options = {}) {
  const { engine = "auto", workerUrls = null, onEngine = () => {} } = options;
  const urls = {
    wasm:
      (workerUrls && workerUrls.wasm) ||
      new URL("./workers/video-worker.js", import.meta.url).href,
    webgpu:
      (workerUrls && workerUrls.webgpu) ||
      new URL("./workers/gpu-worker.js", import.meta.url).href,
  };

  /** @type {{ job: object, resolve: Function, reject: Function, sent: boolean, fellBack: boolean } | null} */
  let pending = null;
  let worker = null;
  let workerEngine = null;
  let workerReady = false;
  let watchdog = null;

  function clearWatchdog() {
    if (watchdog) {
      clearTimeout(watchdog);
      watchdog = null;
    }
  }

  function abortError(reset) {
    const err = new Error("conversion cancelled");
    err.name = "AbortError";
    if (reset) err.workerReset = true;
    return err;
  }

  function settle() {
    clearWatchdog();
    return pending;
  }

  /** Post the pending job once the worker is past its ready handshake. */
  function flush() {
    if (workerReady && pending && !pending.sent) {
      pending.sent = true;
      worker.postMessage({ type: "convert-video", ...pending.job });
      onEngine(workerEngine);
    }
  }

  function spawn(which) {
    if (worker) worker.terminate();
    workerReady = false;
    workerEngine = which;
    worker = new Worker(urls[which], { type: "module" });
    worker.onmessage = function (e) {
      switch (e.data.type) {
        case "ready":
          workerReady = true;
          flush();
          break;
        case "video-progress":
          clearWatchdog(); // cooperative progress during a cancel attempt
          if (pending) onProgressOf(pending, e.data);
          break;
        case "video-result": {
          const run = settle();
          pending = null;
          if (run) {
            run.resolve({
              url: e.data.url,
              blob: e.data.blob,
              report: e.data.report,
              engine: workerEngine,
              fellBack: run.fellBack || undefined,
            });
          }
          break;
        }
        case "video-cancelled": {
          const run = settle();
          pending = null;
          if (run) run.reject(abortError(false));
          break;
        }
        case "anaglyph-error":
          failRun(e.data.message);
          break;
        default:
          break;
      }
    };
    // A worker that dies outside the protocol (uncaught exception) must not
    // leave the caller's promise hanging forever.
    worker.onerror = function (e) {
      failRun("worker error" + (e.message ? " — " + e.message : ""));
    };
  }

  function onProgressOf(run, data) {
    if (run.onProgress) run.onProgress(data);
  }

  function failRun(message) {
    if (!pending) return;
    // A run that failed for lack of WebGPU (adapter vanished after the page
    // probe, e.g. reclaimed by the OS) retries once on the wasm pipeline.
    if (
      workerEngine === "webgpu" &&
      !pending.fellBack &&
      /webgpu|gpu adapter/i.test(message)
    ) {
      pending.fellBack = true;
      spawn("wasm");
      pending.sent = false; // the job never reached the new worker
      flush();
      onEngine("wasm", { fallback: message });
      return;
    }
    const run = settle();
    pending = null;
    run.reject(new Error(message));
  }

  return {
    /** Which engine the current worker runs ("auto" resolved on first use). */
    get engine() {
      return workerEngine;
    },
    /**
     * Convert one file. `overrides.engine` picks (or switches) the pipeline
     * for this run; the converter keeps that worker alive for the next run.
     * Rejects with an `AbortError`-named error on cancel (`.workerReset`
     * when the worker had to be force-reset after cancel()).
     */
    convert: async function (job, overrides = {}) {
      if (pending) throw new Error("a conversion is already running");
      const want = overrides.engine ?? engine;
      if (!["auto", "webgpu", "wasm"].includes(want)) {
        throw new Error(`unknown engine "${want}" (use "auto", "webgpu" or "wasm")`);
      }
      const chosen = want === "auto" ? (await checkSupport()).engine : want;
      if (workerEngine !== chosen || !worker) spawn(chosen);
      return await new Promise((resolve, reject) => {
        pending = {
          job: {
            file: job.file,
            direction: job.direction,
            anaglyphType: job.anaglyphType,
            acceleration: job.acceleration,
            canvasCapture: job.canvasCapture,
            encoderQuality: job.encoderQuality,
            wasmUrl: job.wasmUrl,
          },
          onProgress: job.onProgress,
          resolve,
          reject,
          sent: false,
          fellBack: false,
        };
        flush();
      });
    },
    /** Cooperative cancel; a 1.5 s watchdog force-resets a stuck worker. */
    cancel: function () {
      if (!pending) return;
      if (!pending.sent) {
        const run = pending;
        pending = null;
        run.reject(abortError(false));
        return;
      }
      worker.postMessage({ type: "cancel-video" });
      clearWatchdog();
      watchdog = setTimeout(function () {
        watchdog = null;
        const run = pending;
        pending = null;
        spawn(workerEngine);
        if (run) run.reject(abortError(true));
      }, 1500);
    },
    /** Tear the worker down; a running conversion is rejected as cancelled. */
    dispose: function () {
      const run = settle();
      pending = null;
      if (worker) worker.terminate();
      worker = null;
      workerReady = false;
      if (run) run.reject(abortError(false));
    },
  };
}

function errMessage(err) {
  return String(err && err.message ? err.message : err);
}

function isGpuAvailabilityFailure(err) {
  if (err && err.name === "AbortError") return false;
  return /webgpu|gpu adapter/i.test(errMessage(err));
}

/** Run the job on one pipeline without fallback logic. */
async function runOn(which, job, wasmUrl, signal) {
  if (which === "webgpu") {
    const { convertVideoToAnaglyphGpu, createGpuContext } = await import(
      "./gpu-video/gpu-pipeline.js"
    );
    // The adapter is requested here and handed to the pipeline, which owns
    // it for the run; main-thread runs destroy it again afterwards.
    const gpu = await createGpuContext();
    try {
      return await convertVideoToAnaglyphGpu({ ...job, gpu, signal });
    } finally {
      if (gpu.ownsDevice) gpu.device.destroy();
    }
  }
  const [{ convertVideoToAnaglyph }, factories] = await Promise.all([
    import("./js/video-pipeline.js"),
    import("./js/wasm-factories.js"),
  ]);
  await factories.ensureWasm(wasmUrl);
  return await convertVideoToAnaglyph({
    ...job,
    createBlender: factories.createBlender,
    createTracker: factories.createTracker,
    createYuvConverter: factories.createYuvConverter,
    signal,
  });
}
