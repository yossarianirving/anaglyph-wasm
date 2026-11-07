/**
 * Browser video -> anaglyph video pipeline: WebCodecs hardware decode/encode
 * around the wasm WebGPU blender.
 *
 *   demux (mediabunny) ──► VideoDecoder ──► RGBA copy ──► pairing strategy
 *        │                                                       │
 *        │                                          wasm VideoBlender
 *        │                                                       │
 *   mux (mediabunny) ◄── VideoEncoder ◄── blended RGBA ◄─────────┘
 *
 * The compressed audio track is stream-copied straight from the source into
 * the output container (no decode/encode round trip) when the codec fits the
 * chosen container; otherwise the video is written without audio and the
 * reason is reported in the conversion report.
 *
 * Pairing (js/pairing.js) comes in two flavours: fixed eye assignment for
 * the whole clip, or `auto`, where every frame's horizontal drift feeds the
 * wasm `Tracker` (mirroring anaglyph-rs `Pairer::Auto`) and pairs are emitted
 * once their trend is confirmed — swapping eyes at detected reversals.
 *
 * Timestamps inside this module are microseconds (the WebCodecs unit);
 * mediabunny packets use seconds and are converted at its boundary. All
 * output timestamps are relative to the first pair (`t0`), matching the
 * native pipeline, which starts its output at the first pair's timestamp.
 */

import {
  ALL_FORMATS,
  BlobSource,
  BufferTarget,
  EncodedAudioPacketSource,
  EncodedPacket,
  EncodedPacketSink,
  EncodedVideoPacketSource,
  Input,
  Mp4OutputFormat,
  Output,
  WebMOutputFormat,
} from "../vendor/mediabunny.mjs";
import { createPairStrategy } from "./pairing.js";

/** Encoded chunks handed to the decoder before it has drained. */
const DECODE_QUEUE_LIMIT = 32;
/** Frames handed to the encoder before it has drained. */
const ENCODE_QUEUE_LIMIT = 8;
/** Request an encoder keyframe about every N seconds (seeking-friendly files). */
const KEYFRAME_INTERVAL_S = 2;

class PipelineError extends Error {}

/** Wait until `codec`'s pending-work queue is at or below `limit`. */
async function drainQueue(codec, queueSizeProperty, limit) {
  while (codec.state === "configured" && codec[queueSizeProperty] > limit) {
    await new Promise((resolve) => {
      codec.addEventListener("dequeue", resolve, { once: true });
    });
  }
}

/**
 * Pick an H.264 High-profile level code that covers the frame size and rate.
 * Deliberately conservative: an over-generous level is always valid.
 */
function pickAvcLevel(width, height, frameRate) {
  const pixels = width * height;
  if (pixels <= 1280 * 720) return 0x29; // 4.1
  if (pixels <= 1920 * 1080) return frameRate > 30 ? 0x2a : 0x28; // 4.2 / 4.0
  if (pixels <= 2560 * 1440) return 0x33; // 5.1
  return 0x34; // 5.2
}

/** Rough target bitrate: mid-quality VBR, clamped for tiny and huge frames. */
function targetBitrate(width, height, frameRate) {
  return Math.round(
    Math.min(12_000_000, Math.max(1_500_000, width * height * frameRate * 0.1)),
  );
}

/** Read the metadata + per-track config the pipeline needs up front. */
async function inspectSource(file) {
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  const videoTrack = await input.getPrimaryVideoTrack();
  if (!videoTrack) {
    input.dispose();
    throw new PipelineError("the uploaded file has no video track");
  }
  const audioTrack = await input.getPrimaryAudioTrack();

  const decoderConfig = await videoTrack.getDecoderConfig();
  if (!decoderConfig) {
    input.dispose();
    throw new PipelineError("could not read the video codec configuration");
  }

  // Bounded packet probe (30 packets) for fps/bitrate; both only size the
  // encoder and the progress read-out.
  const stats = await videoTrack.computePacketStats(30);
  const frameRate = Math.round(stats.averagePacketRate) || 30;
  const duration = await input.getDurationFromMetadata().catch(() => null);

  return {
    input,
    videoTrack,
    audioTrack,
    audioCodec: audioTrack ? await audioTrack.getCodec() : null,
    codecString: decoderConfig.codec,
    decoderConfig,
    codedWidth: await videoTrack.getCodedWidth(),
    codedHeight: await videoTrack.getCodedHeight(),
    rotation: await videoTrack.getRotation(),
    flip: await videoTrack.getFlip(),
    frameRate,
    duration,
  };
}

/**
 * Choose the output encoder: try H.264 (MP4) first, then VP9, VP8 and AV1
 * (WebM); within each codec the hardware preference order comes from
 * `hwEncode` — on: `prefer-hardware` then browser-chosen; off:
 * `prefer-software` then browser-chosen. VP8 sits behind VP9 for quality but
 * in front of AV1: where VP9 encode is software-only (typical Firefox), VP8
 * is several times faster at comparable small-frame quality. `codecHint`
 * ("avc" | "vp9" | "vp8" | "av1") moves that family to the front of the
 * order — the benchmark uses it to compare encoder throughput per family on
 * one machine. Returns null when WebCodecs rejects every candidate.
 *
 * Exported for the benchmark page's encoder-only upper-bound probe.
 */
export async function planEncoder(width, height, frameRate, hwEncode, latencyMode, codecHint) {
  if (typeof VideoEncoder === "undefined") return null;

  const level = pickAvcLevel(width, height, frameRate)
    .toString(16)
    .padStart(2, "0");
  let codecs = [
    {
      container: "mp4",
      format: () => new Mp4OutputFormat({ fastStart: "in-memory" }),
      mimeType: "video/mp4",
      videoSourceCodec: "avc",
      codec: `avc1.6400${level}`,
      extra: { avc: { format: "avc" } },
    },
    {
      container: "webm",
      format: () => new WebMOutputFormat(),
      mimeType: "video/webm",
      videoSourceCodec: "vp9",
      codec: "vp09.00.10.08",
      extra: {},
    },
    {
      container: "webm",
      format: () => new WebMOutputFormat(),
      mimeType: "video/webm",
      videoSourceCodec: "vp8",
      codec: "vp8",
      extra: {},
    },
    {
      container: "webm",
      format: () => new WebMOutputFormat(),
      mimeType: "video/webm",
      videoSourceCodec: "av1",
      codec: "av01.0.08M.08",
      extra: {},
    },
  ];
  const accels = hwEncode
    ? ["prefer-hardware", "no-preference"]
    : ["prefer-software", "no-preference"];
  if (codecHint) {
    // Stable move of the hinted family to the front (Array#sort is stable).
    const rank = (c) => (c.videoSourceCodec === codecHint ? 0 : 1);
    codecs = [...codecs].sort((a, b) => rank(a) - rank(b));
  }

  for (const candidate of codecs.flatMap((c) => accels.map((accel) => ({ ...c, accel })))) {
    const config = {
      codec: candidate.codec,
      width,
      height,
      framerate: frameRate,
      bitrate: targetBitrate(width, height, frameRate),
      hardwareAcceleration: accelOrNull(candidate.accel),
      latencyMode,
      ...candidate.extra,
    };
    const support = await VideoEncoder.isConfigSupported(config).catch(
      () => ({ supported: false }),
    );
    if (support.supported) {
      return { ...candidate, format: candidate.format(), encoderConfig: config };
    }
  }
  return null;
}

/** `hardwareAcceleration` is optional; "no-preference" is the default anyway. */
function accelOrNull(accel) {
  return accel === "no-preference" ? undefined : accel;
}

/** Human label for a codec's chosen `hardwareAcceleration` preference. */
function accelLabel(accel) {
  return accel === "prefer-hardware"
    ? "hardware-preferred"
    : accel === "prefer-software"
      ? "software-forced"
      : "browser-chosen";
}

/**
 * Decide whether the source audio can be stream-copied into the output
 * container. MP4 carries AAC; WebM carries Opus (or Vorbis). Anything else
 * stays video-only: browsers cannot reliably encode AAC and audio
 * transcoding is out of scope for the copy path.
 */
function planAudioCopy(audioCodec, container) {
  if (!audioCodec) return { copy: null, reason: "source has no audio track" };
  if (container === "mp4" && audioCodec === "aac") return { copy: "aac", reason: null };
  if (container === "webm" && (audioCodec === "opus" || audioCodec === "vorbis")) {
    return { copy: audioCodec, reason: null };
  }
  return {
    copy: null,
    reason: `audio codec "${audioCodec}" cannot be stream-copied into ${container.toUpperCase()} output`,
  };
}

/**
 * Copy a VideoColorSpace into a plain VideoColorSpaceInit. The WebIDL
 * properties live on the prototype, so an object spread would yield `{}`.
 *
 * Sanitized: WebKit reports `primaries: "unspecified"` for SPSes without
 * colour metadata (not a legal VideoColorPrimaries), and the value echoes
 * through the encoder's first-chunk metadata into the muxer, where
 * mediabunny rejects it. A colorSpace is only passed through when all
 * three fields are concrete and inside the sets the muxer can write into
 * colr boxes; otherwise the encoder's own default (bt709) is used.
 */
const MULTIPLEXABLE_COLOR_SPACE = {
  primaries: new Set(["bt709", "bt470bg", "smpte170m", "bt2020", "smpte432"]),
  transfer: new Set(["bt709", "smpte170m", "linear", "iec61966-2-1", "pq", "hlg"]),
  matrix: new Set(["rgb", "bt709", "bt470bg", "smpte170m", "bt2020-ncl"]),
};

function isMultiplexableColorSpace(colorSpace) {
  return (
    !!colorSpace &&
    MULTIPLEXABLE_COLOR_SPACE.primaries.has(colorSpace.primaries) &&
    MULTIPLEXABLE_COLOR_SPACE.transfer.has(colorSpace.transfer) &&
    MULTIPLEXABLE_COLOR_SPACE.matrix.has(colorSpace.matrix)
  );
}

/**
 * Same sanitization applied to the encoder's first-chunk metadata before
 * handing it to mediabunny: an encoder that echoes an unsupported
 * colorSpace (older WebKit did) must not poison the muxer. Dropping the
 * colorSpace leaves the container's untagged default, never a crash.
 */
function sanitizeChunkMeta(meta) {
  const cs = meta?.decoderConfig?.colorSpace;
  if (!cs || isMultiplexableColorSpace(cs)) return meta;
  const { colorSpace, ...decoderConfig } = meta.decoderConfig;
  return { ...meta, decoderConfig };
}

function colorSpaceInit(colorSpace) {
  if (!isMultiplexableColorSpace(colorSpace)) return undefined;
  return {
    primaries: colorSpace.primaries,
    transfer: colorSpace.transfer,
    matrix: colorSpace.matrix,
    fullRange: colorSpace.fullRange,
  };
}

/**
 * Decide once, per conversion, how decoded frames become dense RGBA:
 *
 * - `"copyTo"` — the browser implements spec `copyTo` conversion to RGBA
 *   (Chromium, modern Firefox).
 * - `"canvas"` — rasterize through `createImageBitmap` + `OffscreenCanvas`.
 *   WebKit (Safari, iOS, WebKitGTK) ignores `copyTo`'s `format` and hands
 *   back the frame's native NV12/I420 planes (with a non-standard
 *   `[{offset, stride}]` return); reading those as RGBA is the garbled
 *   gray output iOS users saw.
 *
 * Detection probes a 4x2 region: the return must report exactly
 * `4*4*2` bytes and the first pixel's alpha lane must be 255 — raw luma
 * planes fail both.
 */
async function detectCopyMode(videoFrame) {
  if (typeof videoFrame.copyTo !== "function") return "canvas";
  try {
    const probe = new Uint8Array(4 * 2 * 4);
    const written = await videoFrame.copyTo(probe, {
      format: "RGBA",
      rect: { x: 0, y: 0, width: 4, height: 2 },
    });
    const bytes = typeof written === "number" ? written : written?.bytesWritten;
    if (bytes !== undefined && bytes !== probe.byteLength) return "canvas";
    // An opaque decoded frame converted to RGBA has alpha 255; NV12 bytes
    // dumped raw put luma/chroma values there instead.
    if (probe[3] !== 255) return "canvas";
    return "copyTo";
  } catch {
    return "canvas";
  }
}

/**
 * Canvas fallback for `copyTo`: rasterize the frame into a persistent
 * `willReadFrequently` 2D context and read dense RGBA into `target` (the
 * frame's pooled buffer). Works in workers, which the pipeline runs in,
 * and performs the YUV→sRGB conversion a spec `copyTo` would do.
 *
 * `rect` is relative to the frame's visible rect (same convention as
 * `copyTo`). `createImageBitmap(VideoFrame)` may hand back the coded or
 * the visible region depending on the engine; the crop origin is anchored
 * accordingly so both pick the same pixels.
 */
async function extractViaCanvas(videoFrame, target, ctx, rect) {
  const bitmap = await createImageBitmap(videoFrame);
  const vis = videoFrame.visibleRect;
  const dx = bitmap.width === videoFrame.codedWidth ? (vis?.x ?? 0) : 0;
  const dy = bitmap.height === videoFrame.codedHeight ? (vis?.y ?? 0) : 0;
  ctx.drawImage(
    bitmap,
    dx + rect.x,
    dy + rect.y,
    rect.width,
    rect.height,
    0,
    0,
    rect.width,
    rect.height,
  );
  bitmap.close?.();
  target.set(ctx.getImageData(0, 0, rect.width, rect.height).data);
}

/** Reusable RGBA frame buffers, pooled because held frames vary with the
 * tracker's confirmation lag (fixed pairing needs two, auto up to its
 * `pending_cap`). */
class BufferPool {
  constructor(byteLength) {
    this.byteLength = byteLength;
    /** @type {ArrayBuffer[]} */
    this.free = [];
    this.held = 0;
    this.maxHeld = 0;
  }

  acquire() {
    const buffer = this.free.pop() ?? new ArrayBuffer(this.byteLength);
    this.held += 1;
    this.maxHeld = Math.max(this.maxHeld, this.held);
    return buffer;
  }

  release(buffer) {
    this.free.push(buffer);
    this.held -= 1;
  }
}

/** One-line summary of detected reversals for the report table. */
function formatReversals(reversals) {
  if (reversals.length === 0) return "none detected";
  return reversals
    .map((r) => `${r.time.toFixed(2)}s ${r.from} -> ${r.to}`)
    .join("; ");
}

/** Render stage time totals compactly for the report table. */
function formatStageMs(stageMs) {
  return `copy ${Math.round(stageMs.copy)} ms · blend ${Math.round(
    stageMs.blend,
  )} ms · encode-wait ${Math.round(stageMs.encodeWait)} ms · decode-wait ${Math.round(
    stageMs.decodeWait,
  )} ms`;
}

/**
 * Convert a video file into an anaglyph video.
 *
 * @param {object} options
 * @param {File | Blob} options.file Source video.
 * @param {"clockwise" | "counter-clockwise" | "auto" | "auto-reverse"}
 *   [options.direction] Eye pairing; `auto` detects the orbit from image
 *   motion (needs `createTracker`).
 * @param {string} [options.anaglyphType] wasm blend flavor name.
 * @param {{ decode?: boolean, encode?: boolean, webgpu?: boolean }}
 *   [options.acceleration] Hardware preferences. `decode` and `encode`
 *   default true; `webgpu` defaults **false**. Turning one off forces the
 *   software path (`prefer-software` for the codecs, CPU blend for WebGPU)
 *   with a browser-chosen fallback when the forced hint is unsupported;
 *   useful on browsers whose hardware path is broken or slower. WebGPU
 *   blending must be enabled explicitly because the CPU SIMD blend beats it
 *   end-to-end in this architecture: every pair still crosses CPU memory
 *   twice (decoder `copyTo` out, `VideoFrame` into the encoder), so the
 *   blend's upload + `mapAsync` readback round-trip is pure added latency.
 *   Measured slower with a real GPU as well (see BENCH.md).
 * @param {"quality" | "realtime"} [options.encoderLatencyMode] Encoder
 *   latency mode; "realtime" trades a little quality for significantly lower
 *   encode latency (and buffering stalls) on some browsers.
 * @param {"avc" | "vp9" | "vp8" | "av1"} [options.encoderCodec] Move one
 *   codec family to the front of the encoder preference order (benchmarking
 *   aid; unsupported hints fall through to the normal order).
 * @param {(width: number, height: number, anaglyphType: string, backend: "cpu" | undefined) => Promise<{blend: (left: Uint8Array, right: Uint8Array) => Promise<Uint8Array>, is_gpu: () => boolean}>} options.createBlender
 *   Factory for the wasm blender. Injected so this module stays free of the
 *   wasm import and remains runnable (with a stub blender) outside a browser.
 *   `backend` is `"cpu"` when WebGPU blending was switched off.
 * @param {(width: number, height: number, reverse: boolean) => Promise<{tracker: object, observe: (frame: object) => {dx: number, reliable: boolean}}>} [options.createTracker]
 *   Factory for the auto-direction tracker + motion observer (wasm exports
 *   `Tracker` and `MotionAnalyzer` in the browser; tests pass stubs).
 * @param {(status: object) => void} [options.onProgress]
 * @param {AbortSignal} [options.signal]
 * @returns {Promise<{ blob: Blob, report: object }>}
 */
export async function convertVideoToAnaglyph(options) {
  const {
    file,
    direction = "clockwise",
    anaglyphType = "color",
    acceleration = {},
    encoderLatencyMode = "quality",
    encoderCodec = null,
    createBlender,
    createTracker,
    onProgress = () => {},
    signal,
  } = options;
  const hwDecode = acceleration.decode !== false;
  const hwEncode = acceleration.encode !== false;
  // Opt-in: CPU SIMD blending wins end-to-end (no GPU round-trips); see BENCH.md.
  const webgpuBlend = acceleration.webgpu === true;

  const startedAt = performance.now();
  const checkAbort = () => {
    if (signal?.aborted) {
      const error = new Error("conversion cancelled");
      error.name = "AbortError";
      throw error;
    }
  };

  /**
   * Rejection that fires the instant the signal aborts (pre-fired when it
   * already has). The no-op catch marks it as handled so runs that never
   * await it — every run that finishes normally — do not surface an
   * unhandled-rejection warning.
   */
  const abortRace = new Promise((_, reject) => {
    const abort = () => {
      const error = new Error("conversion cancelled");
      error.name = "AbortError";
      reject(error);
    };
    if (signal) {
      if (signal.aborted) abort();
      else signal.addEventListener("abort", abort, { once: true });
    }
  });
  abortRace.catch(() => {});

  /**
   * Await `op`, but bail out as soon as the run is cancelled. WebCodecs
   * awaits (`flush()`, a `dequeue` listener, a detached `copyTo`) may never
   * settle once the codecs are torn down — an engine-dependent hazard seen
   * on WebKit-family browsers — and a never-settling await here would strand
   * the caller (and the worker's busy guard) forever: the reported "cancel
   * works but the page is dead until reload" bug. Racing makes cancellation
   * deterministic; the finally block still closes every codec.
   */
  const withAbort = (op) =>
    signal ? Promise.race([op instanceof Promise ? op : Promise.resolve(op), abortRace]) : op;

  const source = await inspectSource(file);
  /** @type {VideoDecoder | null} */
  let decoder = null;
  /** @type {VideoEncoder | null} */
  let encoder = null;
  /** Resized once the frame dimensions are known; never acquires before. */
  let pool = new BufferPool(0);
  /** @type {Map<number, object>} frames whose buffer is not yet released */
  const pendingEntries = new Map();
  /** @type {object[]} reversals confirmed during the run (auto mode) */
  const reversalsFound = [];
  /**
   * Stage time totals in milliseconds, measured inside the processing
   * steps (not wall-clock splits: stages overlap asynchronously). The
   * report shows them; the bench page uses them for its breakdown.
   */
  const stageMs = { copy: 0, blend: 0, encodeWait: 0, decodeWait: 0 };
  /** 0-based index of the next decoded frame (the tracker's frame index). */
  let nextFrameIndex = 0;

  try {
    // Decoder preference: hardware first by default, forced software when
    // switched off; "no-preference" remains the fallback whenever a hint is
    // unsupported for this codec.
    const decodePrefs = hwDecode
      ? [["prefer-hardware", "hardware-preferred"], [undefined, "browser-chosen"]]
      : [["prefer-software", "software-forced"], [undefined, "browser-chosen"]];
    let decoderConfig = null;
    let decoderMode = "browser-chosen";
    for (const [accel, mode] of decodePrefs) {
      const candidate = accel
        ? { ...source.decoderConfig, hardwareAcceleration: accel }
        : { ...source.decoderConfig };
      const support = await VideoDecoder.isConfigSupported(candidate).catch(
        () => ({ supported: false }),
      );
      if (support.supported) {
        decoderConfig = candidate;
        decoderMode = mode;
        break;
      }
    }
    if (!decoderConfig) {
      throw new PipelineError(`this browser cannot decode "${source.codecString}"`);
    }

    // The encoder, blender, muxer, tracker and buffer pool are built lazily
    // from the first *decoded* frame, not from container metadata: decoders
    // differ in how they treat the container's rotation and coded padding
    // (Safari/VideoToolbox bakes transforms into frames, others pass them
    // through untouched). Planning from the frames themselves keeps the RGBA
    // copy size, blend dimensions and encoder configuration consistent on
    // every browser, which is what prevented the garbled mobile-Safari
    // outputs (transposed frames copied into untransposed strides show up
    // as tiled gray anaglyphs).
    /**
     * @typedef {object} PipelineCore
     * @property {number} width
     * @property {number} height
     * @property {{ x: number, y: number, width: number, height: number }} copyRect
     * @property {"copyTo" | "canvas"} copyMode
     * @property {OffscreenCanvasRenderingContext2D|null} copyCtx
     * @property {{ blend: Function, is_gpu: () => boolean }} blender
     * @property {object} strategy
     * @property {object} plan
     * @property {{ copy: string|null, reason: string|null }} audio
     * @property {EncodedAudioPacketSource|null} audioSource
     * @property {Output} output
     * @property {EncodedVideoPacketSource} videoSource
     * @property {string} rotationNote
     */
    /** @type {PipelineCore | null} */
    let core = null;

    // Stages WebCodecs exposes only through callbacks are serialized onto
    // promise chains: frame processing (decoder output) and muxing (encoder
    // output). A callback error is latched and rethrown by the main flow.
    /** @type {Error | null} */
    let fatalError = null;
    let frameChain = Promise.resolve();
    let muxChain = Promise.resolve();
    let videoChunks = 0;

    /** Latch an error from a detached chain step (never reject the chain:
     * pending steps must still run their cleanup, and the main flow rethrows
     * `fatalError` — a rejected chain would surface as an unhandled
     * rejection after a cancel abandoned the awaits). */
    const latch = (e) => {
      fatalError ??= e instanceof Error ? e : new Error(String(e));
    };

    /** Build everything that needs real frame dimensions. Runs once, on the
     * first decoded frame, inside `frameChain`. */
    async function planFromFrame(videoFrame) {
      const vis = videoFrame.visibleRect ?? {
        x: 0,
        y: 0,
        width: videoFrame.codedWidth,
        height: videoFrame.codedHeight,
      };
      // H.264 needs even frame dimensions; blend and encode on the cropped
      // visible area.
      const width = vis.width & ~1;
      const height = vis.height & ~1;
      if (width < 2 || height < 2) {
        throw new PipelineError("video frames are too small to convert");
      }

      // Rotation/flip: container metadata is passed straight through when
      // the decoder hands back frames in the container's coded geometry. If
      // the frames arrive in the *displayed* geometry instead (a decoder
      // that baked the transform, Safari among them), baking must not be
      // applied twice, so the output carries no transform. A 180-degree
      // rotation cannot be told apart by geometry alone; the coded-geometry
      // case wins there, matching every decoder tested so far.
      const srcRot = source.rotation ?? 0;
      const srcFlip = source.flip ?? false;
      const swap = Math.abs(srcRot % 180) === 90;
      const codedMatch =
        videoFrame.codedWidth === source.codedWidth &&
        videoFrame.codedHeight === source.codedHeight;
      const displayMatch =
        videoFrame.codedWidth === (swap ? source.codedHeight : source.codedWidth) &&
        videoFrame.codedHeight === (swap ? source.codedWidth : source.codedHeight);
      let rotation = srcRot;
      let flip = srcFlip;
      let rotationNote;
      if (codedMatch || (!displayMatch && srcRot === 0 && !srcFlip)) {
        rotationNote = srcRot ? `${srcRot}\u00b0 (container metadata)` : "none";
      } else if (displayMatch && (srcRot !== 0 || srcFlip)) {
        rotation = 0;
        flip = false;
        rotationNote = `${srcRot}\u00b0${srcFlip ? " + flip" : ""} baked in by decoder`;
      } else {
        rotationNote = `${srcRot}\u00b0 (container) on ${videoFrame.codedWidth}\u00d7${videoFrame.codedHeight} frames`;
      }

      // How do decoded frames become dense RGBA in this browser? WebKit
      // silently hands back raw NV12/I420 planes for a "RGBA" copyTo; the
      // probe catches that and every frame then goes through the canvas
      // raster path instead. See detectCopyMode().
      const copyMode = await detectCopyMode(videoFrame);
      let copyCtx = null;
      if (copyMode === "canvas") {
        if (typeof OffscreenCanvas === "undefined" || typeof createImageBitmap !== "function") {
          throw new PipelineError(
            "this browser cannot convert decoded video frames to RGBA (no copyTo conversion, no OffscreenCanvas)",
          );
        }
        copyCtx = new OffscreenCanvas(width, height).getContext("2d", {
          willReadFrequently: true,
        });
      }

      const plan = await planEncoder(
        width,
        height,
        source.frameRate,
        hwEncode,
        encoderLatencyMode,
        encoderCodec,
      );
      if (!plan) {
        throw new PipelineError("no supported video encoder found (tried H.264, VP9, VP8, AV1)");
      }
      const audio = planAudioCopy(source.audioCodec, plan.container);

      // ---- Output: muxer with compressed packet sources ----
      const output = new Output({ format: plan.format, target: new BufferTarget() });
      const videoSource = new EncodedVideoPacketSource(plan.videoSourceCodec);
      // No frameRate metadata: the packet timestamps carry the true cadence,
      // and a rounded fps here would pick a coarse timescale that collides
      // (duplicate DTS) for rates like 33.33.
      output.addVideoTrack(videoSource, { rotation, flip });
      /** @type {EncodedAudioPacketSource | null} */
      let audioSource = null;
      if (audio.copy) {
        audioSource = new EncodedAudioPacketSource(audio.copy);
        output.addAudioTrack(audioSource);
      }
      await output.start();

      encoder = new VideoEncoder({
        output(chunk, meta) {
          const data = new Uint8Array(chunk.byteLength);
          chunk.copyTo(data);
          const packet = new EncodedPacket(
            data,
            chunk.type,
            (chunk.timestamp - t0Us) / 1e6,
            (chunk.duration ?? fallbackDurationUs) / 1e6,
          );
          // The first chunk carries the decoder initialization config (avcC,
          // vpcC, ...); later ones need no meta.
          const isFirst = videoChunks++ === 0;
          const safeMeta = isFirst ? sanitizeChunkMeta(meta) : undefined;
          muxChain = muxChain
            .then(() => videoSource.add(packet, safeMeta))
            .catch(latch);
        },
        error(e) {
          fatalError ??= e instanceof Error ? e : new Error(String(e));
        },
      });
      encoder.configure(plan.encoderConfig);

      const blender = await createBlender(
        width,
        height,
        anaglyphType,
        webgpuBlend ? undefined : "cpu",
      );
      pool = new BufferPool(width * height * 4);

      let strategy;
      if (direction === "auto" || direction === "auto-reverse") {
        if (!createTracker) {
          throw new PipelineError(`direction "${direction}" needs a tracker factory`);
        }
        const auto = await createTracker(width, height, direction === "auto-reverse");
        strategy = createPairStrategy(direction, auto);
      } else {
        strategy = createPairStrategy(direction);
      }

      return {
        width,
        height,
        // Crop rect for both frame-copy paths, relative to the frame's
        // visible rect origin (the convention `copyTo`'s rect uses).
        copyRect: { x: 0, y: 0, width, height },
        copyMode,
        copyCtx,
        blender,
        strategy,
        plan,
        audio,
        audioSource,
        output,
        videoSource,
        rotationNote,
      };
    }

    const keyframeEvery = Math.max(
      1,
      Math.round(source.frameRate * KEYFRAME_INTERVAL_S),
    );
    const fallbackDurationUs = Math.round(1e6 / source.frameRate);
    let pairCount = 0;
    /** @type {number | null} */
    let t0Us = null;

    /** Blend an already-paired pair (owned RGBA buffers) and encode it. */
    async function emitPair(pair) {
      t0Us ??= pair.timestamp;
      const blendStart = performance.now();
      const blended = await core.blender.blend(
        new Uint8Array(pair.left.rgba),
        new Uint8Array(pair.right.rgba),
      );
      stageMs.blend += performance.now() - blendStart;

      const waitStart = performance.now();
      await withAbort(drainQueue(encoder, "encodeQueueSize", ENCODE_QUEUE_LIMIT));
      stageMs.encodeWait += performance.now() - waitStart;
      checkAbort();
      const outFrame = new VideoFrame(blended, {
        format: "RGBA",
        codedWidth: core.width,
        codedHeight: core.height,
        timestamp: pair.timestamp - t0Us,
        duration: pair.duration > 0 ? pair.duration : fallbackDurationUs,
        alphaMode: "discard",
        colorSpace: pair.left.colorSpace ?? pair.right.colorSpace,
      });
      encoder.encode(outFrame, { keyFrame: pairCount % keyframeEvery === 0 });
      outFrame.close();
      pairCount += 1;

      if (pairCount % 10 === 0) reportProgress();
    }

    /** Drain what the strategy just confirmed, then recycle spent buffers. */
    async function handleStrategyResult(result) {
      if (result.reversals.length > 0) {
        reversalsFound.push(...result.reversals);
      }
      for (const pair of result.pairs) {
        await emitPair(pair);
      }
      if (result.releasedBefore >= 0) {
        for (const [index, entry] of pendingEntries) {
          if (index >= result.releasedBefore) break; // insertion == index order
          pool.release(entry.rgba);
          entry.rgba = null;
          pendingEntries.delete(index);
        }
      }
    }

    function reportProgress() {
      const elapsed = (performance.now() - startedAt) / 1e3;
      onProgress({
        stage: "convert",
        frames: pairCount,
        totalFrames: source.duration
          ? Math.round(source.duration * source.frameRate)
          : null,
        fps: elapsed > 0 ? pairCount / elapsed : 0,
        gpuBlend: core.blender.is_gpu(),
        auto: core.strategy.auto,
        reversals: reversalsFound.length,
      });
    }

    decoder = new VideoDecoder({
      output(videoFrame) {
        // Every frame is copied to an owned RGBA buffer immediately and the
        // VideoFrame closed: held-back (auto) frames would otherwise pin
        // decoder-owned memory, and motion analysis + blending both need
        // dense pixels. All processing runs serialized on frameChain, so a
        // frame's buffer always outlives the pairs that reference it, and
        // the first frame also plans the rest of the pipeline.
        const entry = {
          index: nextFrameIndex++,
          timestamp: videoFrame.timestamp,
          duration: videoFrame.duration ?? 0,
          colorSpace: colorSpaceInit(videoFrame.colorSpace),
          rgba: null,
        };
        frameChain = frameChain
          .then(async () => {
            core ??= await planFromFrame(videoFrame);
            entry.rgba = pool.acquire();
            const copyStart = performance.now();
            const target = new Uint8Array(entry.rgba);
            if (core.copyMode === "canvas") {
              await extractViaCanvas(videoFrame, target, core.copyCtx, core.copyRect);
            } else {
              const written = await videoFrame.copyTo(target, {
                format: "RGBA",
                rect: core.copyRect,
              });
              // copyTo returns bytes written: a number in older implementations,
              // a VideoFrameCopyToResult ({ length, bytesWritten }) now; some
              // builds resolve undefined and rely on throwing for short copies.
              const bytes =
                typeof written === "number"
                  ? written
                  : (written?.bytesWritten ?? entry.rgba.byteLength);
              if (bytes !== entry.rgba.byteLength) {
                throw new PipelineError(
                  `decoder frame copy wrote ${bytes} of ${entry.rgba.byteLength} expected bytes`,
                );
              }
            }
            stageMs.copy += performance.now() - copyStart;
            videoFrame.close();
            pendingEntries.set(entry.index, entry);
            await handleStrategyResult(core.strategy.push(entry));
          })
          .catch((e) => {
            // Planning, copy failures and cancellation land here; pending
            // steps still run (closing their VideoFrame), the main flow
            // rethrows.
            latch(e);
            try {
              videoFrame.close();
            } catch {
              /* already closed */
            }
          });
      },
      error(e) {
        fatalError ??= e instanceof Error ? e : new Error(String(e));
      },
    });
    decoder.configure(decoderConfig);

    // ---- Feed the decoder from the demuxer ----
    const packetSink = new EncodedPacketSink(source.videoTrack);
    for await (const packet of packetSink.packets()) {
      checkAbort();
      if (fatalError) throw fatalError;
      decoder.decode(packet.toEncodedVideoChunk());
      const waitStart = performance.now();
      await withAbort(drainQueue(decoder, "decodeQueueSize", DECODE_QUEUE_LIMIT));
      stageMs.decodeWait += performance.now() - waitStart;
    }
    // End-of-stream awaits are raced against cancellation: a codec that
    // never settles its flush after teardown must not strand this function
    // (and with it the worker's busy guard).
    await withAbort(decoder.flush());
    await withAbort(frameChain);
    if (fatalError) throw fatalError;

    // End of stream: the strategy commits every held-back pair (the last
    // frame never pairs; like the native pipeline, N inputs give N-1 pairs).
    await handleStrategyResult(core ? core.strategy.flush() : { pairs: [], reversals: [], releasedBefore: -1 });

    if (pairCount === 0) {
      throw new PipelineError(
        "video has fewer than two frames; anaglyph conversion needs at least one frame pair",
      );
    }

    await withAbort(encoder.flush());
    await withAbort(muxChain);
    if (fatalError) throw fatalError;
    reportProgress();

    // ---- Audio: stream-copy compressed packets onto the video timeline ----
    let audioNote = core.audio.reason;
    if (core.audioSource) {
      const audioSink = new EncodedPacketSink(source.audioTrack);
      const decoderConfig = await source.audioTrack.getDecoderConfig();
      const meta = decoderConfig ? { decoderConfig } : undefined;
      const shiftS = (t0Us ?? 0) / 1e6;
      let firstAudio = true;
      for await (const packet of audioSink.packets()) {
        checkAbort();
        const shifted =
          shiftS === 0 ? packet : packet.clone({ timestamp: packet.timestamp - shiftS });
        await core.audioSource.add(shifted, firstAudio ? meta : undefined);
        firstAudio = false;
      }
      audioNote = `${core.audio.copy} (stream copy)`;
    }

    // Last cancellation checkpoint: finalizing the muxer after an abort is
    // pointless and would race the packet sources still being drained.
    checkAbort();
    await core.output.finalize();
    const blob = new Blob([core.output.target.buffer], { type: core.plan.mimeType });

    const elapsed = (performance.now() - startedAt) / 1e3;
    return {
      blob,
      report: {
        frames: pairCount,
        seconds: elapsed,
        fps: elapsed > 0 ? pairCount / elapsed : null,
        container: core.plan.container,
        encoder: `${core.plan.codec} (${accelLabel(core.plan.accel)})`,
        decoder: `${source.codecString} (${decoderMode})`,
        blender: core.blender.is_gpu() ? "WebGPU" : "CPU (fallback)",
        frameCopy: core.copyMode === "canvas" ? "canvas raster" : "copyTo",
        output: `${core.width}\u00d7${core.height}`,
        rotation: core.rotationNote,
        direction: core.strategy.auto
          ? direction === "auto-reverse"
            ? "auto (reversed)"
            : "auto"
          : direction,
        reversals: core.strategy.auto ? formatReversals(reversalsFound) : "n/a",
        audio: audioNote ?? "none",
        width: source.codedWidth,
        height: source.codedHeight,
        frameBuffers: pool.maxHeld,
        timings: formatStageMs(stageMs),
      },
      // Machine-readable stage totals for the benchmark page/harness
      // (`report.timings` above is the human-rendered sibling).
      stageMs,
    };
  } finally {
    // Release codec and buffer resources on every exit path (success, error,
    // cancel). close() on an already-closed codec throws; ignore.
    for (const codec of [decoder, encoder]) {
      try {
        codec?.close();
      } catch {
        /* already closed */
      }
    }
    for (const entry of pendingEntries.values()) {
      if (entry.rgba) pool.release(entry.rgba);
    }
    pendingEntries.clear();
    source.input.dispose();
  }
}
