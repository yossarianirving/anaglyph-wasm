/**
 * WebGPU video -> anaglyph pipeline: JavaScript + WebGPU + mediabunny only.
 *
 *   demux (mediabunny) ──► VideoDecoder ──► VideoFrame ──► pairing strategy
 *        │                                    │    (GPU-resident frames)      │
 *        │                    motion passes ◄──┘                              │
 *        │                    (luma + box + SAD compute)                      │
 *        │                                                                    │
 *   mux (mediabunny) ◄── VideoEncoder ◄── canvas capture ◄── blend fragment ──┘
 *
 * No video pixel ever crosses the CPU/GPU boundary: decoded VideoFrames are
 * imported as external textures for the blend fragment shader and for the
 * motion passes (a luma fragment pass, an exact box downsample, and a SAD
 * compute kernel mirroring anaglyph-rs `motion.rs` — see shaders.js and
 * motion.js), and the encoded side captures the blended result through a
 * WebGPU canvas so the encoder receives a GPU-backed VideoFrame. The only
 * readback left is the tiny per-frame SAD result buffer (~8 KB) plus a
 * 256-byte completion gate per pair.
 *
 * Pairing reuses `js/pairing.js` unchanged; the tracker is the JS port in
 * `tracker.js`. Muxing, encoder planning, audio stream-copy and the
 * rotation/cancel semantics mirror `js/video-pipeline.js` (that file remains
 * the reference for engine quirks; copies here are deliberate so this folder
 * is self-contained).
 *
 * Timestamps inside this module are microseconds; mediabunny packets use
 * seconds and are converted at its boundary. Output timestamps are relative
 * to the first pair (`t0`), matching the native pipeline.
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
import { createPairStrategy } from "../js/pairing.js";
import { DirectionTracker } from "./tracker.js";
import { BLEND_SHADER, BLIT_SHADER } from "./shaders.js";
import { GpuMotion } from "./motion.js";

/** Encoded chunks handed to the decoder before it has drained. */
const DECODE_QUEUE_LIMIT = 32;
/** Frames handed to the encoder before it has drained. */
const ENCODE_QUEUE_LIMIT = 8;
/** Request an encoder keyframe about every N seconds (seeking-friendly files). */
const KEYFRAME_INTERVAL_S = 2;

class PipelineError extends Error {}

/**
 * Eye matrices from anaglyph-rs `anaglyph.rs`, flattened row-major
 * (`m[inputChannel][outputChannel]` at `inputChannel*3 + outputChannel`) —
 * the upload order the blend shader's mat3x3 columns expect.
 */
export const ANAGLYPH_MATRICES = {
  true: [
    0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0.299, 0.587, 0.114,
  ],
  grayscale: [
    0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114,
  ],
  optimized: [
    0, 0.7, 0.3, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 1,
  ],
  color: [
    1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 1,
  ],
  "half-color": [
    0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 1,
  ],
};

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
 * `hwEncode`. Mirrors `js/video-pipeline.js`; see the notes there.
 */
async function planEncoder(width, height, frameRate, hwEncode, latencyMode, codecHint) {
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
 * container. MP4 carries AAC; WebM carries Opus (or Vorbis).
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
 * Copy a VideoColorSpace into a plain VideoColorSpaceInit (properties live
 * on the prototype) with the multiplexability sanitization documented in
 * `js/video-pipeline.js`.
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

/** One-line summary of detected reversals for the report table. */
function formatReversals(reversals) {
  if (reversals.length === 0) return "none detected";
  return reversals
    .map((r) => `${r.time.toFixed(2)}s ${r.from} -> ${r.to}`)
    .join("; ");
}

/** Render stage time totals compactly for the report table. */
function formatStageMs(stageMs) {
  const motion = stageMs.motion
    ? ` · motion ${Math.round(stageMs.motion)} ms`
    : "";
  return `gpu-submit ${Math.round(stageMs.blend)} ms · capture ${Math.round(
    stageMs.capture,
  )} ms · encode-wait ${Math.round(stageMs.encodeWait)} ms · decode-wait ${Math.round(
    stageMs.decodeWait,
  )} ms${motion}`;
}

/** MAP_READ / COPY_DST GPU buffer pool (buffers recycle once unmapped). */
class GpuBufferPool {
  /**
   * @param {GPUDevice} device
   * @param {number} size
   * @param {number} usage
   */
  constructor(device, size, usage) {
    this.device = device;
    this.size = size;
    this.usage = usage;
    /** @type {GPUBuffer[]} */
    this.free = [];
    this.created = 0;
    this.maxHeld = 0;
    this.held = 0;
  }

  acquire() {
    let buffer = this.free.find((b) => b.mapState === "unmapped");
    if (buffer) this.free.splice(this.free.indexOf(buffer), 1);
    else {
      buffer = this.device.createBuffer({ size: this.size, usage: this.usage });
      this.created += 1;
    }
    this.held += 1;
    this.maxHeld = Math.max(this.maxHeld, this.held);
    return buffer;
  }

  release(buffer) {
    this.held -= 1;
    this.free.push(buffer);
  }

  destroy() {
    for (const buffer of this.free) buffer.destroy();
    this.free = [];
  }
}

/**
 * Acquire a VideoFrame as an external texture, coping with the
 * dict-vs-positional `importExternalTexture` API skew across engine
 * versions (the dict form is current spec; older builds take the frame
 * directly). The working form is remembered after the first success.
 */
function makeImporter() {
  let dictForm;
  return (encoder, frame) => {
    const tryDict = () => encoder.importExternalTexture({ externalTexture: frame });
    const tryPositional = () => encoder.importExternalTexture(frame);
    if (dictForm === true) return tryDict();
    if (dictForm === false) return tryPositional();
    try {
      const view = tryDict();
      dictForm = true;
      return view;
    } catch {
      const view = tryPositional();
      dictForm = false;
      return view;
    }
  };
}

/**
 * Probe whether this browser can turn a WebGPU canvas into a VideoFrame —
 * the zero-copy encoder input. Draws one clear pass, submits it, and tries
 * the construction; any throw or geometry mismatch downgrades the pipeline
 * to the `copyTextureToBuffer` + `new VideoFrame(bytes)` path.
 *
 * @returns {Promise<{canvas: OffscreenCanvas, ctx: GPUCanvasContext} | null>}
 */
async function probeCanvasCapture(device, format, width, height) {
  try {
    if (typeof OffscreenCanvas === "undefined" || typeof VideoFrame === "undefined") {
      return null;
    }
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext("webgpu");
    if (!ctx) return null;
    ctx.configure({ device, format, alphaMode: "opaque" });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: ctx.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const frame = new VideoFrame(canvas, { timestamp: 0 });
    const ok = frame.codedWidth === width && frame.codedHeight === height;
    frame.close();
    return ok ? { canvas, ctx } : null;
  } catch {
    return null;
  }
}

/**
 * Request a WebGPU device for the pipeline (throws a PipelineError with a
 * user-facing reason when unavailable).
 */
export async function createGpuContext() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    throw new PipelineError(
      "this browser has no WebGPU support (on iOS, Safari 18 or newer is required)",
    );
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new PipelineError("no WebGPU adapter is available");
  }
  const device = await adapter.requestDevice();
  return {
    device,
    adapter,
    canvasFormat: navigator.gpu.getPreferredCanvasFormat(),
    ownsDevice: true,
  };
}

/**
 * Convert a video file into an anaglyph video entirely on the GPU.
 *
 * @param {object} options
 * @param {File | Blob} options.file Source video.
 * @param {"clockwise" | "counter-clockwise" | "auto" | "auto-reverse"}
 *   [options.direction] Eye pairing.
 * @param {"true" | "grayscale" | "optimized" | "color" | "half-color"}
 *   [options.anaglyphType] Blend flavour (matrices from anaglyph-rs).
 * @param {{ decode?: boolean, encode?: boolean }} [options.acceleration]
 *   Hardware preferences for the two codecs.
 * @param {"quality" | "realtime"} [options.encoderLatencyMode]
 * @param {"avc" | "vp9" | "vp8" | "av1"} [options.encoderCodec]
 * @param {{ device: GPUDevice, canvasFormat: GPUTextureFormat }} [options.gpu]
 *   Pre-created context (otherwise one is requested; the run destroys it).
 * @param {boolean} [options.canvasCapture] Set false to force the
 *   `copyTextureToBuffer` encode path (benchmark A/B of the capture modes).
 * @param {(status: object) => void} [options.onProgress]
 * @param {AbortSignal} [options.signal]
 * @returns {Promise<{ blob: Blob, report: object, stageMs: object }>}
 */
export async function convertVideoToAnaglyphGpu(options) {
  const {
    file,
    direction = "clockwise",
    anaglyphType = "color",
    acceleration = {},
    encoderLatencyMode = "quality",
    encoderCodec = null,
    gpu = null,
    canvasCapture = true,
    onProgress = () => {},
    signal,
  } = options;
  const hwDecode = acceleration.decode !== false;
  const hwEncode = acceleration.encode !== false;

  const matrix = ANAGLYPH_MATRICES[anaglyphType];
  if (!matrix) {
    throw new PipelineError(`unknown anaglyph type "${anaglyphType}"`);
  }

  const startedAt = performance.now();
  const checkAbort = () => {
    if (signal?.aborted) {
      const error = new Error("conversion cancelled");
      error.name = "AbortError";
      throw error;
    }
  };

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

  const withAbort = (op) =>
    signal ? Promise.race([op instanceof Promise ? op : Promise.resolve(op), abortRace]) : op;

  const context = gpu ?? (await createGpuContext());
  const device = context.device;
  const importExternal = makeImporter();

  const source = await inspectSource(file);
  /** @type {VideoDecoder | null} */
  let decoder = null;
  /** @type {VideoEncoder | null} */
  let encoder = null;
  /** @type {Map<number, object>} decoded frames not yet closed */
  const pendingEntries = new Map();
  /** @type {object[]} reversals confirmed during the run (auto mode) */
  const reversalsFound = [];
  /** Stage time totals in ms (see the wasm pipeline's identical accounting;
   * `copy` stays 0 — there is no CPU frame copy in this pipeline at all). */
  const stageMs = { copy: 0, blend: 0, capture: 0, encodeWait: 0, decodeWait: 0, motion: 0 };
  /** 0-based index of the next decoded frame (the tracker's frame index). */
  let nextFrameIndex = 0;
  /** High-water mark of decoded VideoFrames held open at once. */
  let maxFramesHeld = 0;

  try {
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

    // ---- GPU blend + capture resources --------------------------------
    // Built lazily on the first decoded frame (dimensions + crop). The
    // strategy/tracker are built there too so `observe` can read the
    // per-frame GPU sample the chain stores right before pushing.
    /** @type {{
    *   width: number, height: number, plan: object, audio: object,
    *   audioSource: EncodedAudioPacketSource | null, output: Output,
    *   videoSource: EncodedVideoPacketSource, rotationNote: string,
    *   strategy: object, motion: GpuMotion | null,
    *   captureMode: "canvas" | "bytes", canvas: OffscreenCanvas | null,
    *   canvasCtx: GPUCanvasContext | null, targetTex: GPUTexture,
    *   blendU: GPUBuffer, blitBind: GPUBindGroup,
    *   gatePool: GpuBufferPool, pixelPool: GpuBufferPool | null,
    *   captureColorSpace: VideoColorSpaceInit | undefined,
    * } | null} */
    let core = null;

    /** @type {Error | null} */
    let fatalError = null;
    // Per-frame chain: motion readback + pairing (ordered). Per-pair chain:
    // blend submit + capture + encode (ordered, follows pairing order).
    // Mux chain: encoder output -> mediabunny (ordered). Errors latch.
    let frameChain = Promise.resolve();
    let emitChain = Promise.resolve();
    let muxChain = Promise.resolve();
    let videoChunks = 0;
    const latch = (e) => {
      fatalError ??= e instanceof Error ? e : new Error(String(e));
    };

    /** Latest GPU motion sample, read synchronously by `observe`. */
    let pendingSample = null;

    async function planFromFrame(videoFrame) {
      const vis = videoFrame.visibleRect ?? {
        x: 0,
        y: 0,
        width: videoFrame.codedWidth,
        height: videoFrame.codedHeight,
      };
      // H.264 needs even frame dimensions; blend and encode on the cropped
      // visible area (the shader's UV crop drops the coded padding).
      const width = vis.width & ~1;
      const height = vis.height & ~1;
      if (width < 2 || height < 2) {
        throw new PipelineError("video frames are too small to convert");
      }

      // Rotation/flip handling identical to js/video-pipeline.js: container
      // metadata passes through unless the decoder baked the transform.
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

      const crop = {
        x0: vis.x / videoFrame.codedWidth,
        y0: vis.y / videoFrame.codedHeight,
        w: width / videoFrame.codedWidth,
        h: height / videoFrame.codedHeight,
      };

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
      // No frameRate metadata: packet timestamps carry the true cadence
      // (a rounded fps picks a coarse timescale that collides at rates
      // like 33.33 — see js/video-pipeline.js).
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

      // Blend pipeline: matrices uploaded once; UV crop per run.
      const blendModule = device.createShaderModule({ code: BLEND_SHADER });
      const blendPipe = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: blendModule, entryPoint: "vs_main" },
        fragment: {
          module: blendModule,
          entryPoint: "fs_main",
          targets: [{ format: "bgra8unorm" }],
        },
        primitive: { topology: "triangle-list" },
      });
      // BlendU: crop vec4 + two mat3x3 (112 bytes, flat row-major upload
      // puts Rust's rows into the WGSL columns — see shaders.js).
      const blendU = device.createBuffer({
        size: 112,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const u32 = new ArrayBuffer(112);
      new Float32Array(u32, 0, 4).set([crop.x0, crop.y0, crop.w, crop.h]);
      new Float32Array(u32, 16, 18).set(matrix);
      device.queue.writeBuffer(blendU, 0, u32);

      const targetTex = device.createTexture({
        format: "bgra8unorm",
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC,
        width,
        height,
        sampleCount: 1,
      });

      // Blit pipeline: targetTex -> canvas (values already sRGB-encoded).
      const blitModule = device.createShaderModule({ code: BLIT_SHADER });
      const blitPipe = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: blitModule, entryPoint: "vs_main" },
        fragment: {
          module: blitModule,
          entryPoint: "fs_main",
          targets: [{ format: context.canvasFormat }],
        },
        primitive: { topology: "triangle-list" },
      });
      const sampler = device.createSampler({
        magFilter: "nearest",
        minFilter: "nearest",
      });
      const blitBind = device.createBindGroup({
        layout: blitPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: targetTex.createView() },
          { binding: 1, resource: sampler },
        ],
      });

      // Encoder input: GPU canvas capture when the engine supports it,
      // else copyTextureToBuffer -> VideoFrame(BGRA).
      const capture = canvasCapture
        ? await probeCanvasCapture(device, context.canvasFormat, width, height)
        : null;
      const captureMode = capture ? "canvas" : "bytes";
      // Gate buffer: one copied row (256 B) whose mapAsync marks the
      // command buffer complete; full-size pixel buffer only in bytes mode.
      const gatePool = new GpuBufferPool(
        device,
        256,
        GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      );
      const bpr = (width * 4 + 255) & ~255;
      const pixelPool =
        captureMode === "bytes"
          ? new GpuBufferPool(
              device,
              bpr * height,
              GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            )
          : null;

      // ---- Pairing + motion ----
      const strategy = createPairStrategy(direction, {
        observe: () => pendingSample ?? { dx: 0, reliable: false },
        tracker: new DirectionTracker(direction === "auto-reverse"),
      });
      const motion = strategy.auto
        ? new GpuMotion(device, { width, height }, crop)
        : null;

      return {
        width,
        height,
        plan,
        audio,
        audioSource,
        output,
        videoSource,
        rotationNote,
        strategy,
        motion,
        captureMode,
        canvas: capture?.canvas ?? null,
        canvasCtx: capture?.ctx ?? null,
        blendPipe,
        blitPipe,
        targetTex,
        blendU,
        blitBind,
        gatePool,
        pixelPool,
        bpr,
        captureColorSpace: colorSpaceInit(videoFrame.colorSpace),
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

    /**
     * Blend one paired pair and push it into the encoder. Runs on
     * `emitChain` (pairs in order). The command buffer holds the blend
     * pass, the canvas blit (canvas mode) and a mirror copy whose
     * `mapAsync` marks GPU completion — only then is the canvas snapshotted
     * into the encoder's VideoFrame.
     */
    async function emitPair(pair) {
      t0Us ??= pair.timestamp;
      const durationUs = pair.duration > 0 ? pair.duration : fallbackDurationUs;
      const outTimestamp = pair.timestamp - t0Us;

      const waitStart = performance.now();
      await withAbort(drainQueue(encoder, "encodeQueueSize", ENCODE_QUEUE_LIMIT));
      stageMs.encodeWait += performance.now() - waitStart;
      checkAbort();

      const blendStart = performance.now();
      const frameEncoder = device.createCommandEncoder({ label: "blend" });
      const leftView = importExternal(frameEncoder, pair.left.frame);
      const rightView = importExternal(frameEncoder, pair.right.frame);
      const blendBind = device.createBindGroup({
        layout: core.blendPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: leftView },
          { binding: 1, resource: rightView },
          { binding: 2, resource: { buffer: core.blendU } },
        ],
      });
      const pass = frameEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: core.targetTex.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      pass.setPipeline(core.blendPipe);
      pass.setBindGroup(0, blendBind);
      pass.draw(3);
      pass.end();

      // Mirror copy makes command-buffer completion observable via
      // `mapAsync`: one row (256 B) on the canvas path, the full frame on
      // the bytes path (where the copy doubles as the encoder input).
      let gate = null;
      let pixels = null;
      if (core.captureMode === "canvas") {
        const canvasPass = frameEncoder.beginRenderPass({
          colorAttachments: [
            {
              view: core.canvasCtx.getCurrentTexture().createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
              loadOp: "clear",
              storeOp: "store",
            },
          ],
        });
        canvasPass.setPipeline(core.blitPipe);
        canvasPass.setBindGroup(0, core.blitBind);
        canvasPass.draw(3);
        canvasPass.end();
        gate = core.gatePool.acquire();
        frameEncoder.copyTextureToBuffer(
          { texture: core.targetTex },
          { buffer: gate, bytesPerRow: 256 },
          { width: 64, height: 1 },
        );
      } else {
        pixels = core.pixelPool.acquire();
        frameEncoder.copyTextureToBuffer(
          { texture: core.targetTex },
          { buffer: pixels, bytesPerRow: core.bpr },
          { width: core.width, height: core.height },
        );
      }
      device.queue.submit([frameEncoder.finish()]);
      stageMs.blend += performance.now() - blendStart;

      const captureStart = performance.now();
      let outFrame;
      if (core.captureMode === "canvas") {
        await withAbort(gate.mapAsync(GPUMapMode.READ));
        gate.unmap();
        core.gatePool.release(gate);
        // The canvas content is settled (its command buffer completed), so
        // the snapshot the constructor takes is exactly this pair's blend.
        outFrame = new VideoFrame(core.canvas, {
          timestamp: outTimestamp,
          duration: durationUs,
        });
      } else {
        await withAbort(pixels.mapAsync(GPUMapMode.READ));
        const data = new Uint8Array(pixels.getMappedRange().slice(0));
        pixels.unmap();
        core.pixelPool.release(pixels);
        outFrame = new VideoFrame(data, {
          format: "BGRA",
          codedWidth: core.width,
          codedHeight: core.height,
          timestamp: outTimestamp,
          duration: durationUs,
          alphaMode: "discard",
          colorSpace: core.captureColorSpace,
        });
      }
      stageMs.capture += performance.now() - captureStart;

      encoder.encode(outFrame, { keyFrame: pairCount % keyframeEvery === 0 });
      outFrame.close();
      pairCount += 1;

      if (pairCount % 10 === 0) reportProgress();
    }

    /** Close decoded frames strictly older than `releasedBefore` (their
     * pairs have all been submitted — the release is chained behind the
     * emits that referenced them). */
    function releaseFramesBefore(releasedBefore) {
      for (const [index, entry] of pendingEntries) {
        if (index >= releasedBefore) break; // insertion == index order
        entry.frame.close();
        pendingEntries.delete(index);
      }
    }

    function enqueueEmit(pair) {
      emitChain = emitChain
        .then(() => emitPair(pair))
        .catch((e) => {
          latch(e);
          try {
            pair.left.frame.close();
            pair.right.frame.close();
          } catch {
            /* already closed */
          }
        });
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
        gpuBlend: true,
        auto: core?.strategy.auto ?? false,
        reversals: reversalsFound.length,
      });
    }

    decoder = new VideoDecoder({
      output(videoFrame) {
        const entry = {
          index: nextFrameIndex++,
          timestamp: videoFrame.timestamp,
          duration: videoFrame.duration ?? 0,
          frame: videoFrame,
        };
        pendingEntries.set(entry.index, entry);
        maxFramesHeld = Math.max(maxFramesHeld, pendingEntries.size);
        frameChain = frameChain
          .then(async () => {
            core ??= await planFromFrame(videoFrame);
            if (core.motion) {
              // Fire-and-forget GPU passes, then wait for this frame's
              // sample: submitted ahead of the chain, so the GPU work
              // overlaps the previous frames' encodes.
              const t = performance.now();
              core.motion.submit(videoFrame);
              pendingSample = await withAbort(core.motion.nextSample());
              stageMs.motion += performance.now() - t;
            }
            const result = core.strategy.push(entry);
            if (result.reversals.length > 0) {
              reversalsFound.push(...result.reversals);
            }
            for (const pair of result.pairs) enqueueEmit(pair);
            if (result.releasedBefore >= 0) {
              emitChain = emitChain.then(() => releaseFramesBefore(result.releasedBefore)).catch(latch);
            }
          })
          .catch((e) => {
            // Planning/observation failure: latch so the main flow rethrows
            // while later frames still get their (useless) turns — the
            // finally block closes every VideoFrame.
            latch(e);
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
    // End-of-stream awaits raced against cancellation (never-settling codec
    // flushes must not strand the caller — see js/video-pipeline.js).
    await withAbort(decoder.flush());
    await withAbort(frameChain);
    if (fatalError) throw fatalError;

    // End of stream: the strategy commits every held-back pair (the last
    // frame never pairs; like the native pipeline, N inputs give N-1 pairs).
    if (core) {
      const result = core.strategy.flush();
      if (result.reversals.length > 0) reversalsFound.push(...result.reversals);
      for (const pair of result.pairs) enqueueEmit(pair);
    }
    await withAbort(emitChain);
    if (fatalError) throw fatalError;
    for (const entry of pendingEntries.values()) {
      try {
        entry.frame.close();
      } catch {
        /* already closed */
      }
    }
    pendingEntries.clear();

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
        blender: "WebGPU (direct)",
        frameCopy: "none (WebGPU)",
        capture: core.captureMode === "canvas" ? "canvas \u2192 VideoFrame" : "copyToBuffer \u2192 VideoFrame",
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
        frameBuffers: maxFramesHeld,
        timings: formatStageMs(stageMs),
      },
      // Machine-readable stage totals for the benchmark page/harness.
      stageMs,
    };
  } finally {
    // Release codec and GPU resources on every exit path (success, error,
    // cancel). close() on an already-closed codec throws; ignore.
    for (const codec of [decoder, encoder]) {
      try {
        codec?.close();
      } catch {
        /* already closed */
      }
    }
    for (const entry of pendingEntries.values()) {
      try {
        entry.frame.close();
      } catch {
        /* already closed */
      }
    }
    pendingEntries.clear();
    core?.motion?.close();
    if (core) {
      core.targetTex.destroy();
      core.blendU.destroy();
      core.gatePool.destroy();
      core.pixelPool?.destroy();
    }
    if (context.ownsDevice !== false) {
      try {
        device.destroy();
      } catch {
        /* already destroyed */
      }
    }
    source.input.dispose();
  }
}
