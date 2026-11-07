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
 * Choose the output encoder: try H.264 (MP4) first, then VP9 and AV1 (WebM);
 * within each codec prefer a hardware encoder. Returns null when WebCodecs
 * rejects every candidate.
 */
async function planEncoder(width, height, frameRate) {
  if (typeof VideoEncoder === "undefined") return null;

  const level = pickAvcLevel(width, height, frameRate)
    .toString(16)
    .padStart(2, "0");
  const candidates = [
    ...["prefer-hardware", "no-preference"].map((accel) => ({
      container: "mp4",
      format: new Mp4OutputFormat({ fastStart: "in-memory" }),
      mimeType: "video/mp4",
      videoSourceCodec: "avc",
      codec: `avc1.6400${level}`,
      extra: { avc: { format: "avc" } },
      accel,
    })),
    ...["prefer-hardware", "no-preference"].flatMap((accel) => [
      {
        container: "webm",
        format: new WebMOutputFormat(),
        mimeType: "video/webm",
        videoSourceCodec: "vp9",
        codec: "vp09.00.10.08",
        extra: {},
        accel,
      },
      {
        container: "webm",
        format: new WebMOutputFormat(),
        mimeType: "video/webm",
        videoSourceCodec: "av1",
        codec: "av01.0.08M.08",
        extra: {},
        accel,
      },
    ]),
  ];

  for (const candidate of candidates) {
    const config = {
      codec: candidate.codec,
      width,
      height,
      framerate: frameRate,
      bitrate: targetBitrate(width, height, frameRate),
      hardwareAcceleration: accelOrNull(candidate.accel),
      latencyMode: "quality",
      ...candidate.extra,
    };
    const support = await VideoEncoder.isConfigSupported(config).catch(
      () => ({ supported: false }),
    );
    if (support.supported) {
      return { ...candidate, encoderConfig: config };
    }
  }
  return null;
}

/** `hardwareAcceleration` is optional; "no-preference" is the default anyway. */
function accelOrNull(accel) {
  return accel === "prefer-hardware" ? "prefer-hardware" : undefined;
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
 */
function colorSpaceInit(colorSpace) {
  if (!colorSpace) return undefined;
  return {
    primaries: colorSpace.primaries,
    transfer: colorSpace.transfer,
    matrix: colorSpace.matrix,
    fullRange: colorSpace.fullRange,
  };
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

/**
 * Convert a video file into an anaglyph video.
 *
 * @param {object} options
 * @param {File | Blob} options.file Source video.
 * @param {"clockwise" | "counter-clockwise" | "auto" | "auto-reverse"}
 *   [options.direction] Eye pairing; `auto` detects the orbit from image
 *   motion (needs `createTracker`).
 * @param {string} [options.anaglyphType] wasm blend flavor name.
 * @param {(width: number, height: number, anaglyphType: string) => Promise<{blend: (left: Uint8Array, right: Uint8Array) => Promise<Uint8Array>, is_gpu: () => boolean}>} options.createBlender
 *   Factory for the wasm blender. Injected so this module stays free of the
 *   wasm import and remains runnable (with a stub blender) outside a browser.
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
    createBlender,
    createTracker,
    onProgress = () => {},
    signal,
  } = options;

  const startedAt = performance.now();
  const checkAbort = () => {
    if (signal?.aborted) {
      const error = new Error("conversion cancelled");
      error.name = "AbortError";
      throw error;
    }
  };

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
  /** 0-based index of the next decoded frame (the tracker's frame index). */
  let nextFrameIndex = 0;

  try {
    // Prefer a hardware decoder, but fall back to letting the browser pick
    // (software decoders, or hardware without the hint) when the hint makes
    // the configuration unsupported.
    let decoderConfig = { ...source.decoderConfig, hardwareAcceleration: "prefer-hardware" };
    let decoderMode = "hardware-preferred";
    let decoderSupport = await VideoDecoder.isConfigSupported(decoderConfig).catch(
      () => ({ supported: false }),
    );
    if (!decoderSupport.supported) {
      decoderConfig = { ...source.decoderConfig };
      decoderMode = "browser-chosen";
      decoderSupport = await VideoDecoder.isConfigSupported(decoderConfig).catch(
        () => ({ supported: false }),
      );
    }
    if (!decoderSupport.supported) {
      throw new PipelineError(`this browser cannot decode "${source.codecString}"`);
    }

    // H.264 needs even frame dimensions; blend and encode on the cropped
    // frame. Rotation/flip are container metadata and pass through untouched.
    const width = source.codedWidth & ~1;
    const height = source.codedHeight & ~1;
    if (width < 2 || height < 2) {
      throw new PipelineError("video frames are too small to convert");
    }

    const plan = await planEncoder(width, height, source.frameRate);
    if (!plan) {
      throw new PipelineError("no supported video encoder found (tried H.264, VP9, AV1)");
    }
    const audio = planAudioCopy(source.audioCodec, plan.container);

    const blender = await createBlender(width, height, anaglyphType);
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

    // ---- Output: muxer with compressed packet sources ----
    const output = new Output({ format: plan.format, target: new BufferTarget() });
    const videoSource = new EncodedVideoPacketSource(plan.videoSourceCodec);
    // No frameRate metadata: the packet timestamps carry the true cadence,
    // and a rounded fps here would pick a coarse timescale that collides
    // (duplicate DTS) for rates like 33.33.
    output.addVideoTrack(videoSource, {
      rotation: source.rotation,
      flip: source.flip,
    });
    /** @type {EncodedAudioPacketSource | null} */
    let audioSource = null;
    if (audio.copy) {
      audioSource = new EncodedAudioPacketSource(audio.copy);
      output.addAudioTrack(audioSource);
    }
    await output.start();

    // Stages WebCodecs exposes only through callbacks are serialized onto
    // promise chains: frame processing (decoder output) and muxing (encoder
    // output). A callback error is latched and rethrown by the main flow.
    /** @type {Error | null} */
    let fatalError = null;
    let frameChain = Promise.resolve();
    let muxChain = Promise.resolve();
    let videoChunks = 0;

    const keyframeEvery = Math.max(
      1,
      Math.round(source.frameRate * KEYFRAME_INTERVAL_S),
    );
    const fallbackDurationUs = Math.round(1e6 / source.frameRate);
    let pairCount = 0;
    /** @type {number | null} */
    let t0Us = null;

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
        muxChain = muxChain.then(() =>
          videoSource.add(packet, isFirst ? meta : undefined),
        );
      },
      error(e) {
        fatalError ??= e instanceof Error ? e : new Error(String(e));
      },
    });
    encoder.configure(plan.encoderConfig);

    const sourceRect =
      width === source.codedWidth && height === source.codedHeight
        ? undefined
        : { x: 0, y: 0, width, height };

    /** Blend an already-paired pair (owned RGBA buffers) and encode it. */
    async function emitPair(pair) {
      t0Us ??= pair.timestamp;
      const blended = await blender.blend(
        new Uint8Array(pair.left.rgba),
        new Uint8Array(pair.right.rgba),
      );

      await drainQueue(encoder, "encodeQueueSize", ENCODE_QUEUE_LIMIT);
      checkAbort();
      const outFrame = new VideoFrame(blended, {
        format: "RGBA",
        codedWidth: width,
        codedHeight: height,
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
        gpuBlend: blender.is_gpu(),
        auto: strategy.auto,
        reversals: reversalsFound.length,
      });
    }

    decoder = new VideoDecoder({
      output(videoFrame) {
        // Every frame is copied to an owned RGBA buffer immediately and the
        // VideoFrame closed: held-back (auto) frames would otherwise pin
        // decoder-owned memory, and motion analysis + blending both need
        // dense pixels. All processing runs serialized on frameChain, so a
        // frame's buffer always outlives the pairs that reference it.
        const entry = {
          index: nextFrameIndex++,
          timestamp: videoFrame.timestamp,
          duration: videoFrame.duration ?? 0,
          colorSpace: colorSpaceInit(videoFrame.colorSpace),
          rgba: null,
        };
        frameChain = frameChain.then(async () => {
          entry.rgba = pool.acquire();
          await videoFrame.copyTo(new Uint8Array(entry.rgba), {
            format: "RGBA",
            rect: sourceRect,
          });
          videoFrame.close();
          pendingEntries.set(entry.index, entry);
          await handleStrategyResult(strategy.push(entry));
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
      await drainQueue(decoder, "decodeQueueSize", DECODE_QUEUE_LIMIT);
    }
    await decoder.flush();
    await frameChain;
    if (fatalError) throw fatalError;

    // End of stream: the strategy commits every held-back pair (the last
    // frame never pairs; like the native pipeline, N inputs give N-1 pairs).
    await handleStrategyResult(strategy.flush());

    if (pairCount === 0) {
      throw new PipelineError(
        "video has fewer than two frames; anaglyph conversion needs at least one frame pair",
      );
    }

    await encoder.flush();
    await muxChain;
    if (fatalError) throw fatalError;
    reportProgress();

    // ---- Audio: stream-copy compressed packets onto the video timeline ----
    let audioNote = audio.reason;
    if (audioSource) {
      const audioSink = new EncodedPacketSink(source.audioTrack);
      const decoderConfig = await source.audioTrack.getDecoderConfig();
      const meta = decoderConfig ? { decoderConfig } : undefined;
      const shiftS = (t0Us ?? 0) / 1e6;
      let firstAudio = true;
      for await (const packet of audioSink.packets()) {
        checkAbort();
        const shifted =
          shiftS === 0 ? packet : packet.clone({ timestamp: packet.timestamp - shiftS });
        await audioSource.add(shifted, firstAudio ? meta : undefined);
        firstAudio = false;
      }
      audioNote = `${audio.copy} (stream copy)`;
    }

    await output.finalize();
    const blob = new Blob([output.target.buffer], { type: plan.mimeType });

    const elapsed = (performance.now() - startedAt) / 1e3;
    return {
      blob,
      report: {
        frames: pairCount,
        seconds: elapsed,
        fps: elapsed > 0 ? pairCount / elapsed : null,
        container: plan.container,
        encoder: `${plan.codec} (${
          plan.accel === "prefer-hardware" ? "hardware-preferred" : "browser-chosen"
        })`,
        decoder: `${source.codecString} (${decoderMode})`,
        blender: blender.is_gpu() ? "WebGPU" : "CPU (fallback)",
        direction: strategy.auto
          ? direction === "auto-reverse"
            ? "auto (reversed)"
            : "auto"
          : direction,
        reversals: strategy.auto ? formatReversals(reversalsFound) : "n/a",
        audio: audioNote ?? "none",
        width: source.codedWidth,
        height: source.codedHeight,
        frameBuffers: pool.maxHeld,
      },
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
