/**
 * Shared WebCodecs encoder planning for both video pipelines
 * (`js/video-pipeline.js` wasm blend, `gpu-video/gpu-pipeline.js` all-GPU).
 * Both feed the same `VideoEncoder` through mediabunny, so codec probing,
 * hardware-acceleration preference, bitrate sizing, and the output-quality
 * ladder live here once.
 *
 * ## Output quality options
 *
 * WebCodecs exposes no QP/CRF surface for file encoding — the portable
 * quality lever is the target bitrate (`bitrateMode` stays "variable",
 * which every engine supports and which suits file output). The
 * `encoderQuality` tiers scale the bitrate on a bits-per-pixel-per-frame
 * ladder with per-tier clamps:
 *
 * | tier       | bpp  | clamp          | 480×270@30 | 1080p@30   |
 * | ---------- | ---- | -------------- | ---------- | ---------- |
 * | standard   | 0.10 | 1.5–12 Mbps    |  1.5 Mbps  | 12 Mbps    |
 * | high       | 0.25 | 3–24 Mbps      |  3 Mbps    | 15.5 Mbps  |
 * | maximum    | 0.50 | 6–48 Mbps      |  6 Mbps    | 31 Mbps    |
 *
 * `standard` reproduces the original sizing (0.1 bpp clamped 1.5–12 Mbps);
 * already-compressed, blocky sources benefit from `high`/`maximum` because
 * the encoder stops re-quantizing the source blocks harder than the source
 * itself. Hardware encoders (which have less rate-distortion cleverness
 * than x264/libvpx at equal bitrate) benefit the most.
 *
 * When an encoder rejects a high-bitrate config (hardware level caps,
 * vendor limits), `planEncoder` degrades the tier — maximum → high →
 * standard — before falling back to another codec candidate, so quality
 * options never turn an exotic device into a hard failure.
 */

import { Mp4OutputFormat, WebMOutputFormat } from "../vendor/mediabunny.mjs";

/** Quality tiers for `planEncoder(…, quality)`. `bpp` is bits per pixel
 * per frame; `min`/`max` bound the resulting target in bits/s. */
export const QUALITY_LEVELS = {
  standard: { bpp: 0.1, min: 1_500_000, max: 12_000_000 },
  high: { bpp: 0.25, min: 3_000_000, max: 24_000_000 },
  maximum: { bpp: 0.5, min: 6_000_000, max: 48_000_000 },
};

/** Ordered high → low for the degrade-on-reject ladder. */
const QUALITY_ORDER = ["maximum", "high", "standard"];

/**
 * Target bitrate for a frame size/rate at one quality tier.
 *
 * @param {number} width
 * @param {number} height
 * @param {number} frameRate
 * @param {keyof typeof QUALITY_LEVELS} [quality]
 * @returns {number} bits/s
 */
export function targetBitrate(width, height, frameRate, quality = "standard") {
  const level = QUALITY_LEVELS[quality];
  if (!level) {
    throw new Error(
      `unknown encoder quality "${quality}" (expected one of ${Object.keys(QUALITY_LEVELS).join(", ")})`,
    );
  }
  return Math.round(
    Math.min(level.max, Math.max(level.min, width * height * frameRate * level.bpp)),
  );
}

/**
 * Pick an H.264 High-profile level code that covers the frame size and rate.
 * Deliberately conservative: an over-generous level is always valid.
 */
export function pickAvcLevel(width, height, frameRate) {
  const pixels = width * height;
  if (pixels <= 1280 * 720) return 0x29; // 4.1
  if (pixels <= 1920 * 1080) return frameRate > 30 ? 0x2a : 0x28; // 4.2 / 4.0
  if (pixels <= 2560 * 1440) return 0x33; // 5.1
  return 0x34; // 5.2
}

/** `hardwareAcceleration` is optional; "no-preference" is the default anyway. */
export function accelOrNull(accel) {
  return accel === "no-preference" ? undefined : accel;
}

/** Human label for a codec's chosen `hardwareAcceleration` preference. */
export function accelLabel(accel) {
  return accel === "prefer-hardware"
    ? "hardware-preferred"
    : accel === "prefer-software"
      ? "software-forced"
      : "browser-chosen";
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
 * `quality` sizes the bitrate (see the module header). When every candidate
 * is rejected at the requested tier, the whole codec/accel ladder retries
 * one tier down before giving up; the returned plan carries the achieved
 * tier in `plan.quality`.
 *
 * Exported for the benchmark pages' encoder-only upper-bound probe.
 *
 * @param {number} width
 * @param {number} height
 * @param {number} frameRate
 * @param {boolean} hwEncode
 * @param {"quality" | "realtime"} latencyMode
 * @param {string | null} [codecHint]
 * @param {keyof typeof QUALITY_LEVELS} [quality]
 */
export async function planEncoder(
  width,
  height,
  frameRate,
  hwEncode,
  latencyMode,
  codecHint,
  quality = "standard",
) {
  if (typeof VideoEncoder === "undefined") return null;
  if (!QUALITY_LEVELS[quality]) {
    throw new Error(
      `unknown encoder quality "${quality}" (expected one of ${Object.keys(QUALITY_LEVELS).join(", ")})`,
    );
  }

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

  // Requested tier first, then degrade maximum → high → standard so an
  // encoder that rejects the high-bitrate config still produces a video.
  const start = QUALITY_ORDER.indexOf(quality);
  const tiers = start === -1 ? QUALITY_ORDER : QUALITY_ORDER.slice(start);
  tiers.push("standard"); // quality === "standard" already; harmless dup
  for (const tier of [...new Set(tiers)]) {
    for (const candidate of codecs.flatMap((c) =>
      accels.map((accel) => ({ ...c, accel })),
    )) {
      const config = {
        codec: candidate.codec,
        width,
        height,
        framerate: frameRate,
        bitrate: targetBitrate(width, height, frameRate, tier),
        hardwareAcceleration: accelOrNull(candidate.accel),
        latencyMode,
        ...candidate.extra,
      };
      const support = await VideoEncoder.isConfigSupported(config).catch(
        () => ({ supported: false }),
      );
      if (support.supported) {
        return { ...candidate, tier, format: candidate.format(), encoderConfig: config };
      }
    }
  }
  return null;
}

/**
 * Report suffix describing the achieved quality tier, e.g. `quality=high`
 * — or `quality=maximum→standard (encoder cap)` when the requested tier
 * was rejected and the planner degraded.
 *
 * @param {string} requested
 * @param {string} achieved  plan.tier
 */
export function qualityLabel(requested, achieved) {
  return requested === achieved
    ? `quality=${requested}`
    : `quality=${requested}\u2192${achieved} (encoder cap)`;
}
