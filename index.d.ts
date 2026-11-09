/**
 * Type surface for the anaglyph-wasm public entry (index.js).
 */

export type Engine = "auto" | "webgpu" | "wasm";
export type ResolvedEngine = "webgpu" | "wasm";
export type Direction =
  | "clockwise"
  | "counter-clockwise"
  | "auto"
  | "auto-reverse";
export type AnaglyphType =
  | "color"
  | "half-color"
  | "grayscale"
  | "optimized"
  | "true";
export type EncoderQuality = "standard" | "high" | "maximum";

/** Which pipeline runs what (hw decode/encode default on; `webgpu` opts the
 * wasm pipeline into GPU blending — usually slower, off by default). */
export interface Acceleration {
  decode?: boolean;
  encode?: boolean;
  webgpu?: boolean;
}

/** Per-frame progress emitted through onProgress / video-progress. */
export interface ProgressStatus {
  frames: number;
  totalFrames?: number;
  fps: number;
  /** wasm pipeline: whether the GPU blender is active for this run. */
  gpuBlend?: boolean;
  /** auto direction mode: detected reversals so far. */
  auto?: boolean;
  reversals?: number;
  [key: string]: unknown;
}

/** Conversion report (keys vary slightly per pipeline; e.g. `container`,
 * `codec`, `encoder`, `blender`, `direction`, `fps`, `seconds`). */
export interface ConversionReport {
  container: string;
  [key: string]: unknown;
}

export interface Support {
  webcodecs: boolean;
  h264Encode: boolean;
  webgpu: boolean;
  /** What convertVideo's engine "auto" picks in this browser. */
  engine: ResolvedEngine;
}

export interface EngineNote {
  /** Message of the webgpu failure that triggered the wasm fallback. */
  fallback?: string;
}

/** Rejects with `Error` whose `name === "AbortError"` on cancel;
 * `.workerReset === true` when a stuck worker had to be force-reset. */
export interface ConvertVideoOptions {
  /** Stereo source video (MP4/H.264 works best). */
  file: File | Blob;
  direction?: Direction;
  anaglyphType?: AnaglyphType;
  /** Default "auto": webgpu when an adapter answers, else wasm. */
  engine?: Engine;
  acceleration?: Acceleration;
  /** webgpu pipeline only: zero-copy canvas encoder capture. */
  canvasCapture?: boolean;
  encoderQuality?: EncoderQuality;
  /** Override for fetching anaglyph_wasm_bg.wasm (bundler escape hatch). */
  wasmUrl?: string | URL | null;
  onProgress?: (status: ProgressStatus) => void;
  onEngine?: (engine: ResolvedEngine, note?: EngineNote) => void;
  signal?: AbortSignal;
}

export interface ConvertVideoResult {
  blob: Blob;
  report: ConversionReport;
  /** The pipeline that actually ran. */
  engine: ResolvedEngine;
  /** Set when a webgpu run failed for adapter reasons and wasm re-ran it. */
  fellBack?: boolean;
}

/** Job for the worker-based converter (no signal — use cancel()). */
export interface ConvertJob {
  file: File | Blob;
  direction?: Direction;
  anaglyphType?: AnaglyphType;
  acceleration?: Acceleration;
  canvasCapture?: boolean;
  encoderQuality?: EncoderQuality;
  wasmUrl?: string | URL | null;
  onProgress?: (status: ProgressStatus) => void;
}

export interface ConvertResult extends ConvertVideoResult {
  /** Object URL of the result (created inside the worker). */
  url?: string;
}

export interface VideoConverter {
  /** Engine of the live worker (null before the first convert). */
  readonly engine: ResolvedEngine | null;
  convert(
    job: ConvertJob,
    overrides?: { engine?: Engine },
  ): Promise<ConvertResult>;
  cancel(): void;
  dispose(): void;
}

/** Probe the browser once (cached). */
export function checkSupport(force?: boolean): Promise<Support>;

/** Run one conversion on the calling thread. */
export function convertVideo(
  options: ConvertVideoOptions,
): Promise<ConvertVideoResult>;

export interface CreateVideoConverterOptions {
  /** Default per-run engine (resolved "auto" per browser at convert time). */
  engine?: Engine;
  /** Override the worker script URLs (defaults resolve next to index.js). */
  workerUrls?: { wasm?: string; webgpu?: string };
  onEngine?: (engine: ResolvedEngine, note?: EngineNote) => void;
}

/** Worker-based converter: same promise semantics as convertVideo, plus
 * cooperative cancel with a force-reset watchdog. */
export function createVideoConverter(
  options?: CreateVideoConverterOptions,
): VideoConverter;
