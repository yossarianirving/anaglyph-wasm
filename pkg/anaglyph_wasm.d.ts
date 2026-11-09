/* tslint:disable */
/* eslint-disable */

export class Anaglyph {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    get_height(): number;
    get_width(): number;
    /**
     * Convert the stored gif into an anaglyph gif, returned as raw gif
     * bytes.
     *
     * With the `gpu` feature enabled, frame blending runs on the GPU via
     * WebGPU and falls back to a CPU loop when no usable adapter is
     * available (the reason is logged to the console); without it, blending
     * stays on the CPU. The method is async so the JS API is stable across
     * both builds: callers must `await` the returned promise.
     */
    gif_to_anaglyph_async(anaglyph_type?: string | null, video_direction?: string | null): Promise<Uint8Array>;
    /**
     * Same conversion as [`Self::gif_to_anaglyph_async`] but always on the
     * CPU and fully synchronous (blocks the event loop until done). Used by
     * the benchmark page to compare CPU vs WebGPU from the same build.
     */
    gif_to_anaglyph_cpu(anaglyph_type?: string | null, video_direction?: string | null): Uint8Array;
    static new(): Anaglyph;
    set_gif(gif: Uint8Array | number[]): void;
    set_left_image(width: number, height: number, left_image: Uint8Array | number[]): boolean;
    set_left_image_raw(left_image: Uint8Array | number[]): void;
    set_right_image(width: number, height: number, right_image: Uint8Array | number[]): boolean;
    set_right_image_raw(right_image: Uint8Array | number[]): void;
    set_stereoscopic_image(width: number, height: number, stereo_image: Uint8Array | number[]): boolean;
    to_anaglyph(anaglyph_type: string | null | undefined, offset_x: number, offset_y: number): AnaglyphResult;
}

export class AnaglyphResult {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    get_image(): Uint8Array;
    height: number;
    width: number;
}

/**
 * Per-frame camera drift estimator for auto-direction pairing. Wraps the
 * native `MotionEstimator` and consumes the same dense RGBA frames the
 * pipeline already produces for [`VideoBlender`]: the estimator downsamples
 * RGBA directly (identical luma to the RGB24 path, verified in anaglyph-rs
 * tests), so there is no packing pass per frame.
 */
export class MotionAnalyzer {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Whether the last [`Self::observe_rgba`] sample was trustworthy
     * (false: scene cut, motion beyond the search range, textureless input,
     * or the very first frame).
     */
    is_reliable(): boolean;
    /**
     * New estimator for `width` x `height` frames. Every
     * [`Self::observe_rgba`] call must supply exactly `width * height * 4`
     * bytes.
     */
    static new(width: number, height: number): MotionAnalyzer;
    /**
     * Match the next frame (dense RGBA) against its predecessor and return
     * the signed horizontal drift in source pixels per frame, positive when
     * the image content moves right. Check [`Self::is_reliable`] before
     * trusting the value.
     */
    observe_rgba(rgba: Uint8Array | number[]): number;
}

/**
 * Camera orbit direction tracker for `direction: "auto"`: wraps the native
 * [`anaglyph_rs::motion::DirectionTracker`] and exposes the trend-to-eye-
 * pairing mapping, so the JS pairing window gates and labels its output
 * exactly like the native CLI's auto mode.
 */
export class Tracker {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Highest frame index whose pairing can no longer change; pairs whose
     * newer frame is at or below it may be emitted. `-1` while nothing is
     * confirmed yet.
     */
    confirmed_through(): number;
    /**
     * Eye pairing for the frames following `confirmed_through()`, as
     * `"clockwise"` or `"counter-clockwise"` (the un-reversed base pairing
     * while no trend has formed).
     */
    current_direction(): string;
    /**
     * End of stream: everything held back commits to the current trend.
     */
    finish(): void;
    /**
     * `reverse` flips the detected pairing, for rigs whose physical
     * rotation runs opposite to the convention (the CLI's `auto-reverse`).
     */
    constructor(reverse: boolean);
    /**
     * Record the drift sample of frame `index` (0-based decode order) at
     * `time_secs` (source presentation time, seconds).
     */
    push(index: number, time_secs: number, dx: number, reliable: boolean): void;
    /**
     * Drain the reversals confirmed since the last call, packed as f64
     * quads `[frame, time_secs, from_clockwise, to_clockwise]` with the
     * direction slots 1 for clockwise, 0 for counter-clockwise (already
     * mapped through `reverse`).
     */
    take_reversals(): Float64Array;
}

/**
 * Per-frame anaglyph blender for video conversion, backing the WebCodecs
 * pipeline in `js/video-pipeline.js`.
 *
 * The JS side owns decode/encode (hardware `VideoDecoder` / `VideoEncoder`)
 * and the stereo pairing window; this struct only blends one frame pair at a
 * time: dense RGBA in, dense RGBA out, matching the gif blend in
 * `anaglyph-rs`. With the `gpu` feature the blend runs through WebGPU
 * (`GpuBlender::blend_rgba_async`) and falls back to a single-threaded CPU
 * loop when no usable adapter exists; the chosen path is logged and exposed
 * via [`Self::is_gpu`] so the UI can report it.
 */
export class VideoBlender {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Blend one stereo pair of dense RGBA frames (`4 * width * height` bytes
     * each) and return the blended frame. Left/right assignment (which eye
     * sees which decoded frame) is the caller's choice, so auto-direction can
     * swap them per pair.
     */
    blend(left: Uint8Array | number[], right: Uint8Array | number[]): Promise<Uint8Array>;
    /**
     * Whether frames blend on WebGPU (false means the CPU fallback is in
     * use).
     */
    is_gpu(): boolean;
    /**
     * Create a blender for `width` x `height` RGBA frames. Async because the
     * WebGPU adapter request must not block the event loop; it resolves even
     * when no GPU is found (the blender then runs on the CPU).
     *
     * A static factory rather than an `async` constructor: wasm-bindgen
     * deprecates async constructors because they emit invalid TypeScript.
     * Use `await VideoBlender.new(...)`.
     *
     * `backend` optionally forces `"cpu"` (skips the GPU probe entirely);
     * anything else, including omission, prefers WebGPU with a CPU fallback.
     * Benchmarks use this to compare both paths in one build.
     */
    static new(width: number, height: number, anaglyph_type?: string | null, backend?: string | null): Promise<VideoBlender>;
}

/**
 * Zero-copy raw 4:2:0 → RGBA converter for the WebKit frame path.
 *
 * WebKit's `VideoFrame.copyTo` ignores the requested output format and
 * hands back native NV12/I420 planes (described by `[{offset, stride}]`
 * results), and every copyTo costs ~1.4 ms of fixed round-trip latency
 * whatever the destination size (BENCH.md "Engine notes"). To keep the
 * pixel work around that floor minimal, both buffers live in wasm memory:
 * `planes_view()` is the copyTo target (the browser memcpy's planes
 * straight into wasm memory), `convert()` turns them into dense RGBA
 * cropped to the visible region, and `dst_view()` reads the result — no
 * bindgen slice copies in either direction.
 *
 * The returned views alias wasm linear memory. They stay valid only
 * until the next call that can grow memory, so callers must re-fetch
 * `planes_view()` every frame and consume `dst_view()` immediately
 * (before the next `convert`).
 */
export class YuvFrameConverter {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Convert the planes last copied into `planesView()` to dense RGBA
     * (`dstView`). `planar` selects I420 (else NV12); `bt709` selects the
     * BT.709 matrix over BT.601. Returns `false` (dst untouched) when the
     * geometry does not fit the plane bytes or output size.
     */
    convert(y_offset: number, y_stride: number, uv_offset: number, uv_stride: number, v_offset: number, x: number, y: number, width: number, height: number, planar: boolean, bt709: boolean): boolean;
    /**
     * Converted RGBA output (`width * height * 4` bytes). Consume before
     * the next `convert`; the view detaches if memory grows.
     */
    dstView(): Uint8Array;
    /**
     * `plane_bytes` bounds the raw plane buffer (a coded 4:2:0 frame with
     * slack for padding: `codedWidth * codedHeight * 2`); `width` ×
     * `height` is the RGBA output (the even visible crop, fixed for the
     * whole conversion).
     */
    constructor(plane_bytes: number, width: number, height: number);
    /**
     * Scratch buffer for one raw frame: hand this to `copyTo` (no format
     * option). Re-fetch every frame; the view detaches if memory grows.
     */
    planesView(): Uint8Array;
}

/**
 * Whether this build blends CPU-side frames with WebAssembly SIMD (false:
 * scalar fallback, e.g. a `target-cpu=mvp` build).
 */
export function blend_is_simd(): boolean;

/**
 * Blend a dense RGBA pair on the CPU with the matrix for `anaglyph_type`,
 * forcing either the SIMD or the scalar reference loop. Exists so the e2e
 * suite can prove both loops are byte-identical from the browser; regular
 * callers use [`VideoBlender::blend`].
 */
export function cpu_blend_debug(left: Uint8Array | number[], right: Uint8Array | number[], anaglyph_type: string | null | undefined, scalar: boolean): Uint8Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_anaglyph_free: (a: number, b: number) => void;
    readonly __wbg_anaglyphresult_free: (a: number, b: number) => void;
    readonly __wbg_get_anaglyphresult_height: (a: number) => number;
    readonly __wbg_get_anaglyphresult_width: (a: number) => number;
    readonly __wbg_motionanalyzer_free: (a: number, b: number) => void;
    readonly __wbg_set_anaglyphresult_height: (a: number, b: number) => void;
    readonly __wbg_set_anaglyphresult_width: (a: number, b: number) => void;
    readonly __wbg_tracker_free: (a: number, b: number) => void;
    readonly __wbg_videoblender_free: (a: number, b: number) => void;
    readonly __wbg_yuvframeconverter_free: (a: number, b: number) => void;
    readonly anaglyph_get_height: (a: number) => number;
    readonly anaglyph_get_width: (a: number) => number;
    readonly anaglyph_gif_to_anaglyph_async: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly anaglyph_gif_to_anaglyph_cpu: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly anaglyph_new: () => number;
    readonly anaglyph_set_gif: (a: number, b: number, c: number) => void;
    readonly anaglyph_set_left_image: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly anaglyph_set_left_image_raw: (a: number, b: number, c: number) => void;
    readonly anaglyph_set_right_image: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly anaglyph_set_right_image_raw: (a: number, b: number, c: number) => void;
    readonly anaglyph_set_stereoscopic_image: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly anaglyph_to_anaglyph: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly anaglyphresult_get_image: (a: number) => [number, number];
    readonly blend_is_simd: () => number;
    readonly cpu_blend_debug: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
    readonly motionanalyzer_is_reliable: (a: number) => number;
    readonly motionanalyzer_new: (a: number, b: number) => [number, number, number];
    readonly motionanalyzer_observe_rgba: (a: number, b: number, c: number) => [number, number, number];
    readonly tracker_confirmed_through: (a: number) => number;
    readonly tracker_current_direction: (a: number) => [number, number];
    readonly tracker_finish: (a: number) => void;
    readonly tracker_new: (a: number) => number;
    readonly tracker_push: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly tracker_take_reversals: (a: number) => [number, number];
    readonly videoblender_blend: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly videoblender_is_gpu: (a: number) => number;
    readonly videoblender_new: (a: number, b: number, c: number, d: number, e: number, f: number) => any;
    readonly yuvframeconverter_convert: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => number;
    readonly yuvframeconverter_dstView: (a: number) => any;
    readonly yuvframeconverter_new: (a: number, b: number, c: number) => [number, number, number];
    readonly yuvframeconverter_planesView: (a: number) => any;
    readonly wasm_bindgen__convert__closures_____invoke__h11e6b4212b48f600: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h20184457d846f68e: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h43ae07ab71b87fb4: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h43ae07ab71b87fb4_38: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h43ae07ab71b87fb4_39: (a: number, b: number, c: any) => [number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
