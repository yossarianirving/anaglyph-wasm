/**
 * WebGPU motion estimation for `auto` pairing.
 *
 * Per frame three GPU passes replace the wasm block matcher's CPU copy +
 * match entirely:
 *
 * 1. `LUMA_SHADER` fragment pass: imported VideoFrame -> full-size
 *    R32Float luma texture (x255, BT.601 weights). This is the only pass
 *    that can read `texture_external`, and it runs as part of the frame's
 *    normal submission — no CPU copy.
 * 2. `BOX_DOWNSAMPLE` compute: exact box average into the 256-wide
 *    analysis texture (same integer ranges as motion.rs), double-buffered.
 * 3. `makeSadCompute` compute: per-block best subpixel shift / quality
 *    against the previous frame's analysis texture; only the tiny
 *    gridW*gridH*4-float result buffer is read back (~8 KB).
 *
 * `aggregateBlocks` reproduces `match_frames`' CPU-side statistics
 * exactly (flat + saturated blocks pre-rejected by the kernel, quality
 * ratio, upper median, /scale), and `pairing.js` receives `{dx,
 * reliable}` through its synchronous `observe` contract — the pipeline
 * awaits the readback before pushing the frame into the strategy.
 *
 * Submissions are fire-and-forget on the decode-output callback (one
 * frame of GPU lead over the CPU chain); `nextSample()` drains the
 * matching readback FIFO.
 */

import { LUMA_SHADER, BOX_DOWNSAMPLE, makeSadCompute } from "./shaders.js";

/** Mirror of the motion.rs constants. */
export const TARGET_WIDTH = 256;
export const BLOCK = 8;
export const MAX_SHIFT = 12;
export const MIN_BLOCK_VARIANCE = 6.0;
export const CUT_QUALITY_LIMIT = 0.45;
export const MIN_BLOCK_RATIO = 0.2;

/**
 * Analysis geometry for a `width` x `height` (visible-crop) frame:
 * analysis size, scale back to source pixels, block grid, and the integer
 * box ranges (`ranges()` in motion.rs). Pure — unit-tested in node.
 *
 * @param {number} width
 * @param {number} height
 */
export function analysisGeometry(width, height) {
  const downscale = width > TARGET_WIDTH;
  const scale = downscale ? TARGET_WIDTH / width : 1.0;
  const w = downscale ? TARGET_WIDTH : width;
  const h = downscale ? Math.max(1, Math.trunc(height * scale)) : height;
  /** @param {number} small @param {number} big */
  const ranges = (small, big) => {
    const starts = new Int32Array(small);
    const ends = new Int32Array(small);
    for (let i = 0; i < small; i++) {
      const start = Math.floor((i * big) / small);
      starts[i] = start;
      ends[i] = Math.max(start + 1, Math.floor(((i + 1) * big) / small));
    }
    return { starts, ends };
  };
  const xr = ranges(w, width);
  const yr = ranges(h, height);
  // match_frames guard: needs room for one block plus both search margins.
  const matchable = w >= 2 * MAX_SHIFT + BLOCK && h >= BLOCK;
  const gridW = matchable ? Math.floor((w - 2 * MAX_SHIFT) / BLOCK) : 0;
  const gridH = matchable ? Math.floor(h / BLOCK) : 0;
  return {
    width,
    height,
    w,
    h,
    scale,
    gridW,
    gridH,
    xStarts: xr.starts,
    xEnds: xr.ends,
    yStarts: yr.starts,
    yEnds: yr.ends,
  };
}

/**
 * Reproduce `match_frames`' reduction over the kernel's per-block output
 * (`[shift, meanSad, stdDev, qualified]` x gridW*gridH): quality gate,
 * then the upper median shift scaled back to source pixels.
 *
 * @param {Float32Array} blocks
 * @param {ReturnType<typeof analysisGeometry>} geo
 * @returns {{dx: number, reliable: boolean}}
 */
export function aggregateBlocks(blocks, geo) {
  const unreliable = { dx: 0, reliable: false };
  if (geo.gridW === 0 || geo.gridH === 0) return unreliable;
  const shifts = [];
  let sadSum = 0;
  let devSum = 0;
  for (let i = 0, b = 0; i < blocks.length; i += 4, b++) {
    if (blocks[i + 3] === 1) {
      shifts.push(blocks[i]);
      sadSum += blocks[i + 1];
      devSum += blocks[i + 2];
    }
  }
  const qualified = shifts.length;
  if (qualified === 0 || qualified < geo.gridW * geo.gridH * MIN_BLOCK_RATIO) {
    return unreliable;
  }
  const quality = (sadSum / qualified) / (devSum / qualified + 1.0);
  if (quality > CUT_QUALITY_LIMIT) return unreliable;
  shifts.sort((a, b) => a - b);
  return { dx: shifts[qualified >> 1] / geo.scale, reliable: true };
}

/** Result of one motion pass, or the "no predecessor" marker. */
const NO_PREV = { dx: 0, reliable: false };

/**
 * GPU motion pipeline for one stream. All textures/buffers/pipelines are
 * created once for the plan's crop; `submit()` runs the three passes for
 * one decoded frame and `nextSample()` resolves its drift sample.
 */
export class GpuMotion {
  /**
   * @param {GPUDevice} device
   * @param {{width: number, height: number}} visible frame size the
   *   analysis runs on (luma texture size; VideoFrames may be larger —
   *   the luma pass crops to `crop`).
   * @param {{x0: number, y0: number, w: number, h: number}} crop
   *   visible-rect crop in coded-UV units (same uniform as the blend pass).
   */
  constructor(device, { width, height }, crop) {
    this.device = device;
    this.crop = crop;
    this.geo = analysisGeometry(width, height);
    const { w, h, gridW, gridH } = this.geo;

    // Workgroup strips for both frames must fit device storage memory.
    const memLimit = device.limits.maxComputeWorkgroupStorageSize || 16384;
    const tileW = Math.max(
      1,
      Math.min(64, Math.floor((Math.floor(memLimit / 64) - 2 * MAX_SHIFT) / BLOCK)),
    );
    this.tileW = tileW;

    const f32Bytes = gridW * gridH * 4 * 4;
    this._blockBytes = f32Bytes;

    /** Full-size luma (fragment target) + double-buffered analysis. */
    const lumaFlags =
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const smallFlags =
      GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    const f32Tex = { format: "r32float", usage: lumaFlags, dimension: "2d" };
    /** @type {GPUTexture} */
    this.fullTex = device.createTexture({
      ...f32Tex,
      width,
      height,
      sampleCount: 1,
    });
    /** @type {GPUTexture[]} */
    this.smallTex = [0, 1].map((i) =>
      device.createTexture({
        format: "r32float",
        usage: smallFlags,
        width: Math.max(1, w),
        height: Math.max(1, h),
        sampleCount: 1,
      }),
    );

    // Uniforms (written once; crop is fixed at plan time).
    this.lumaU = this._uniform(new Float32Array([crop.x0, crop.y0, crop.w, crop.h]));
    this.boxU = this._uniform(
      new Uint32Array([width, height, Math.max(1, w), Math.max(1, h)]),
    );
    // SadU mixes u32 and f32 words in one 20-byte uniform struct.
    const sadWords = new ArrayBuffer(24);
    new Uint32Array(sadWords, 0, 4).set([w, h, gridW, gridH]);
    new Float32Array(sadWords, 16, 1)[0] = MIN_BLOCK_VARIANCE;
    this.sadU = this._uniform(sadWords);

    // Box ranges table: [x0, x1] * w then [y0, y1] * h.
    const ranges = new Int32Array((w + h) * 2);
    for (let i = 0; i < w; i++) {
      ranges[i * 2] = this.geo.xStarts[i];
      ranges[i * 2 + 1] = this.geo.xEnds[i];
    }
    for (let i = 0; i < h; i++) {
      ranges[w * 2 + i * 2] = this.geo.yStarts[i];
      ranges[w * 2 + i * 2 + 1] = this.geo.yEnds[i];
    }
    this.rangesBuf = device.createBuffer({
      size: ranges.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.rangesBuf, 0, ranges);

    /** Per-frame SAD output (single buffer; queue serialization orders
     * each frame's copy-to-readback ahead of the next frame's writes). */
    this.blockBuf =
      f32Bytes > 0
        ? device.createBuffer({
            size: f32Bytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          })
        : null;

    // Pipelines.
    const lumaMod = device.createShaderModule({ code: LUMA_SHADER });
    const boxMod = device.createShaderModule({ code: BOX_DOWNSAMPLE });
    const sadMod = device.createShaderModule({ code: makeSadCompute(tileW, MIN_BLOCK_VARIANCE) });
    this.lumaPipe = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: lumaMod, entryPoint: "vs_main" },
      fragment: {
        module: lumaMod,
        entryPoint: "fs_main",
        targets: [{ format: "r32float" }],
      },
      primitive: { topology: "triangle-list" },
    });
    this.boxPipe = device.createComputePipeline({
      layout: "auto",
      compute: { module: boxMod, entryPoint: "main" },
    });
    this.sadPipe =
      f32Bytes > 0
        ? device.createComputePipeline({
            layout: "auto",
            compute: { module: sadMod, entryPoint: "main" },
          })
        : null;

    // Shared compute bind groups (box is fully static; sad's prev/cur swap
    // alternates between two groups).
    this.boxBind = device.createBindGroup({
      layout: this.boxPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.fullTex.createView() },
        { binding: 1, resource: this.smallTex[0].createView() },
        { binding: 2, resource: { buffer: this.rangesBuf } },
        { binding: 3, resource: { buffer: this.boxU } },
      ],
    });
    this.boxBindAlt = device.createBindGroup({
      layout: this.boxPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.fullTex.createView() },
        { binding: 1, resource: this.smallTex[1].createView() },
        { binding: 2, resource: { buffer: this.rangesBuf } },
        { binding: 3, resource: { buffer: this.boxU } },
      ],
    });
    this.sadBinds = this.sadPipe
      ? [0, 1].map((prevIdx) =>
          device.createBindGroup({
            layout: this.sadPipe.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: this.smallTex[prevIdx].createView() },
              { binding: 1, resource: this.smallTex[1 - prevIdx].createView() },
              { binding: 2, resource: { buffer: this.blockBuf } },
              { binding: 3, resource: { buffer: this.sadU } },
            ],
          }),
        )
      : null;

    // Readback buffer pool (MAP_READ buffers can't be reused while
    // mapped; one per outstanding frame, bounded by the pending cap).
    /** @type {GPUBuffer[]} */
    this._pool = [];
    /** @type {Array<{buffer: GPUBuffer | null, immediate: boolean}>} */
    this._pending = [];
    this._havePrev = false;
    this._curIdx = 0;
    this._closed = false;
    this._importIsDict = undefined;
  }

  _uniform(data) {
    const buf = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  _takeBuffer() {
    let buf = this._pool.pop();
    if (!buf || buf.mapState !== "unmapped") {
      buf = this.device.createBuffer({
        size: Math.max(this._blockBytes, 16),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
    return buf;
  }

  /** Import a VideoFrame, coping with the dict-vs-positional API skew. */
  _import(device, encoder, frame) {
    const tryDict = () =>
      encoder.importExternalTexture({ externalTexture: frame });
    const tryPos = () => encoder.importExternalTexture(frame);
    if (this._importIsDict === true) return tryDict();
    if (this._importIsDict === false) return tryPos();
    try {
      const view = tryDict();
      this._importIsDict = true;
      return view;
    } catch {
      const view = tryPos();
      this._importIsDict = false;
      return view;
    }
  }

  /**
   * Enqueue this frame's motion passes. Call from the decode-output
   * callback (the VideoFrame must still be open). Fire-and-forget: the
   * matching sample comes out of `nextSample()` in the same order.
   *
   * @param {VideoFrame} frame
   */
  submit(frame) {
    if (this._closed) return;
    const device = this.device;
    const encoder = device.createCommandEncoder({ label: "motion" });
    const view = this._import(device, encoder, frame);

    // 1. Luma -> full-size r32float.
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.fullTex.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.lumaPipe);
    pass.setBindGroup(0, device.createBindGroup({
      layout: this.lumaPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: view },
        { binding: 1, resource: { buffer: this.lumaU } },
      ],
    }));
    pass.draw(3);
    pass.end();

    // 2. Box downsample into the analysis texture. `_curIdx` is the
    // texture holding the previous frame's luma; this frame writes the
    // other one.
    const writingIdx = this._havePrev ? 1 - this._curIdx : 0;
    const cp = encoder.beginComputePass();
    cp.setPipeline(this.boxPipe);
    cp.setBindGroup(
      0,
      writingIdx === 0 ? this.boxBind : this.boxBindAlt,
    );
    cp.dispatchWorkgroups(
      Math.max(1, Math.ceil(this.geo.w / 8)),
      Math.max(1, Math.ceil(this.geo.h / 8)),
    );

    // 3. SAD against the stored predecessor + readback.
    if (this._havePrev && this.sadPipe) {
      const readBuf = this._takeBuffer();
      cp.setPipeline(this.sadPipe);
      cp.setBindGroup(0, this.sadBinds[this._curIdx]);
      cp.dispatchWorkgroups(
        Math.ceil(this.geo.gridW / this.tileW),
        this.geo.gridH,
      );
      cp.end();
      encoder.copyBufferToBuffer(this.blockBuf, 0, readBuf, 0, this._blockBytes);
      device.queue.submit([encoder.finish()]);
      this._pending.push({ buffer: readBuf, immediate: false });
    } else {
      cp.end();
      device.queue.submit([encoder.finish()]);
      this._pending.push({ buffer: null, immediate: true });
    }
    // The texture written this frame is next frame's "prev".
    this._curIdx = writingIdx;
    this._havePrev = true;
  }

  /**
   * Resolve the next frame's drift sample (FIFO with `submit`).
   * @returns {Promise<{dx: number, reliable: boolean}>}
   */
  async nextSample() {
    const item = this._pending.shift();
    if (!item) throw new Error("GpuMotion.nextSample: no pending frame");
    if (item.immediate) return { ...NO_PREV };
    const { buffer } = item;
    await buffer.mapAsync(GPUMapMode.READ);
    let sample;
    if (this._closed) {
      buffer.destroy();
      sample = { ...NO_PREV };
    } else {
      const blocks = new Float32Array(buffer.getMappedRange().slice(0), 0, this._blockBytes / 4);
      buffer.unmap();
      this._pool.push(buffer);
      sample = aggregateBlocks(blocks, this.geo);
    }
    return sample;
  }

  /** Drop pending readbacks (cancel path): unmap what mapped, destroy. */
  close() {
    this._closed = true;
    for (const item of this._pending) {
      if (item.buffer) item.buffer.destroy();
    }
    this._pending = [];
    for (const buf of this._pool) buf.destroy();
    this._pool = [];
    this.fullTex.destroy();
    for (const t of this.smallTex) t.destroy();
    for (const b of [this.lumaU, this.boxU, this.sadU, this.rangesBuf, this.blockBuf]) {
      if (b) b.destroy();
    }
  }
}
