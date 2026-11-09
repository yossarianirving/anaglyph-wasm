/**
 * WebGPU motion estimation for `auto` pairing.
 *
 * Per frame three GPU passes replace the wasm block matcher's CPU copy +
 * match entirely:
 *
 * 1. `makeLumaShader` fragment pass: decoded frame -> full-size R32Float
 *    luma texture (x255, BT.601 weights). Source mode `external` imports
 *    the VideoFrame as `texture_external` inside the pass — the only pass
 *    that can, and it runs as part of the frame's normal submission, no
 *    CPU copy. Source mode `texture2d` (Firefox) samples the staged view
 *    `submit()` receives from the pipeline's canvas stager.
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

import { makeLumaShader, BOX_DOWNSAMPLE, makeSadCompute } from "./shaders.js";
import { makeImporter } from "./external.js";

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
   *   analysis runs on (luma texture size; the external texture is the
   *   frame's visible rect, so this matches its dimensions up to the
   *   odd-dimension trim expressed by `crop`).
   * @param {{x0: number, y0: number, w: number, h: number}} crop
   *   source window in image uv units (same uniform as the blend pass).
   * @param {0 | 1} [flip] external-texture y flip from the orientation
   *   calibration (see `probeExternalOrientation`).
   * @param {"external" | "texture2d"} [source] luma-pass frame source
   *   mode; `texture2d` for engines staging frames via
   *   `copyExternalImageToTexture` (submit() then takes a staged view).
   */
  constructor(
    device,
    { width, height },
    crop,
    flip = 0,
    source = "external",
    { earlyArm = false, batchK = 8 } = {},
  ) {
    this.device = device;
    this.crop = crop;
    /** Batching switch: pipelined emit reads motion samples back in
     * batches (see _batchK); the serial emit path (iOS, canvas capture)
     * keeps the per-frame await order that was verified on device. */
    this._earlyArm = earlyArm;
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

    /** Full-size luma (fragment target) + double-buffered analysis.
     * Rendering into r32float needs the float32-renderable feature; fall
     * back to r16float (both sample fine via textureLoad in the compute
     * passes, and luma*255 is exactly representable in either). */
    this.lumaFormat = device.features.has("float32-renderable")
      ? "r32float"
      : "r16float";
    const lumaFlags =
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const smallFlags =
      GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    /** @type {GPUTexture} */
    this.fullTex = device.createTexture({
      format: this.lumaFormat,
      usage: lumaFlags,
      size: [width, height],
      sampleCount: 1,
    });
    /** @type {GPUTexture[]} */
    this.smallTex = [0, 1].map((i) =>
      device.createTexture({
        format: "r32float",
        usage: smallFlags,
        size: [Math.max(1, w), Math.max(1, h)],
        sampleCount: 1,
      }),
    );

    // Uniforms (written once; crop and flip are fixed at plan time).
    this.lumaU = this._uniform(
      new Float32Array([crop.x0, crop.y0, crop.w, crop.h, flip, 0, 0, 0]),
    );
    // Clamp-to-edge sampler for the external (video) samples; required by
    // the current WGSL textureSampleBaseClampToEdge overload.
    this.sampler = device.createSampler();
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
    const lumaMod = device.createShaderModule({ code: makeLumaShader(source) });
    const boxMod = device.createShaderModule({ code: BOX_DOWNSAMPLE });
    const sadMod = device.createShaderModule({ code: makeSadCompute(tileW, MIN_BLOCK_VARIANCE) });
    this.lumaPipe = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: lumaMod, entryPoint: "vs_main" },
      fragment: {
        module: lumaMod,
        entryPoint: "fs_main",
        targets: [{ format: this.lumaFormat }],
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

    // Readback: with early arm (pipelined emit), samples land in batched
    // buffers — up to _batchK frames share one MAP_READ buffer and ONE
    // mapAsync armed at batch close. Each map completion is a GPU-process
    // round trip; one per K frames amortizes them. Batch buffers come from
    // a free list (recycled once every slot has been read and the batch
    // unmapped) so submit() never has to wait for one — with a fixed pair,
    // a lagging pump would stall the decode chain. Without early arm
    // (serial emit: iOS, canvas capture), the classic one-buffer-per-frame
    // pool keeps the device-verified per-frame await order.
    /** @type {GPUBuffer[]} */
    this._pool = [];
    /** @type {GPUBuffer[]} unmapped batch buffers ready for reuse */
    this._batchPool = [];
    this._batchK = earlyArm ? Math.max(1, batchK) : 1;
    /** @type {{buffer: GPUBuffer, slotsUsed: number, consumed: number,
     *   settled: boolean, map: Promise<void> | null,
     *   settledP: Promise<void>, resolveSettled: () => void} | null} */
    this._activeBatch = null;
    /** @type {Array<{buffer: GPUBuffer | null, immediate: boolean}>} */
    this._pending = [];
    this._havePrev = false;
    this._curIdx = 0;
    this._closed = false;
    this._importer = makeImporter(device);
    /** diagnostics: batch-map armed→settled latency total/count */
    this.mapMs = 0;
    this.mapN = 0;
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

  /** Start a batch accepting new frame slots; takes a recycled buffer or
   * creates one (the pool stays small — buffers return once fully read). */
  _openBatch() {
    const buffer =
      this._batchPool.pop() ??
      this.device.createBuffer({
        size: Math.max(this._blockBytes * this._batchK, 16),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    let resolveSettled;
    const settledP = new Promise((resolve) => {
      resolveSettled = resolve;
    });
    this._activeBatch = {
      buffer,
      slotsUsed: 0,
      consumed: 0,
      settled: false,
      map: null,
      settledP,
      resolveSettled,
    };
    return this._activeBatch;
  }

  /** Stop the active batch accepting slots and start its single readback
   * map. mapAsync cannot be armed before every write of the batch is
   * enqueued (writes to a mapped buffer are invalid), so the map covering
   * K frames can only start once the batch is full — or at stream end via
   * `flush()`. */
  _closeBatch() {
    const batch = this._activeBatch;
    if (!batch || batch.slotsUsed === 0) return;
    this._activeBatch = null;
    const armedAt = performance.now();
    batch.map = batch.buffer.mapAsync(GPUMapMode.READ);
    const settle = () => {
      this.mapMs += performance.now() - armedAt;
      this.mapN += 1;
      batch.settled = true;
      batch.resolveSettled();
    };
    // Failures settle too; nextSample() re-awaits and surfaces them.
    batch.map.then(settle, settle);
  }

  /** Close a partial batch so its frames' samples can resolve once the
   * stream ends (no further submits will arrive to fill it). */
  flush() {
    this._closeBatch();
  }

  /** Import a VideoFrame as an external texture (entry-point probing in
   * external.js; import inside the render pass that samples it). */
  _import(pass, frame) {
    return this._importer(pass, frame);
  }

  /**
   * Enqueue this frame's motion passes. Call from the decode-output
   * callback (the VideoFrame must still be open). Fire-and-forget: the
   * matching sample comes out of `nextSample()` in the same order.
   *
   * In `texture2d` mode the pipeline stages the frame itself
   * (drawImage + copyExternalImageToTexture) and passes the resulting
   * texture view; the VideoFrame is then only a fallback import target.
   *
   * @param {VideoFrame} frame
   * @param {GPUTextureView | null} [stagedView] pre-staged frame texture
   *   view (`texture2d` source mode).
   */
  submit(frame, stagedView = null) {
    if (this._closed) return;
    const device = this.device;
    const encoder = device.createCommandEncoder({ label: "motion" });

    // 1. Luma -> full-size r32float. In external mode the frame is
    // imported inside the render pass (views cannot leave it); in
    // texture2d mode the staged view is bound directly.
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
    const view = stagedView ?? this._import(pass, frame);
    pass.setPipeline(this.lumaPipe);
    pass.setBindGroup(0, device.createBindGroup({
      layout: this.lumaPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: view },
        { binding: 1, resource: { buffer: this.lumaU } },
        { binding: 2, resource: this.sampler },
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
      cp.setPipeline(this.sadPipe);
      cp.setBindGroup(0, this.sadBinds[this._curIdx]);
      cp.dispatchWorkgroups(
        Math.ceil(this.geo.gridW / this.tileW),
        this.geo.gridH,
      );
      cp.end();
      if (this._batchK > 1) {
        // Batched early arm: copy into this frame's slot of the active
        // batch and arm the one map covering the batch when it fills.
        const batch = this._activeBatch ?? this._openBatch();
        const offset = batch.slotsUsed * this._blockBytes;
        encoder.copyBufferToBuffer(
          this.blockBuf,
          0,
          batch.buffer,
          offset,
          this._blockBytes,
        );
        batch.slotsUsed += 1;
        device.queue.submit([encoder.finish()]);
        if (batch.slotsUsed === this._batchK) this._closeBatch();
        this._pending.push({ batch, offset, immediate: false });
      } else {
        const readBuf = this._takeBuffer();
        encoder.copyBufferToBuffer(
          this.blockBuf,
          0,
          readBuf,
          0,
          this._blockBytes,
        );
        device.queue.submit([encoder.finish()]);
        this._pending.push({ buffer: readBuf, immediate: false, settled: false });
      }
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
   * Whether the next `nextSample()` would resolve without waiting for
   * the GPU (the head readback's map already settled, or the frame is
   * the seed frame with no sample).
   * @returns {boolean}
   */
  isReady() {
    const item = this._pending[0];
    if (!item || item.immediate) return true;
    // A batch item is ready once its (batch-level) map has settled; an
    // unclosed head batch is not — only flush() can close it, so the
    // pump waits on headSettled(), which resolves at close + map.
    return item.batch ? item.batch.settled : item.settled;
  }

  /**
   * Promise that settles once the head pending sample's map has
   * resolved or failed (failures surface through `nextSample()`).
   * Resolves immediately when nothing is outstanding.
   * @returns {Promise<void>}
   */
  headSettled() {
    const item = this._pending[0];
    if (!item || item.immediate) return Promise.resolve();
    if (item.batch) return item.batch.settledP;
    return (item.map ?? item.buffer.mapAsync(GPUMapMode.READ)).then(
      () => {},
      () => {},
    );
  }

  /**
   * Resolve the next frame's drift sample (FIFO with `submit`).
   * @returns {Promise<{dx: number, reliable: boolean}>}
   */
  async nextSample() {
    const item = this._pending.shift();
    if (!item) throw new Error("GpuMotion.nextSample: no pending frame");
    if (item.immediate) return { ...NO_PREV };
    if (item.batch) {
      const batch = item.batch;
      await batch.map;
      if (this._closed) return { ...NO_PREV };
      // View the slot's blocks straight from the mapped range — no copy.
      // aggregateBlocks reads synchronously while still mapped, and the
      // batch unmaps only after its last frame has been consumed.
      const blocks = new Float32Array(
        batch.buffer.getMappedRange(item.offset, this._blockBytes),
      );
      const sample = aggregateBlocks(blocks, this.geo);
      if (++batch.consumed === batch.slotsUsed) {
        batch.buffer.unmap();
        this._batchPool.push(batch.buffer);
      }
      return sample;
    }
    const { buffer } = item;
    await (item.map ?? buffer.mapAsync(GPUMapMode.READ));
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
      // Batch items share their buffer; double destroy is a no-op.
      const buf = item.batch ? item.batch.buffer : item.buffer;
      if (buf) buf.destroy();
    }
    this._pending = [];
    this._activeBatch = null;
    for (const buf of this._pool) buf.destroy();
    this._pool = [];
    for (const buf of this._batchPool) buf.destroy();
    this._batchPool = [];
    this.fullTex.destroy();
    for (const t of this.smallTex) t.destroy();
    for (const b of [this.lumaU, this.boxU, this.sadU, this.rangesBuf, this.blockBuf]) {
      if (b) b.destroy();
    }
  }
}
