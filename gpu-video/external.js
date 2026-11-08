/**
 * VideoFrame → GPUExternalTexture import, across the API's entry-point
 * history:
 *
 * - `GPUDevice.importExternalTexture({ source })` — current spec
 *   (Chrome 151+; the device-level form takes the source in a descriptor
 *   and infers the color space from the frame).
 * - `GPURenderPassEncoder.importExternalTexture({ externalTexture })` —
 *   the previous form (Safari 18).
 * - `GPURenderPassEncoder.importExternalTexture(frame)` — the oldest form.
 *
 * `makeImporter` probes once and remembers the working entry point. The
 * returned view is bindable as `{ resource: view }` in a fragment-stage
 * bind group in all cases; importing inside the render pass that samples
 * it is legal under every variant.
 *
 * `probeExternalOrientation` calibrates the y direction of imported
 * textures: the WebGPU convention is uv.y = 0 at the source's top row,
 * but the only universal way to know what an engine actually delivers is
 * to sample a known gradient. A 2D canvas painted top-white/bottom-black
 * becomes a `VideoFrame` (WebCodecs canvas capture is upright row 0 = top
 * — verified on Chromium) and is imported and sampled through the same
 * machinery as the video frames; if the texel at uv (0.5, 0.5/H) reads
 * dark, this engine stores external textures bottom-up and the shaders
 * must flip uv.y.
 */
export function makeImporter(device) {
  let form = null;  const candidates = (pass, frame) => {
    const tries = [];
    if (typeof device?.importExternalTexture === "function") {
      tries.push(["deviceSrc", () => device.importExternalTexture({ source: frame })]);
      tries.push(["devicePos", () => device.importExternalTexture(frame)]);
    }
    if (typeof pass?.importExternalTexture === "function") {
      tries.push([
        "passDict",
        () => pass.importExternalTexture({ externalTexture: frame }),
      ]);
      tries.push(["passPos", () => pass.importExternalTexture(frame)]);
    }
    return tries;
  };
  return (pass, frame) => {
    if (form) {
      const hit = candidates(pass, frame).find(([name]) => name === form);
      if (hit) return hit[1]();
      form = null; // the remembered entry point vanished; re-probe
    }
    for (const [name, fn] of candidates(pass, frame)) {
      try {
        const view = fn();
        form = name;
        return view;
      } catch {
        /* try the next form */
      }
    }
    throw new Error("no importExternalTexture entry point found");
  };
}

/**
 * Whether this engine can import VideoFrames as external textures at all
 * (any of the entry-point forms above). Firefox ships WebGPU without
 * `importExternalTexture` — callers fall back to the canvas-staging mode
 * (`drawImage` + `queue.copyExternalImageToTexture` + `texture_2d`).
 *
 * @param {GPUDevice} device
 * @returns {boolean}
 */
export function supportsExternalImport(device) {
  return (
    typeof device?.importExternalTexture === "function" ||
    typeof GPURenderPassEncoder?.prototype?.importExternalTexture === "function"
  );
}

/**
 * Returns 1 when this engine stores imported external textures bottom-up
 * (uv.y = 0 is the bottom row) and 0 for the spec top-left convention.
 * Any probe failure (no canvas VideoFrame support, import error) falls
 * back to 0 — the convention the shaders are written for by default.
 *
 * @param {GPUDevice} device
 * @param {(pass: GPURenderPassEncoder, source: VideoFrame | HTMLCanvasElement) => GPUExternalTexture} importExternal
 * @returns {Promise<0 | 1>}
 */
export async function probeExternalOrientation(device, importExternal) {
  const SIZE = 8;
  let probeFrame = null;
  let rb = null;
  let tex = null;
  try {
    if (typeof OffscreenCanvas === "undefined" || typeof VideoFrame === "undefined") {
      return 0;
    }
    // Top half white, bottom half black, hard edge at row SIZE/2.
    const canvas = new OffscreenCanvas(SIZE, SIZE);
    const ctx = canvas.getContext("2d", { alpha: false });
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, SIZE, SIZE / 2);
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, SIZE / 2, SIZE, SIZE / 2);
    probeFrame = new VideoFrame(canvas, { timestamp: 0 });

    const module = device.createShaderModule({ code: /* wgsl */ `
      @group(0) @binding(0) var srcTex: texture_external;
      @group(0) @binding(1) var samp: sampler;
      @fragment fn fs_main() -> @location(0) vec4<f32> {
        // First texel row (uv.y = 0.5 / SIZE): white under a top-left
        // origin, black under a bottom-up store.
        return textureSampleBaseClampToEdge(srcTex, samp, vec2<f32>(0.5 / ${SIZE}.0, 0.5 / ${SIZE}.0));
      }` });
    const pipe = device.createRenderPipeline({
      layout: "auto",
      fragment: { module, entryPoint: "fs_main", targets: [{ format: "bgra8unorm" }] },
    });
    tex = device.createTexture({
      format: "bgra8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      size: [1, 1],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: tex.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    const view = importExternal(pass, probeFrame);
    pass.setPipeline(pipe);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: view },
        { binding: 1, resource: device.createSampler({ magFilter: "nearest", minFilter: "nearest" }) },
      ],
    }));
    pass.draw(1);
    pass.end();
    rb = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer(
      { texture: tex },
      { buffer: rb, bytesPerRow: 4 },
      { width: 1, height: 1 },
    );
    device.queue.submit([encoder.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(rb.getMappedRange());
    const luma = 0.299 * bytes[2] + 0.587 * bytes[1] + 0.114 * bytes[0];
    rb.unmap();
    return luma < 128 ? 1 : 0;
  } catch {
    return 0;
  } finally {
    try {
      probeFrame?.close();
    } catch {
      /* already closed */
    }
    rb?.destroy();
    tex?.destroy();
  }
}
