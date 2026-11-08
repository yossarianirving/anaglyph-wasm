/**
 * WGSL shader sources for the WebGPU anaglyph pipeline.
 *
 * - `makeBlendShader(source)`: fullscreen fragment pass compositing two video
 *   frames into the anaglyph — a direct port of `combine_slices` in
 *   anaglyph-rs `anaglyph.rs`: `out[c] = dot(L, m0[c*3..c*3+3]) +
 *   dot(R, m1[c*3..c*3+3])`, clamped to [0,1]. The eye matrices are
 *   uploaded as uniforms; the Rust table stores output channel c's
 *   coefficients in flat row c, while WGSL `m * rgb` distributes input
 *   channel k across matrix COLUMN k, so the upload transposes
 *   (`packBlendUniform` in gpu-pipeline.js).
 *   Like the wasm video path there is no per-pair parallax shift — motion
 *   estimation only picks the eye order.
 * - `makeLumaShader(source)`: grayscale of one frame into the R32Float
 *   analysis texture, storing `y * 255` so thresholds match the Rust
 *   estimator's 0..255 domain.
 * - `SAD_COMPUTE` (`makeSadCompute`): per-block best horizontal shift, SAD
 *   and variance over the small luma texture, mirroring the block matcher in
 *   motion.rs (same 8px blocks, ±12px search, first-minimum tie-break).
 *   Each workgroup stages one row strip of `TILE_W` block columns in
 *   workgroup memory so every texel is read from memory once; the CPU
 *   reproduces `observe`'s median + reliable-ratio over the readback.
 *
 * Source modes (`source` parameter):
 * - `"external"` — `texture_external` sampled with
 *   `textureSampleBaseClampToEdge(tex, samp, uv)` and an explicit
 *   clamp-to-edge sampler (the sampler-less overload was removed from WGSL;
 *   filtering stays nearest / clamp-to-edge by spec). Used where
 *   `importExternalTexture` exists: Chrome, Safari/iOS.
 * - `"texture2d"` — `texture_2d<f32>` sampled with `textureSample`. For
 *   engines without `importExternalTexture` (Firefox): the pipeline stages
 *   each VideoFrame through a 2D canvas (`drawImage`) plus
 *   `queue.copyExternalImageToTexture` into an `rgba8unorm` texture, which
 *   carries the frame's visible content top-row-first per spec.
 *
 * Coordinate convention: `FULLSCREEN_VS` emits uv with uv.y = 0 at the
 * TOP of the destination — the WebGPU texture convention (texel row 0 is
 * the top row, both for imported external textures and regular ones). A
 * sampled identity copy therefore preserves orientation through blend →
 * targetTex → blit → canvas and through the `copyTextureToBuffer` bytes
 * path alike. Measured on Chromium: an imported VideoFrame external
 * texture is the frame's *visible rect* (coded padding already cropped,
 * independent of where the rect sits in the coded buffer) stored top-left
 * origin, exactly matching ffmpeg's decoded rows.
 *
 * The `crop`/`flip` uniforms keep the shaders engine-quirk tolerant:
 * `crop` (x0, y0, w, h) is in image (top-left) units and normally selects
 * the full texture — it only trims the last row/column for odd visible
 * dimensions; `flip` (0/1, from `probeExternalOrientation` in
 * external.js) mirrors uv.y for a hypothetical engine that stores
 * external textures bottom-up (always 0 in texture2d mode, whose copy is
 * spec-top-origin). Sampling maps `crop` first, then flips, so the crop
 * window always describes image rows regardless of storage.
 *
 * Sampling returns display-encoded (sRGB-ish) values in both modes — the
 * same domain the Rust pipeline's u8 blending works in; no
 * transfer-function conversion needed.
 */

/** Shared vertex stage: fullscreen triangle carrying destination UVs
 * (uv.y = 0 at the top row, texture convention). */
const FULLSCREEN_VS = /* wgsl */ `
struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  var out: VOut;
  out.pos = vec4<f32>(p[vi], 0.0, 1.0);
  out.uv = vec2<f32>((p[vi].x + 1.0) * 0.5, (1.0 - p[vi].y) * 0.5);
  return out;
}`;

const TEX_TYPE = {
  external: "texture_external",
  texture2d: "texture_2d<f32>",
};

const SAMPLE_CALL = {
  external: "textureSampleBaseClampToEdge",
  texture2d: "textureSample",
};

/** Flip-aware crop sampling shared by the frame-source passes.
 * `crop` selects a window in image (top-left) uv units; `flip` mirrors
 * about the image center afterwards for bottom-up external stores. */
function sampleCroppedWgsl(source) {
  return /* wgsl */ `
fn sampleCropped(
  tex: ${TEX_TYPE[source]},
  uv: vec2<f32>,
  crop: vec4<f32>,
  flip: f32,
) -> vec4<f32> {
  // uv in [0,1] spans the output image; crop selects the source window
  // (normally the full frame; odd-dimension output trims the last row
  // or column). Out-of-range coordinates edge-replicate (the sampler is
  // clamp-to-edge), matching the Rust pipeline cropping before blending.
  var src = crop.xy + uv * crop.zw;
  if (flip > 0.5) {
    src.y = 1.0 - src.y;
  }
  return ${SAMPLE_CALL[source]}(tex, samp, src);
}`;
}

/**
 * Blend pass uniforms (128 bytes, see `planFromFrame` in gpu-pipeline.js):
 *
 *   crop:  vec4<f32>     source window in image uv units (x0, y0, w, h)
 *   flags: vec4<f32>     flags.x = external-texture y flip (0/1)
 *   m0:    mat3x3<f32>   left-eye matrix (Rust rows = WGSL columns)
 *   m1:    mat3x3<f32>   right-eye matrix
 *
 * @param {"external" | "texture2d"} [source] frame source mode.
 */
export function makeBlendShader(source = "external") {
  return /* wgsl */ `
struct BlendU {
  crop: vec4<f32>,
  flags: vec4<f32>,
  m0: mat3x3<f32>,
  m1: mat3x3<f32>,
};

@group(0) @binding(0) var leftTex: ${TEX_TYPE[source]};
@group(0) @binding(1) var rightTex: ${TEX_TYPE[source]};
@group(0) @binding(2) var<uniform> u: BlendU;
// Explicit sampler for the samples: the current WGSL spec dropped the
// sampler-less textureSampleBaseClampToEdge overload. Addressing must be
// clamp-to-edge (external-texture filtering is fixed by spec anyway).
@group(0) @binding(3) var samp: sampler;

${FULLSCREEN_VS}

// Samples are display-encoded (sRGB-ish) — the same domain the Rust
// pipeline blends u8 values in.
${sampleCroppedWgsl(source)}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let l = sampleCropped(leftTex, in.uv, u.crop, u.flags.x).rgb;
  let r = sampleCropped(rightTex, in.uv, u.crop, u.flags.x).rgb;
  let color = u.m0 * l + u.m1 * r;
  return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}`;
}

/** External-texture blend shader (default source mode). */
export const BLEND_SHADER = makeBlendShader("external");

/**
 * Luma pass (32-byte uniform): `crop: vec4<f32>` + `flags: vec4<f32>`
 * (flags.x = y flip) as in the blend shader. Writes Rec.601 luma scaled
 * to 0..255 into an R32Float target so the analysis domain matches the
 * Rust estimator's u8 input.
 *
 * @param {"external" | "texture2d"} [source] frame source mode.
 */
export function makeLumaShader(source = "external") {
  return /* wgsl */ `
struct LumaU {
  crop: vec4<f32>,
  flags: vec4<f32>,
};

@group(0) @binding(0) var srcTex: ${TEX_TYPE[source]};
@group(0) @binding(1) var<uniform> u: LumaU;
@group(0) @binding(2) var samp: sampler;

${FULLSCREEN_VS}

// Samples are display-encoded (sRGB-ish) — the same domain the Rust
// estimator's u8 luma works in (see the note in the blend shader).
${sampleCroppedWgsl(source)}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let c = sampleCropped(srcTex, in.uv, u.crop, u.flags.x).rgb;
  let y = dot(c, vec3<f32>(0.299, 0.587, 0.114));
  return vec4<f32>(y * 255.0, 0.0, 0.0, 1.0);
}`;
}

/** External-texture luma shader (default source mode). */
export const LUMA_SHADER = makeLumaShader("external");


/**
 * Exact box downsample from the full-size luma texture to the analysis
 * size, using the same integer pixel ranges as Rust's `ranges()`
 * (uploaded once as `ranges` = [x0,x1] x outW then [y0,y1] x outH). This
 * reproduces the estimator's box average (its integer-vs-float luma
 * difference is inherent to reading decoder output through the fragment
 * stage) and avoids point-sampling aliasing at large scale factors.
 *
 * Bindings:
 *   0  src     texture_2d<f32>            full-size luma (textureLoad)
 *   1  dst     texture_storage_2d<r32float, write>
 *   2  ranges  storage array<i32>
 *   3  uniform srcW, srcH, outW, outH (u32 x4)
 */
export const BOX_DOWNSAMPLE = /* wgsl */ `
struct BoxU {
  srcW: u32,
  srcH: u32,
  outW: u32,
  outH: u32,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<storage, read> ranges: array<i32>;
@group(0) @binding(3) var<uniform> u: BoxU;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.outW || g.y >= u.outH) {
    return;
  }
  let x0 = ranges[g.x * 2u];
  let x1 = ranges[g.x * 2u + 1u];
  let yBase = u.outW * 2u;
  let y0 = ranges[yBase + g.y * 2u];
  let y1 = ranges[yBase + g.y * 2u + 1u];
  var sum = 0.0;
  var n = 0u;
  for (var y: i32 = y0; y < y1; y = y + 1) {
    for (var x: i32 = x0; x < x1; x = x + 1) {
      sum = sum + textureLoad(src, vec2<i32>(x, y), 0).r;
      n = n + 1u;
    }
  }
  textureStore(dst, vec2<i32>(i32(g.x), i32(g.y)),
    vec4<f32>(sum / f32(max(n, 1u)), 0.0, 0.0, 1.0));
}`;

/**
 * Blit pass: copy the blend target texture to the WebGPU canvas at 1:1
 * size. Values in `targetTex` are already sRGB-encoded by the blend
 * shader, so this copies raw (plain sampling, no transfer conversion).
 *
 * Bindings: 0 = texture_2d<f32>, 1 = sampler (linear-clamp; exact texel
 * centers make the filter a no-op).
 */
export const BLIT_SHADER = /* wgsl */ `
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;

${FULLSCREEN_VS}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  return textureSample(srcTex, srcSampler, in.uv);
}`;

/**
 * SAD compute pass over the analysis-size luma of two frames, mirroring
 * `match_block` / `match_frames` in motion.rs per block:
 *
 * - variance of the PREV block; below `minVariance` the block is skipped
 *   (flat), written as qualified = 0;
 * - search d in [-MAX_SHIFT, +MAX_SHIFT] for min SAD(prev[x], cur[x+d]),
 *   first-minimum tie-break (iteration order + strict `<`);
 * - |best d| == MAX_SHIFT saturates the search window -> skipped;
 * - parabolic subpixel refinement from the SAD curve at best d ± 1;
 * - qualified blocks write [shift, meanSad (best SAD / 64 px),
 *   stdDev (sqrt of prev variance), 1.0].
 *
 * `observe`'s aggregation (quality ratio, median, /scale) runs on the CPU
 * over this buffer — see `aggregateBlocks` in motion.js.
 *
 * `TILE_W` (block columns per workgroup) is a template slot sized to fit
 * both staged strips in `maxComputeWorkgroupStorageSize`; dispatch is
 * (ceil(gridW / TILE_W), gridH), one block row per workgroup, block grid
 * columns starting at luma x = MAX_SHIFT + gx * BLOCK exactly like Rust.
 *
 * Bindings:
 *   0  prev      texture_2d<f32> (R32Float, 0..255)
 *   1  cur       texture_2d<f32>
 *   2  blockOut  storage array<f32>, 4 per block
 *   3  uniform   srcW, srcH, gridW, gridH (u32 x4) + minVariance (f32)
 */
export function makeSadCompute(tileW, minVariance) {
  return /* wgsl */ `
const BLOCK: u32 = 8u;
const MAX_SHIFT: u32 = 12u;
const TILE_W: u32 = ${tileW}u;
const STRIP_W: u32 = TILE_W * BLOCK + 2u * MAX_SHIFT;

struct SadU {
  srcW: u32,
  srcH: u32,
  gridW: u32,
  gridH: u32,
  minVariance: f32,
};

@group(0) @binding(0) var prevTex: texture_2d<f32>;
@group(0) @binding(1) var curTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> blockOut: array<f32>;
@group(0) @binding(3) var<uniform> u: SadU;

var<workgroup> prevStrip: array<f32, STRIP_W * BLOCK>;
var<workgroup> curStrip: array<f32, STRIP_W * BLOCK>;

fn sadAt(c0: u32, d: i32) -> f32 {
  var sad = 0.0;
  for (var y: u32 = 0u; y < BLOCK; y = y + 1u) {
    let row = y * STRIP_W;
    for (var x: u32 = 0u; x < BLOCK; x = x + 1u) {
      let a = prevStrip[row + c0 + x];
      let c = curStrip[u32(i32(row + c0) + i32(x) + d)];
      sad = sad + abs(a - c);
    }
  }
  return sad;
}

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tile = wid.x;
  let blockRow = wid.y;
  let lane = lid.x;

  // Stage BLOCK rows for both frames, columns
  // [tile*TILE_W*BLOCK, ... + STRIP_W): the first MAX_SHIFT columns are
  // the left search margin, the last MAX_SHIFT the right one. Block gx's
  // left edge in strip coordinates is therefore (gx - tile*TILE_W)*BLOCK
  // + MAX_SHIFT. Coordinates are in-bounds by construction of the grid
  // (blocks keep MAX_SHIFT of image margin); clamp only protects the
  // staging margins.
  let srcX0 = i32(tile * TILE_W * BLOCK);
  let srcY0 = i32(blockRow * BLOCK);
  for (var i: u32 = lane; i < STRIP_W * BLOCK; i = i + 64u) {
    let sx = clamp(srcX0 + i32(i % STRIP_W), 0, i32(u.srcW) - 1);
    let sy = clamp(srcY0 + i32(i / STRIP_W), 0, i32(u.srcH) - 1);
    let c = vec2<i32>(sx, sy);
    prevStrip[i] = textureLoad(prevTex, c, 0).r;
    curStrip[i] = textureLoad(curTex, c, 0).r;
  }
  workgroupBarrier();

  for (var b: u32 = lane; b < TILE_W; b = b + 64u) {
    let gx = tile * TILE_W + b;
    if (gx >= u.gridW) {
      break;
    }
    let outIdx = (blockRow * u.gridW + gx) * 4u;
    let c0 = b * BLOCK + MAX_SHIFT;

    // Variance of the prev block (Rust skips flat blocks before search).
    var sum = 0.0;
    var sumSq = 0.0;
    for (var y2: u32 = 0u; y2 < BLOCK; y2 = y2 + 1u) {
      let row2 = y2 * STRIP_W;
      for (var x2: u32 = 0u; x2 < BLOCK; x2 = x2 + 1u) {
        let v = prevStrip[row2 + c0 + x2];
        sum = sum + v;
        sumSq = sumSq + v * v;
      }
    }
    let n = f32(BLOCK * BLOCK);
    let mean = sum / n;
    let varr = max(0.0, sumSq / n - mean * mean);
    if (varr < u.minVariance) {
      blockOut[outIdx + 0u] = 0.0;
      blockOut[outIdx + 1u] = 0.0;
      blockOut[outIdx + 2u] = 0.0;
      blockOut[outIdx + 3u] = 0.0;
      continue;
    }

    var bestD: i32 = 0;
    var bestSad = f32(1.0e30);
    for (var d: i32 = -i32(MAX_SHIFT); d <= i32(MAX_SHIFT); d = d + 1) {
      let s = sadAt(c0, d);
      if (s < bestSad) {
        bestSad = s;
        bestD = d;
      }
    }
    if (abs(bestD) == i32(MAX_SHIFT)) {
      // True shift outside the search window; block carries no usable
      // information and its SAD cannot judge match quality.
      blockOut[outIdx + 0u] = 0.0;
      blockOut[outIdx + 1u] = 0.0;
      blockOut[outIdx + 2u] = 0.0;
      blockOut[outIdx + 3u] = 0.0;
      continue;
    }

    // Parabolic subpixel refinement, same as Rust (denom > 0 guard; a
    // positive finite denominator makes the offset finite, so the Rust
    // is_finite check is implied here too).
    let shiftPx = f32(bestD);
    let sm = sadAt(c0, bestD - 1);
    let sp = sadAt(c0, bestD + 1);
    let denom = sm + sp - 2.0 * bestSad;
    var shift = shiftPx;
    if (denom > 0.0) {
      let offset = (sm - sp) / (2.0 * denom);
      shift = shiftPx + clamp(offset, -0.5, 0.5);
    }

    blockOut[outIdx + 0u] = shift;
    blockOut[outIdx + 1u] = bestSad / n;
    blockOut[outIdx + 2u] = sqrt(varr);
    blockOut[outIdx + 3u] = 1.0;
  }
}`;
}
