/**
 * WGSL validation for gpu-video/shaders.js via the `naga` CLI
 * (`cargo install naga-cli`).
 *
 * `texture_external` is Web-only (naga does not know it) and
 * `textureSampleBaseClampToEdge` takes no sampler argument, so before
 * handing a shader to naga we textually substitute them with the nearest
 * portable equivalents (`texture_2d<f32>` + `textureSample` with a dummy
 * sampler at an otherwise-unused binding). Everything else — types,
 * matrix arithmetic, storage/workgroup memory rules, control flow — is
 * validated verbatim. The script also checks the SAD kernel's workgroup
 * memory against the WebGPU spec minimum for the tile widths motion.js
 * may choose.
 *
 * Run: `node gpu-video/validate-shaders.mjs` (skips when naga is absent).
 */

import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  BLEND_SHADER,
  LUMA_SHADER,
  BLIT_SHADER,
  BOX_DOWNSAMPLE,
  makeSadCompute,
} from "./shaders.js";

const hasNaga = spawnSync("naga", ["--version"], { stdio: "ignore" }).status === 0;
if (!hasNaga) {
  console.log("naga not installed (cargo install naga-cli) — skipping WGSL validation");
  process.exit(0);
}

/** Substitute external-texture constructs naga cannot parse. */
function portable(shader) {
  return shader
    .replaceAll(
      "texture_external",
      "texture_2d<f32>",
    )
    .replace(
      "@group(0) @binding(0)",
      "@group(0) @binding(99) var dummySampler: sampler;\n@group(0) @binding(0)",
    )
    .replace(
      /textureSampleBaseClampToEdge\(([^,]+), ([^)]+)\)/g,
      "textureSample($1, dummySampler, $2)",
    );
}

const dir = mkdtempSync(join(tmpdir(), "wgsl-"));
let failures = 0;
const cases = {
  blend: BLEND_SHADER,
  luma: LUMA_SHADER,
  blit: BLIT_SHADER,
  box: BOX_DOWNSAMPLE,
  // The two tile widths motion.js computes: spec-minimum (16 KB) and
  // 32 KB workgroup storage.
  sad_tile29: makeSadCompute(29, 6.0),
  sad_tile61: makeSadCompute(61, 6.0),
};

for (const [name, shader] of Object.entries(cases)) {
  const file = join(dir, `${name}.wgsl`);
  writeFileSync(file, portable(shader));
  const out = join(dir, `${name}.out.wgsl`);
  const result = spawnSync("naga", [file, out], { encoding: "utf8" });
  if (result.status === 0) {
    console.log(`ok    ${name}`);
  } else {
    failures += 1;
    console.log(`FAIL  ${name}`);
    console.log((result.stderr || result.stdout).trim().slice(0, 2000));
  }
}

// Workgroup memory accounting (both strips): (tileW*8 + 24) * 8 * 4B * 2.
for (const [tileW, limit] of [
  [29, 16384],
  [61, 32768],
]) {
  const bytes = (tileW * 8 + 24) * 8 * 4 * 2;
  const ok = bytes <= limit;
  console.log(
    `${ok ? "ok  " : "FAIL"}  sad tileW=${tileW}: ${bytes} B workgroup memory (limit ${limit} B)`,
  );
  if (!ok) failures += 1;
}

process.exit(failures === 0 ? 0 : 1);
