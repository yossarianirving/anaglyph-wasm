/**
 * Blend uniform packing tests: the WGSL interpretation of the eye
 * matrices uploaded by packBlendUniform must equal anaglyph-rs
 * combine_slices (`out[c] = dot3(l, m[c*3..c*3+3]) + dot3(r, ...)`).
 *
 * The color matrices are diagonal and therefore transpose-invariant, so
 * a transposition slip only shows up on the gray/optimized/half-color
 * types — exactly the green-image regression this file guards.
 */
import test from "node:test";
import assert from "node:assert/strict";

import { ANAGLYPH_MATRICES, packBlendUniform } from "./gpu-pipeline.js";

/** anaglyph-rs combine_slices (src/anaglyph.rs) in JS. */
function combineSlices(l, r, m) {
  const out = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const left = m[c * 3] * l[0] + m[c * 3 + 1] * l[1] + m[c * 3 + 2] * l[2];
    const right =
      m[9 + c * 3] * r[0] + m[9 + c * 3 + 1] * r[1] + m[9 + c * 3 + 2] * r[2];
    out[c] = Math.min(1, Math.max(0, left + right));
  }
  return out;
}

/**
 * What the GPU computes from the uniform bytes: WGSL mat3x3 column k at
 * float offset base + 4k (vec3 at 16-byte stride, padding skipped), and
 * `m * v` = Σ_k v[k] * column_k.
 */
function wgslBlend(uf, l, r) {
  const out = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    let acc = 0;
    for (let k = 0; k < 3; k++) {
      acc += uf[8 + 4 * k + i] * l[k];
      acc += uf[20 + 4 * k + i] * r[k];
    }
    out[i] = Math.min(1, Math.max(0, acc));
  }
  return out;
}

const crop = { x0: 0, y0: 0, w: 1, h: 1 };

/** Deterministic pseudo-random channel triples, corners included. */
function samples() {
  const values = [0, 0.25, 0.5, 0.75, 1];
  const out = [];
  for (const a of values) {
    for (const b of values) {
      for (const c of values) {
        out.push([[a, b, c], [1 - a, c, b]]);
      }
    }
  }
  return out;
}

test("blend uniform: WGSL product matches combine_slices for every type", () => {
  for (const [type, matrix] of Object.entries(ANAGLYPH_MATRICES)) {
    const uf = packBlendUniform(matrix, crop, 0);
    assert.equal(uf.length, 32);
    for (const [l, r] of samples()) {
      const expected = combineSlices(l, r, matrix);
      const actual = wgslBlend(uf, l, r);
      for (let c = 0; c < 3; c++) {
        assert.ok(
          Math.abs(expected[c] - actual[c]) < 1e-6,
          `${type} l=[${l}] r=[${r}] channel ${c}: rust ${expected[c]} vs wgsl ${actual[c]}`,
        );
      }
    }
  }
});

test("blend uniform: half-color encodes red = luma(left), green/blue = right", () => {
  const uf = packBlendUniform(ANAGLYPH_MATRICES["half-color"], crop, 0);
  const luma = (c) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
  const l = [0.2, 0.6, 0.4];
  const r = [0.9, 0.3, 0.7];
  const out = wgslBlend(uf, l, r);
  assert.ok(Math.abs(out[0] - luma(l)) < 1e-6, `red: ${out[0]} != luma(left)`);
  assert.equal(out[1], r[1]);
  assert.equal(out[2], r[2]);
});

test("blend uniform: grayscale is red = luma(left), green/blue = luma(right)", () => {
  const uf = packBlendUniform(ANAGLYPH_MATRICES.grayscale, crop, 0);
  const l = [0.8, 0.2, 0.1];
  const r = [0.1, 0.7, 0.3];
  const out = wgslBlend(uf, l, r);
  // Rust rows: left eye luma on row 0, right eye luma on rows 1+2.
  const lumaL = 0.299 * l[0] + 0.587 * l[1] + 0.114 * l[2];
  const lumaR = 0.299 * r[0] + 0.587 * r[1] + 0.114 * r[2];
  assert.ok(Math.abs(out[0] - lumaL) < 1e-6, `red: ${out[0]} != ${lumaL}`);
  assert.ok(Math.abs(out[1] - lumaR) < 1e-6);
  assert.ok(Math.abs(out[2] - lumaR) < 1e-6);
});

test("blend uniform: crop and y-flip land at offsets 0..7", () => {
  const uf = packBlendUniform(
    ANAGLYPH_MATRICES.color,
    { x0: 0.125, y0: 0.25, w: 0.5, h: 0.75 },
    1,
  );
  assert.deepEqual([uf[0], uf[1], uf[2], uf[3], uf[4]], [0.125, 0.25, 0.5, 0.75, 1]);
});
