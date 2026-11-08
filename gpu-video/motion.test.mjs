/**
 * Pure-logic tests for the GPU motion module: analysis geometry must
 * reproduce motion.rs `MotionEstimator::new` / `match_frames` grid maths,
 * and `aggregateBlocks` must reproduce the quality gate + upper-median
 * reduction over the compute kernel's per-block results.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  analysisGeometry,
  aggregateBlocks,
  MAX_SHIFT,
  BLOCK,
  MIN_BLOCK_RATIO,
} from "./motion.js";

test("geometry: 1080p downscales to 256 wide with truncating height", () => {
  const geo = analysisGeometry(1920, 1080);
  assert.equal(geo.w, 256);
  assert.equal(geo.h, Math.trunc(1080 * (256 / 1920))); // 144
  assert.ok(Math.abs(geo.scale - 256 / 1920) < 1e-9);
  assert.equal(geo.gridW, Math.floor((256 - 2 * MAX_SHIFT) / BLOCK)); // 29
  assert.equal(geo.gridH, Math.floor(geo.h / BLOCK)); // 18
});

test("geometry: sub-target widths are matched 1:1", () => {
  const geo = analysisGeometry(200, 120);
  assert.equal(geo.w, 200);
  assert.equal(geo.h, 120);
  assert.equal(geo.scale, 1);
  assert.equal(geo.gridW, Math.floor((200 - 2 * MAX_SHIFT) / BLOCK)); // 22
  assert.equal(geo.gridH, 15);
});

test("geometry: too narrow to match gives an empty grid", () => {
  assert.equal(analysisGeometry(2 * MAX_SHIFT + BLOCK - 1, 64).gridW, 0);
  assert.equal(analysisGeometry(2 * MAX_SHIFT + BLOCK - 1, 64).gridH, 0);
  assert.equal(analysisGeometry(640, 6).gridH, 0); // height < BLOCK
});

test("geometry: box ranges tile the source exactly", () => {
  const geo = analysisGeometry(3840, 2160);
  assert.equal(geo.xStarts[0], 0);
  assert.equal(geo.xEnds[geo.w - 1], 3840);
  assert.equal(geo.yStarts[0], 0);
  assert.equal(geo.yEnds[geo.h - 1], 2160);
  for (let i = 0; i < geo.w; i++) {
    const start = Math.floor((i * 3840) / geo.w);
    assert.equal(geo.xStarts[i], start, `x start ${i}`);
    assert.ok(geo.xEnds[i] > geo.xStarts[i], `x range ${i} empty`);
    if (i > 0) assert.equal(geo.xStarts[i], geo.xEnds[i - 1], `x seam ${i}`);
  }
});

/** Pack per-block kernel results: [shift, meanSad, stdDev, qualified]. */
function blocks(entries) {
  const out = new Float32Array(entries.length * 4);
  entries.forEach(([shift, sad, dev, qualified], i) => {
    out.set([shift, sad, dev, qualified], i * 4);
  });
  return out;
}

const geo8 = { gridW: 4, gridH: 2, scale: 1 };

test("aggregate: upper median of qualified shifts", () => {
  const b = blocks([
    [1, 1, 3, 1], [2, 1, 3, 1], [3, 1, 3, 1], [4, 1, 3, 1],
    [5, 1, 3, 1], [6, 1, 3, 1], [7, 1, 3, 1], [8, 1, 3, 1],
  ]);
  // quality = 1 / (3 + 1) = 0.25 <= 0.45; median index 8/2 = 4 -> 5.
  assert.deepEqual(aggregateBlocks(b, geo8), { dx: 5, reliable: true });
});

test("aggregate: scale converts matched pixels to source pixels", () => {
  const b = blocks(repeatQual(8, 2));
  const geo = { gridW: 4, gridH: 2, scale: 0.5 };
  assert.equal(aggregateBlocks(b, geo).dx, 2 / 0.5); // median 2 / scale
});

function repeatQual(n, shift) {
  return new Array(n).fill([shift, 1, 3, 1]);
}

test("aggregate: no qualified blocks -> unreliable", () => {
  const b = blocks(repeatQual(8, 0).map(([s, a, d]) => [s, a, d, 0]));
  assert.deepEqual(aggregateBlocks(b, geo8), { dx: 0, reliable: false });
});

test("aggregate: too few qualified blocks -> unreliable", () => {
  const entries = repeatQual(8, 3).map((e, i) => (i === 0 ? e : [0, 0, 0, 0]));
  // 1 qualified of 8 grid cells < MIN_BLOCK_RATIO (0.2) when 1 < 1.6.
  assert.ok(1 < 8 * MIN_BLOCK_RATIO);
  assert.deepEqual(aggregateBlocks(blocks(entries), geo8), {
    dx: 0,
    reliable: false,
  });
});

test("aggregate: poor mean match quality reads as a cut", () => {
  const entries = repeatQual(8, 3).map(([s]) => [s, 5, 5, 1]);
  // quality = 5 / (5 + 1) = 0.833 > 0.45.
  assert.deepEqual(aggregateBlocks(blocks(entries), geo8), {
    dx: 0,
    reliable: false,
  });
});

test("aggregate: empty grid -> unreliable", () => {
  assert.deepEqual(
    aggregateBlocks(new Float32Array(0), { gridW: 0, gridH: 0, scale: 1 }),
    { dx: 0, reliable: false },
  );
});
