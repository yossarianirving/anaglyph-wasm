/**
 * Unit tests for the pairing module: `node --test js/`
 *
 * Semantics mirror anaglyph-rs `Pairer::Fixed` (window of two frames, output
 * stamped with the older frame's presentation time).
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { assignPair, FramePairer } from "./pairing.js";

test("clockwise pairing: newer frame is the left eye", () => {
  const a = { id: "a" };
  const b = { id: "b" };
  const pair = assignPair(a, b, "clockwise");
  assert.equal(pair.left, b);
  assert.equal(pair.right, a);
});

test("counter-clockwise pairing: older frame is the left eye", () => {
  const a = { id: "a" };
  const b = { id: "b" };
  const pair = assignPair(a, b, "counter-clockwise");
  assert.equal(pair.left, a);
  assert.equal(pair.right, b);
});

test("first frame returns null, every later frame completes a pair", () => {
  const pairer = new FramePairer("clockwise");
  assert.equal(pairer.push({ timestamp: 0 }), null);

  const pair1 = pairer.push({ timestamp: 1000 });
  assert.equal(pair1.left.timestamp, 1000);
  assert.equal(pair1.right.timestamp, 0);

  const pair2 = pairer.push({ timestamp: 2000 });
  assert.equal(pair2.left.timestamp, 2000);
  assert.equal(pair2.right.timestamp, 1000);
});

test("output timestamp is the older frame's, duration is the gap", () => {
  const pairer = new FramePairer("counter-clockwise");
  pairer.push({ timestamp: 500 });
  const pair = pairer.push({ timestamp: 1300 });
  assert.equal(pair.timestamp, 500);
  assert.equal(pair.duration, 800);
});

test("non-advancing timestamps fall back to the frame duration", () => {
  const pairer = new FramePairer();
  pairer.push({ timestamp: 100 });
  const pair = pairer.push({ timestamp: 100, duration: 33 });
  assert.equal(pair.duration, 33);
});

test("held() exposes the trailing frame that never paired", () => {
  const pairer = new FramePairer();
  const first = { timestamp: 0 };
  const second = { timestamp: 10 };
  pairer.push(first);
  pairer.push(second);
  assert.equal(pairer.held(), second);
});

test("unknown direction is rejected", () => {
  assert.throws(() => new FramePairer("sideways"), /unknown direction/);
});
