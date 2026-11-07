/**
 * Unit tests for the pairing module: `node --test js/`
 *
 * Semantics mirror anaglyph-rs `Pairer` (fixed and auto): window of two
 * frames, output stamped with the older frame's presentation time, auto mode
 * gated on the tracker's `confirmed_through`.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  assignPair,
  AutoFramePairer,
  createPairStrategy,
  FramePairer,
  unpackReversals,
} from "./pairing.js";

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
  assert.equal(pairer.push({ index: 0, timestamp: 0 }), null);

  const pair1 = pairer.push({ index: 1, timestamp: 1000 });
  assert.equal(pair1.left.timestamp, 1000);
  assert.equal(pair1.right.timestamp, 0);

  const pair2 = pairer.push({ index: 2, timestamp: 2000 });
  assert.equal(pair2.left.timestamp, 2000);
  assert.equal(pair2.right.timestamp, 1000);
});

test("output timestamp is the older frame's, duration is the gap", () => {
  const pairer = new FramePairer("counter-clockwise");
  pairer.push({ index: 0, timestamp: 500 });
  const pair = pairer.push({ index: 1, timestamp: 1300 });
  assert.equal(pair.timestamp, 500);
  assert.equal(pair.duration, 800);
});

test("non-advancing timestamps fall back to the frame duration", () => {
  const pairer = new FramePairer();
  pairer.push({ index: 0, timestamp: 100 });
  const pair = pairer.push({ index: 1, timestamp: 100, duration: 33 });
  assert.equal(pair.duration, 33);
});

test("held() exposes the trailing frame that never paired", () => {
  const pairer = new FramePairer();
  const first = { index: 0, timestamp: 0 };
  const second = { index: 1, timestamp: 10 };
  pairer.push(first);
  pairer.push(second);
  assert.equal(pairer.held(), second);
});

test("unknown direction is rejected", () => {
  assert.throws(() => new FramePairer("sideways"), /unknown direction/);
});

/**
 * Stub tracker mirroring the wasm `Tracker` shape. `script[frameIndex]`
 * optionally overwrites `confirmed` / `direction` / `reversals` after that
 * frame's push.
 */
function stubTracker(script = {}) {
  const pushes = [];
  const tracker = {
    pushes,
    confirmed: -1,
    direction: "clockwise",
    pendingReversals: [],
    push(index, timeSecs, dx, reliable) {
      pushes.push({ index, timeSecs, dx, reliable });
      const step = script[index] || {};
      if ("confirmed" in step) tracker.confirmed = step.confirmed;
      if ("direction" in step) tracker.direction = step.direction;
      if ("reversals" in step) tracker.pendingReversals = step.reversals;
    },
    confirmedThrough() {
      return tracker.confirmed;
    },
    currentDirection() {
      return tracker.direction;
    },
    finish() {
      tracker.confirmed = pushes.at(-1)?.index ?? -1;
    },
    takeReversals() {
      const out = tracker.pendingReversals;
      tracker.pendingReversals = [];
      return out;
    },
  };
  return tracker;
}

const frames = (count, gapUs = 30_000) =>
  Array.from({ length: count }, (_, i) => ({
    index: i,
    timestamp: i * gapUs,
    duration: gapUs,
    rgba: `rgba-${i}`,
  }));

test("auto: nothing emits until the tracker confirms", () => {
  const tracker = stubTracker(); // confirmed stays -1
  const pairer = new AutoFramePairer({ tracker, observe: () => ({ dx: 0, reliable: false }) });
  for (const frame of frames(5)) {
    const result = pairer.push(frame);
    assert.deepEqual(result.pairs, []);
    assert.equal(result.releasedBefore, -1);
  }
  assert.equal(pairer.held().length, 5);
  assert.equal(pairer.maxPending, 5);
});

test("auto: frames stream out as confirmation catches up", () => {
  // Steady trend: every new frame extends the extremum, so confirmation
  // reaches the frame just pushed and the newest pair becomes ready at once.
  const script = {};
  for (let i = 1; i <= 6; i++) script[i] = { confirmed: i };
  const pairer = new AutoFramePairer({ tracker: stubTracker(script), observe: () => ({ dx: -2, reliable: true }) });

  const emitted = [];
  let last;
  for (const frame of frames(6)) {
    last = pairer.push(frame);
    emitted.push(...last.pairs);
  }
  assert.equal(emitted.length, 5);
  assert.deepEqual(
    emitted.map((p) => [p.right.index, p.left.index]),
    [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 5],
    ],
  );
  assert.equal(last.releasedBefore, 5); // only frame 5 still pending
});

test("auto: per-pair direction follows the tracker", () => {
  const tracker = stubTracker({
    1: { confirmed: 1, direction: "clockwise" },
    2: { confirmed: 2, direction: "clockwise" },
    3: { confirmed: 3, direction: "counter-clockwise" },
  });
  const pairer = new AutoFramePairer({ tracker, observe: () => ({ dx: 1, reliable: true }) });

  const out = [];
  for (const frame of frames(4)) out.push(pairer.push(frame));

  const first = out[1].pairs[0]; // pair (0,1) under clockwise: newer = left
  assert.equal(first.left.index, 1);
  assert.equal(first.right.index, 0);

  const third = out[3].pairs[0]; // pair (2,3) under counter-clockwise: older = left
  assert.equal(third.left.index, 2);
  assert.equal(third.right.index, 3);
});

test("auto: a confirmation jump drains multiple pairs at once", () => {
  // Reversal-style jump: confirmed -1 until frame 4, then everything at once.
  const tracker = stubTracker({ 4: { confirmed: 4 } });
  const pairer = new AutoFramePairer({ tracker, observe: () => ({ dx: 1, reliable: true }) });

  const results = frames(5).map((f) => pairer.push(f));
  assert.equal(results[4].pairs.length, 4);
  assert.equal(results[4].releasedBefore, 4); // frames 0..3 freed, frame 4 kept
  assert.equal(pairer.held().length, 1);
});

test("auto: pair timing uses the older frame", () => {
  const tracker = stubTracker({ 1: { confirmed: 1 } });
  const pairer = new AutoFramePairer({ tracker });
  pairer.push({ index: 0, timestamp: 1000, duration: 400 });
  const { pairs } = pairer.push({ index: 1, timestamp: 1400, duration: 400 });
  assert.equal(pairs[0].timestamp, 1000);
  assert.equal(pairs[0].duration, 400);
});

test("auto: flush commits and drains the tail", () => {
  const tracker = stubTracker(); // never confirms
  const pairer = new AutoFramePairer({ tracker });
  for (const frame of frames(3)) pairer.push(frame);

  const { pairs, releasedBefore } = pairer.flush();
  assert.deepEqual(
    pairs.map((p) => [p.right.index, p.left.index]),
    [
      [0, 1],
      [1, 2],
    ],
  );
  assert.equal(releasedBefore, 2);
  // Only the trailing unpaired frame is still held.
  assert.deepEqual(
    pairer.held().map((f) => f.index),
    [2],
  );
});

test("auto: tracker receives seconds and observed samples", () => {
  const tracker = stubTracker();
  const pairer = new AutoFramePairer({
    tracker,
    observe: (frame) => ({ dx: frame.index * 0.5, reliable: frame.index > 0 }),
  });
  pairer.push({ index: 0, timestamp: 2_000_000 });
  pairer.push({ index: 1, timestamp: 2_030_000 });

  assert.deepEqual(tracker.pushes, [
    { index: 0, timeSecs: 2, dx: 0, reliable: false },
    { index: 1, timeSecs: 2.03, dx: 0.5, reliable: true },
  ]);
});

test("auto: reversals surface once, right after the confirming push", () => {
  const reversal = [3, 0.09, 1, 0]; // frame 3 @ 0.09s, clockwise -> ccw
  const tracker = stubTracker({
    3: { confirmed: 3, reversals: reversal },
  });
  const pairer = new AutoFramePairer({ tracker });

  const results = frames(4).map((f) => pairer.push(f));
  assert.deepEqual(results[3].reversals, [
    { frame: 3, time: 0.09, from: "clockwise", to: "counter-clockwise" },
  ]);
  assert.deepEqual(results[1].reversals, []);
  // Drained once: the next push reports nothing new.
  const after = pairer.push({ index: 4, timestamp: 120_000 });
  assert.deepEqual(after.reversals, []);
});

test("unpackReversals maps direction bits and tolerates partial quads", () => {
  const out = unpackReversals([
    10, 1.5, 0, 1, // ccw -> cw
    20, 3.25, 1, 0, // cw -> ccw
    30, // truncated tail is ignored
  ]);
  assert.equal(out.length, 2);
  assert.deepEqual(out[0], { frame: 10, time: 1.5, from: "counter-clockwise", to: "clockwise" });
  assert.deepEqual(out[1], { frame: 20, time: 3.25, from: "clockwise", to: "counter-clockwise" });
});

test("strategy: fixed pairs every push and releases the older frame", () => {
  const strategy = createPairStrategy("clockwise");
  assert.equal(strategy.auto, false);
  assert.equal(strategy.push({ index: 0, timestamp: 0 }).pairs.length, 0);
  const r = strategy.push({ index: 1, timestamp: 1000 });
  assert.equal(r.pairs.length, 1);
  assert.equal(r.releasedBefore, 1);
  assert.deepEqual(r.reversals, []);
});

test("strategy: auto needs a tracker, fixed needs none", () => {
  assert.throws(() => createPairStrategy("auto"), /needs a tracker/);
  assert.throws(() => createPairStrategy("auto-reverse"), /needs a tracker/);
  assert.equal(createPairStrategy("auto", { tracker: stubTracker() }).auto, true);
});

test("strategy: flush on fixed has no trailing pairs", () => {
  const strategy = createPairStrategy("counter-clockwise");
  strategy.push({ index: 0, timestamp: 0 });
  strategy.push({ index: 1, timestamp: 10 });
  assert.deepEqual(strategy.flush().pairs, []);
});
