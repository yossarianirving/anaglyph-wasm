/**
 * DirectionTracker unit tests — ports of the Rust `DirectionTracker` tests
 * in anaglyph-rs `motion.rs` (`zigzag_finds_turning_points`,
 * `scene_cut_commits_pending_but_is_not_a_reversal`,
 * `pending_cap_bounds_ambiguity`, `finish_*`, `holds_frames_*`), plus the
 * trend→direction mapping and an integration pass through
 * `js/pairing.js`'s `AutoFramePairer` on a boomerang script.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { DirectionTracker, trendToDirection } from "./tracker.js";
import { unpackReversals } from "../js/pairing.js";

/** Push frames `start..start+dxs.length-1` with constant drift per frame. */
function feed(tracker, start, dxs) {
  dxs.forEach((dx, i) => {
    const frame = start + i;
    tracker.push(frame, frame / 30.0, dx, true);
  });
}

const repeat = (value, count) => new Array(count).fill(value);

test("trend to direction mapping (incl. reverse)", () => {
  assert.equal(trendToDirection("increasing", false), "counter-clockwise");
  assert.equal(trendToDirection("decreasing", false), "clockwise");
  assert.equal(trendToDirection(null, false), "clockwise");
  assert.equal(trendToDirection("increasing", true), "clockwise");
  assert.equal(trendToDirection("decreasing", true), "counter-clockwise");
  assert.equal(trendToDirection(null, true), "counter-clockwise");
});

test("zigzag finds turning points", () => {
  const t = new DirectionTracker();
  feed(t, 1, repeat(2.0, 30)); // rightward through frame 30
  assert.equal(t.currentDirection(), "counter-clockwise");
  assert.equal(t.confirmedThrough(), 30);
  feed(t, 31, repeat(-2.0, 30)); // leftward through frame 60
  let r = unpackReversals(t.takeReversals());
  assert.equal(r.length, 1);
  assert.equal(r[0].frame, 31);
  assert.equal(r[0].from, "counter-clockwise");
  assert.equal(r[0].to, "clockwise");
  feed(t, 61, repeat(2.0, 30));
  r = unpackReversals(t.takeReversals());
  assert.equal(r.length, 1);
  assert.equal(r[0].frame, 61);
  assert.equal(r[0].from, "clockwise");
  assert.equal(r[0].to, "counter-clockwise");
  assert.equal(t.currentDirection(), "counter-clockwise");
});

test("noisy drift still detects both reversals", () => {
  // Same xorshift64 pseudo-noise as the Rust test (BigInt domain).
  let state = 0x2545f4914f6cdd1dn;
  const noise = () => {
    state =
      (state * 6364136223846793005n + 1442695040888963407n) &
      0xffffffffffffffffn;
    return Number(state >> 40n) / 2 ** 24 * 1.4 - 0.7;
  };
  const t = new DirectionTracker();
  feed(t, 1, repeat(0, 40).map(() => 2.0 + noise()));
  feed(t, 41, repeat(0, 40).map(() => -2.0 + noise()));
  feed(t, 81, repeat(0, 40).map(() => 2.0 + noise()));
  const r = unpackReversals(t.takeReversals());
  assert.equal(r.length, 2, `reversals: ${JSON.stringify(r)}`);
  assert.ok(Math.abs(r[0].frame - 41) <= 3 && Math.abs(r[1].frame - 81) <= 3);
  assert.equal(t.currentDirection(), "counter-clockwise");
});

test("holds frames until the trend is decided", () => {
  const t = new DirectionTracker();
  feed(t, 1, repeat(0.5, 7)); // drift 3.5 < start threshold 4
  assert.equal(t.currentDirection(), "clockwise"); // undecided fallback
  assert.equal(t.confirmedThrough(), -1);
  feed(t, 8, repeat(0.5, 1)); // drift 4.0 decides it
  assert.equal(t.currentDirection(), "counter-clockwise");
  assert.equal(t.confirmedThrough(), 8);
});

test("scene cut commits pending but is not a reversal", () => {
  const t = new DirectionTracker();
  feed(t, 1, repeat(2.0, 19));
  t.push(20, 20 / 30, 0, false); // cut: unreliable
  assert.equal(t.confirmedThrough(), 20);
  assert.equal(t.currentDirection(), "counter-clockwise");
  feed(t, 21, repeat(2.0, 10)); // drift keeps rising through the cut
  assert.equal(t.takeReversals().length, 0);
  feed(t, 31, repeat(-2.0, 30));
  const r = unpackReversals(t.takeReversals());
  assert.equal(r.length, 1);
  // The true extremum (frame 30, before the drift turned) is preserved
  // across the forced commit.
  assert.equal(r[0].frame, 31);
});

test("pending cap bounds ambiguity", () => {
  const t = new DirectionTracker();
  feed(t, 1, repeat(2.0, 25)); // Increasing decided at frame 4, ext at 25
  feed(t, 26, repeat(0.0, 25)); // flat: no new extremum, no retreat
  assert.ok(t.confirmedThrough() >= 45);
  assert.equal(t.takeReversals().length, 0);
  feed(t, 51, repeat(-2.0, 30));
  const r = unpackReversals(t.takeReversals());
  assert.equal(r.length, 1);
  // The boundary moved to just past the commit point (not the extremum at
  // frame 25), but the swing was still measured against the extremum value.
  assert.ok(r[0].frame >= 45 && r[0].frame <= 47, `reversal: ${JSON.stringify(r)}`);
});

test("finish commits the tail without a reversal", () => {
  const t = new DirectionTracker();
  feed(t, 1, repeat(2.0, 20));
  feed(t, 21, repeat(-0.5, 10)); // retreat 5 < swing: not a reversal
  assert.equal(t.currentDirection(), "counter-clockwise");
  assert.ok(t.confirmedThrough() < 30);
  t.finish();
  assert.equal(t.confirmedThrough(), 30);
  assert.equal(t.takeReversals().length, 0);
});

test("finish without ever deciding a trend", () => {
  const t = new DirectionTracker();
  feed(t, 1, repeat(0.1, 10));
  assert.equal(t.confirmedThrough(), -1);
  t.finish();
  assert.equal(t.confirmedThrough(), 10);
  assert.equal(t.currentDirection(), "clockwise");
});

test("reverse flips the packed reversal direction bits", () => {
  const t = new DirectionTracker(true);
  feed(t, 1, repeat(2.0, 30));
  feed(t, 31, repeat(-2.0, 30));
  const r = unpackReversals(t.takeReversals());
  assert.equal(r.length, 1);
  assert.equal(r[0].from, "clockwise"); // increasing + reverse
  assert.equal(r[0].to, "counter-clockwise");
  assert.equal(t.currentDirection(), "counter-clockwise"); // decreasing + reverse
});

test("auto pairing follows the tracker through a boomerang", async () => {
  const { AutoFramePairer } = await import("../js/pairing.js");
  const tracker = new DirectionTracker();
  const samples = [];
  let pushIndex = 0;
  const pairer = new AutoFramePairer({
    observe: () => samples[pushIndex++] ?? { dx: 0, reliable: false },
    tracker,
  });
  const push = (dx, reliable = true) => {
    samples.push({ dx, reliable });
    const index = samples.length; // frames 1..n, timestamps from 0
    return pairer.push({ index, timestamp: ((index - 1) * 1e6) / 30, duration: 0 });
  };

  // Drift right for 40 frames (direction decides at frame 4), then back.
  const drained = [];
  let reversals = [];
  for (let i = 0; i < 40; i++) {
    const r = push(2.0);
    drained.push(...r.pairs);
    reversals.push(...r.reversals);
  }
  assert.ok(drained.length > 0);
  // Frames up to the confirmed extremum drain eagerly; all on the
  // increasing trend -> counter-clockwise, older frame is left.
  assert.ok(drained.every((p) => p.left.timestamp < p.right.timestamp));
  // (clockwise pairs assign the newer frame to left; here every drained
  // pair predates any reversal, so assignment must be older-left.)
  assert.equal(reversals.length, 0);

  const beforeSwitch = drained.length;
  for (let i = 40; i < 80; i++) {
    const r = push(-2.0);
    drained.push(...r.pairs);
    reversals.push(...r.reversals);
  }
  assert.equal(reversals.length, 1, "the boomerang turn is confirmed");
  const newPairs = drained.slice(beforeSwitch);
  // Pairs after the turn flip the eye order (newer frame feeds left).
  assert.ok(
    newPairs.some((p) => p.left.timestamp > p.right.timestamp),
    "post-reversal pairs assign the newer frame to the left eye",
  );

  // Flush commits the held-back tail.
  const tail = pairer.flush();
  drained.push(...tail.pairs);
  assert.ok(drained.length >= 78); // N inputs give N-1 pairs eventually
  assert.equal(drained[0].timestamp, 0);
  assert.ok(drained.at(-1).right.timestamp > drained.at(-2).right.timestamp);
});
