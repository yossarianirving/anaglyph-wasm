/**
 * Unit tests for the shared encoder planner (js/encoder-plan.js).
 * `VideoEncoder` is stubbed via the global — the planner only calls
 * `isConfigSupported`, so the whole codec/quality ladder is testable in
 * node (the mediabunny format classes are constructed lazily, so stubs
 * only need to be truthy).
 */
import assert from "node:assert/strict";
import test, { afterEach } from "node:test";

import {
  QUALITY_LEVELS,
  accelLabel,
  pickAvcLevel,
  planEncoder,
  qualityLabel,
  targetBitrate,
} from "./encoder-plan.js";

const originalEncoder = globalThis.VideoEncoder;
afterEach(() => {
  globalThis.VideoEncoder = originalEncoder;
});

/** Install a stub: supports everything, records configs. */
function stubSupportAll(reject = () => false) {
  const seen = [];
  globalThis.VideoEncoder = {
    isConfigSupported(config) {
      seen.push(config);
      return Promise.resolve({ supported: !reject(config) });
    },
  };
  return seen;
}

test("targetBitrate: standard tier reproduces the original sizing", () => {
  // clamp low end (tiny frame)
  assert.equal(targetBitrate(160, 90, 15, "standard"), 1_500_000);
  // clamp high end (huge frame)
  assert.equal(targetBitrate(3840, 2160, 60, "standard"), 12_000_000);
  // mid formula: 1280*720*30*0.1 = 2_764_800
  assert.equal(targetBitrate(1280, 720, 30, "standard"), 2_764_800);
  // default tier is standard
  assert.equal(targetBitrate(640, 360, 30), targetBitrate(640, 360, 30, "standard"));
});

test("targetBitrate: high and maximum scale bpp and clamps", () => {
  // 480x270@30: both tiers hit their floors
  assert.equal(targetBitrate(480, 270, 30, "high"), 3_000_000);
  assert.equal(targetBitrate(480, 270, 30, "maximum"), 6_000_000);
  // 1920x1080@30: 0.25 bpp = 15.55M (inside clamp), 0.5 bpp = 31.1M
  assert.equal(targetBitrate(1920, 1080, 30, "high"), 15_552_000);
  assert.equal(targetBitrate(1920, 1080, 30, "maximum"), 31_104_000);
  // clamps: 4K@60 maximum → capped at 48M
  assert.equal(targetBitrate(3840, 2160, 60, "maximum"), 48_000_000);
});

test("targetBitrate: unknown tier throws", () => {
  assert.throws(() => targetBitrate(640, 360, 30, "lossless"), /unknown encoder quality/);
});

test("pickAvcLevel covers size and rate", () => {
  assert.equal(pickAvcLevel(1280, 720, 30), 0x29);
  assert.equal(pickAvcLevel(1920, 1080, 30), 0x28);
  assert.equal(pickAvcLevel(1920, 1080, 60), 0x2a);
  assert.equal(pickAvcLevel(2560, 1440, 30), 0x33);
  assert.equal(pickAvcLevel(3840, 2160, 30), 0x34);
});

test("planEncoder: quality tiers size the chosen config", async () => {
  for (const [quality, expect] of [
    ["standard", 1_500_000],
    ["high", 3_000_000],
    ["maximum", 6_000_000],
  ]) {
    const seen = stubSupportAll();
    const plan = await planEncoder(480, 270, 30, true, "quality", null, quality);
    assert.equal(plan.videoSourceCodec, "avc");
    assert.equal(plan.container, "mp4");
    assert.equal(plan.tier, quality);
    assert.equal(plan.encoderConfig.bitrate, expect);
    assert.equal(plan.accel, "prefer-hardware");
  }
});

test("planEncoder: unknown quality throws before probing", async () => {
  const seen = stubSupportAll();
  await assert.rejects(
    () => planEncoder(480, 270, 30, true, "quality", null, "ultra"),
    /unknown encoder quality/,
  );
  assert.equal(seen.length, 0);
});

test("planEncoder: encoder-cap ladder degrades maximum → high → standard", async () => {
  // Reject every maximum-tier config (>= 6 Mbps at this size).
  const seen = stubSupportAll((c) => c.bitrate >= 6_000_000);
  const plan = await planEncoder(480, 270, 30, false, "quality", null, "maximum");
  assert.equal(plan.tier, "high");
  assert.equal(plan.encoderConfig.bitrate, 3_000_000);
  // First pass probed every candidate at the maximum bitrate before
  // falling back; the last probe (the accepted config) is at high.
  assert.ok(seen.some((c) => c.bitrate === 6_000_000));
  assert.equal(seen.at(-1).bitrate, 3_000_000);
});

test("planEncoder: standard rejection yields null", async () => {
  stubSupportAll(() => true);
  const plan = await planEncoder(480, 270, 30, true, "quality", null, "high");
  assert.equal(plan, null);
});

test("planEncoder: codec hint reorders", async () => {
  const seen = stubSupportAll();
  const plan = await planEncoder(480, 270, 30, true, "quality", "vp9", "high");
  assert.equal(plan.videoSourceCodec, "vp9");
  assert.equal(seen[0].codec, "vp09.00.10.08");
});

test("qualityLabel: match and degrade forms", () => {
  assert.equal(qualityLabel("high", "high"), "quality=high");
  assert.equal(qualityLabel("maximum", "standard"), "quality=maximum→standard (encoder cap)");
});

test("accelLabel covers all preferences", () => {
  assert.equal(accelLabel("prefer-hardware"), "hardware-preferred");
  assert.equal(accelLabel("prefer-software"), "software-forced");
  assert.equal(accelLabel("no-preference"), "browser-chosen");
});

test("QUALITY_LEVELS ladders are monotonic", () => {
  assert.deepEqual(Object.keys(QUALITY_LEVELS), ["standard", "high", "maximum"]);
  assert.ok(QUALITY_LEVELS.standard.bpp < QUALITY_LEVELS.high.bpp);
  assert.ok(QUALITY_LEVELS.high.bpp < QUALITY_LEVELS.maximum.bpp);
});
