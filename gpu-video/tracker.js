/**
 * Direction tracker for `auto` pairing — a pure-JS port of
 * `DirectionTracker` in anaglyph-rs `motion.rs`, exposing the same surface
 * the wasm `Tracker` export gives `js/pairing.js`:
 *
 *   push(index, timeSecs, dx, reliable)
 *   confirmedThrough() -> number   (-1 while nothing is confirmed)
 *   currentDirection() -> "clockwise" | "counter-clockwise"
 *   finish()
 *   takeReversals() -> number[]    packed quads [frame, time, fromBit, toBit]
 *
 * The integrated drift over reliable samples forms a zig-zag; its turning
 * points are the reversals. A turning point is only provable once the drift
 * retreats by `swing_frames * speed` (at least `min_drift`), so `confirmed
 * Through` tracks the current extremum candidate: frames up to it are final
 * even while the trend itself might still flip. Ported arithmetic is
 * line-for-line; unit tests mirror the Rust ones (`pending_cap` forcing,
 * cut commit, reversal placement).
 */

const SPEED_SMOOTHING = 0.1;

/** Tuning knobs; drift units are source pixels (TrackerConfig::default). */
export const TRACKER_CONFIG = {
  startFrames: 4.0,
  swingFrames: 8.0,
  minDrift: 4.0,
  pendingCap: 20,
};

/**
 * Map a drift trend onto an eye pairing (motion.rs `trend_to_direction`).
 * Convention: booth background drifts with the camera, so content drifting
 * right means the newer frame is the right-eye view (counter-clockwise);
 * drifting left gives clockwise; no trend falls back to clockwise.
 */
export function trendToDirection(trend, reverse) {
  const clockwiseForwards = trend === "increasing" ? false : true;
  return clockwiseForwards !== reverse ? "clockwise" : "counter-clockwise";
}

export class DirectionTracker {
  /** @param {boolean} reverse inverts the trend-to-pairing mapping */
  constructor(reverse = false, cfg = TRACKER_CONFIG) {
    this.cfg = cfg;
    this.reverse = reverse;
    this.drift = 0.0;
    this.speed = 0.0;
    /** @type {"increasing" | "decreasing" | null} */
    this.trend = null;
    // Current trend's extremum candidate: where the next reversal, if any,
    // will be placed.
    this.extValue = 0.0;
    this.extFrame = 0;
    this.extTime = 0.0;
    this.lastFrame = null;
    this.confirmed = null;
    /** @type {Array<{frame: number, time: number, from: string, to: string}>} */
    this.reversals = [];
  }

  /**
   * Record the sample that *ends* at `frame` (motion from frame-1 to frame)
   * at source time `timeSecs`. Frames must arrive in increasing order.
   */
  push(frame, timeSecs, dx, reliable) {
    this.lastFrame = frame;
    if (!reliable) {
      // A cut (or an unmatchable pair) says nothing about the trend, but
      // holding frames back for it buys nothing: commit the pending stretch
      // and carry the current trend across the cut.
      this.extFrame = frame;
      this.extTime = timeSecs;
      this.confirmed = frame;
      return;
    }
    this.drift += dx;
    this.speed =
      this.speed === 0.0
        ? Math.abs(dx)
        : this.speed * (1.0 - SPEED_SMOOTHING) + Math.abs(dx) * SPEED_SMOOTHING;
    if (this.trend === null) {
      // No direction yet: wait for enough drift to tell a slow start from
      // noise.
      const start = Math.max(this.cfg.startFrames * this.speed, this.cfg.minDrift);
      if (this.drift >= start) this.trend = "increasing";
      else if (this.drift <= -start) this.trend = "decreasing";
      else return;
      this.extValue = this.drift;
      this.extFrame = frame;
      this.extTime = timeSecs;
      this.confirmed = frame;
      return;
    }
    const madeNewExtremum =
      this.trend === "increasing"
        ? this.drift > this.extValue
        : this.drift < this.extValue;
    if (madeNewExtremum) {
      this.extValue = this.drift;
      this.extFrame = frame;
      this.extTime = timeSecs;
    } else if (
      Math.abs(this.drift - this.extValue) >=
      Math.max(this.cfg.swingFrames * this.speed, this.cfg.minDrift)
    ) {
      const to = this.trend === "increasing" ? "decreasing" : "increasing";
      this.reversals.push({
        frame: this.extFrame + 1,
        time: this.extTime,
        from: this.trend,
        to,
      });
      // The current frame starts the new trend's extremum candidate: drift
      // has been moving away from the turning point ever since.
      this.trend = to;
      this.extValue = this.drift;
      this.extFrame = frame;
      this.extTime = timeSecs;
    } else if (frame - this.extFrame > this.cfg.pendingCap) {
      // A long ambiguous stretch means near-zero motion, where pairing
      // hardly matters: commit it to bound latency. The extremum *value*
      // stays, so a later reversal still measures against the true turning
      // point.
      this.extFrame = frame;
      this.extTime = timeSecs;
    }
    this.confirmed =
      this.confirmed === null ? this.extFrame : Math.max(this.confirmed, this.extFrame);
  }

  /** End of stream: everything held back is final, on the current trend. */
  finish() {
    if (this.lastFrame !== null) {
      this.extFrame = this.lastFrame;
      this.confirmed = this.lastFrame;
    }
  }

  /** Last frame index whose trend assignment can no longer change. */
  confirmedThrough() {
    return this.confirmed ?? -1;
  }

  /** Eye pairing for the frames following `confirmedThrough()`. */
  currentDirection() {
    return trendToDirection(this.trend, this.reverse);
  }

  /**
   * Drain detected reversals as packed f64 quads
   * `[frame, time, fromBit, toBit]` (direction bits already mapped through
   * `reverse`: 1 = clockwise, 0 = counter-clockwise), matching the wasm
   * `Tracker.take_reversals()` payload `pairing.js` unpacks.
   */
  takeReversals() {
    const out = [];
    for (const r of this.reversals) {
      out.push(r.frame);
      out.push(r.time);
      out.push(trendToDirection(r.from, this.reverse) === "clockwise" ? 1 : 0);
      out.push(trendToDirection(r.to, this.reverse) === "clockwise" ? 1 : 0);
    }
    this.reversals = [];
    return out;
  }
}
