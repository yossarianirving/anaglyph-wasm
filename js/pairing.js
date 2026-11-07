/**
 * Stereo pairing for temporal-stereo ("360 booth") videos.
 *
 * Consecutive decoded frames (in presentation order) form a stereo pair.
 * Which frame feeds which eye is set by the camera's orbit direction and
 * mirrors `Pairer` in anaglyph-rs:
 *
 * - `clockwise`: the newer frame is the left-eye view, the older the right.
 * - `counter-clockwise`: the older frame is the left-eye view.
 * - `auto`: the direction is read off the image motion (a wasm
 *   `DirectionTracker` decides, with hysteresis) and re-resolved per pair, so
 *   "boomerang" passes swap eyes at the confirmed turning points.
 *
 * Kept free of WebCodecs/wasm imports so the logic is unit-testable in node:
 * the auto strategy takes `observe` + `tracker` stubs with the same shape as
 * the wasm `MotionAnalyzer` / `Tracker` exports.
 */

/** Eye assignment for one pair; `left` / `right` are opaque frame objects. */
export function assignPair(previous, current, direction) {
  return direction === "counter-clockwise"
    ? { left: previous, right: current }
    : { left: current, right: previous };
}

/**
 * Completed pair from two consecutive frames: the output frame inherits the
 * *older* frame's timestamp and the gap between the two frames as duration,
 * matching the native pipeline (which encodes the pair under `front_time`).
 */
function makePair(older, newer, direction) {
  const duration =
    newer.timestamp > older.timestamp
      ? newer.timestamp - older.timestamp
      : (newer.duration ?? 0);
  return {
    ...assignPair(older, newer, direction),
    timestamp: older.timestamp,
    duration,
  };
}

/**
 * Streaming two-frame pairing window. `push` every decoded frame in
 * presentation order; every push after the first returns the completed pair.
 */
export class FramePairer {
  /**
   * @param {"clockwise" | "counter-clockwise"} direction
   */
  constructor(direction = "clockwise") {
    if (direction !== "clockwise" && direction !== "counter-clockwise") {
      throw new Error(`unknown direction: ${direction}`);
    }
    this.direction = direction;
    /** @type {object | null} last pushed frame */
    this._previous = null;
  }

  /**
   * Register the next decoded frame.
   *
   * @param {{index: number, timestamp: number, duration?: number}} frame
   * Timestamps in microseconds; frames must arrive in presentation order.
   * @returns {{left: object, right: object, timestamp: number, duration: number} | null}
   * the completed pair, or null while the window is still filling.
   */
  push(frame) {
    const previous = this._previous;
    this._previous = frame;
    if (!previous) return null;
    return makePair(previous, frame, this.direction);
  }

  /**
   * The last pushed frame while it is still waiting for its partner. The
   * owner must release it (close any underlying resources) at end of stream.
   */
  held() {
    return this._previous;
  }
}

/**
 * Uniform pairing interface used by the video pipeline. A strategy consumes
 * every decoded frame via `push` and hands back the pairs that are ready to
 * be blended and encoded:
 *
 *   { pairs, releasedBefore, reversals }
 *
 * - `pairs`: completed pairs (left/right eyes assigned).
 * - `releasedBefore`: frame indices strictly below it will never be part of
 *   another pair; the pipeline frees their pixel buffers. -1 for "nothing".
 * - `reversals`: auto-mode direction flips confirmed since the last call
 *   (empty for fixed pairing).
 *
 * `flush()` returns the pairs the trailing frame completes at end of stream.
 * Frames still held after `flush()` (the last, unpaired frame) are released
 * by the pipeline's cleanup.
 */
export function createPairStrategy(direction, deps) {
  if (direction === "auto" || direction === "auto-reverse") {
    if (!deps || !deps.tracker) {
      throw new Error(`direction "${direction}" needs a tracker (deps.tracker)`);
    }
    return new AutoFramePairer({
      observe: deps.observe ?? (() => ({ dx: 0, reliable: false })),
      tracker: deps.tracker,
    });
  }
  return new FixedPairStrategy(direction);
}

/** Fixed-direction pairing: every push completes a pair immediately. */
class FixedPairStrategy {
  /**
   * @param {"clockwise" | "counter-clockwise"} direction
   */
  constructor(direction) {
    this.pairer = new FramePairer(direction);
  }

  get auto() {
    return false;
  }

  /** @param {{index: number, timestamp: number, duration?: number}} frame */
  push(frame) {
    const pair = this.pairer.push(frame);
    return {
      pairs: pair ? [pair] : [],
      // The frame two back is consumed once its pair completes; in a
      // two-frame window that is `frame.index - 1`.
      releasedBefore: pair ? frame.index : -1,
      reversals: [],
    };
  }

  flush() {
    return { pairs: [], releasedBefore: -1, reversals: [] };
  }

  held() {
    return this.pairer.held();
  }

  maxPending() {
    return 2;
  }
}

/**
 * Auto-direction pairing: mirrors the native `Pairer::Auto`. Each frame is
 * observed for horizontal drift and reported to the tracker; pairs stay held
 * back until the tracker's `confirmedThrough()` covers their *newer* frame
 * (all frames up to it are then guaranteed to keep their trend), and every
 * drained pair is labelled with the tracker's current direction. This is
 * eager like the native drain loop (`while pairer.ready()`): frames only wait
 * as long as the hysteresis needs, bounded by the tracker's `pending_cap`.
 */
export class AutoFramePairer {
  /**
   * @param {{observe: (frame: object) => {dx: number, reliable: boolean},
   *          tracker: {push(index: number, timeSecs: number, dx: number,
   *            reliable: boolean): void, confirmedThrough(): number,
   *            currentDirection(): string, finish(): void,
   *            takeReversals(): Float64Array | number[]}}} deps
   *   `tracker` matches the wasm `Tracker` export (see `unpackReversals`).
   */
  constructor({ observe, tracker }) {
    this._observe = observe ?? (() => ({ dx: 0, reliable: false }));
    this._tracker = tracker;
    /** @type {object[]} consecutive frames not fully drained yet */
    this._window = [];
    this.maxPending = 0;
  }

  get auto() {
    return true;
  }

  /**
   * Observe the frame's drift, update the tracker, and drain every pair the
   * tracker has now confirmed.
   *
   * @param {{index: number, timestamp: number, duration?: number}} frame
   */
  push(frame) {
    const { dx, reliable } = this._observe(frame);
    this._tracker.push(frame.index, frame.timestamp / 1e6, dx, reliable);
    this._window.push(frame);
    this.maxPending = Math.max(this.maxPending, this._window.length);
    return this._drain();
  }

  /** End of stream: commit the held-back stretch and drain everything. */
  flush() {
    this._tracker.finish();
    return this._drain();
  }

  /** Frames whose buffers the pipeline has not released yet. */
  held() {
    return this._window;
  }

  _drain() {
    const confirmed = this._tracker.confirmedThrough();
    // Pair (w[i], w[i+1]) may be emitted while w[i+1].index <= confirmed.
    let n = 0;
    while (n + 1 < this._window.length && this._window[n + 1].index <= confirmed) {
      n += 1;
    }
    const direction = this._tracker.currentDirection();
    const pairs = [];
    for (let i = 0; i < n; i++) {
      pairs.push(makePair(this._window[i], this._window[i + 1], direction));
    }
    // The newest drained frame starts the next pair: keep it and everything
    // after it; strictly-older frames are done.
    const releasedBefore =
      n > 0 ? this._window[n].index : -1;
    this._window = this._window.slice(n);
    return {
      pairs,
      releasedBefore,
      reversals: unpackReversals(this._tracker.takeReversals()),
    };
  }
}

/**
 * Unpack the wasm `Tracker.take_reversals()` payload: packed f64 quads
 * `[frame, time_secs, from_clockwise, to_clockwise]` (direction slots are 1
 * for clockwise, 0 for counter-clockwise, already mapped through `reverse`).
 *
 * @param {ArrayLike<number>} packed
 * @returns {Array<{frame: number, time: number, from: string, to: string}>}
 */
export function unpackReversals(packed) {
  const names = ["counter-clockwise", "clockwise"];
  const out = [];
  for (let i = 0; i + 3 < packed.length; i += 4) {
    out.push({
      frame: packed[i],
      time: packed[i + 1],
      from: names[packed[i + 2]] ?? "unknown",
      to: names[packed[i + 3]] ?? "unknown",
    });
  }
  return out;
}
