/**
 * Stereo pairing for temporal-stereo ("360 booth") videos.
 *
 * Consecutive decoded frames (in presentation order) form a stereo pair.
 * Which frame feeds which eye is set by the camera's orbit direction and
 * mirrors `Pairer::front_pixels` in anaglyph-rs:
 *
 * - `clockwise`: the newer frame is the left-eye view, the older the right.
 * - `counter-clockwise`: the older frame is the left-eye view.
 *
 * Kept free of WebCodecs/wasm imports so the logic is unit-testable in node.
 * Auto-direction tracking (M2) will plug the native `DirectionTracker` in
 * here by resolving `direction` per pair instead of at construction.
 */

/** Eye assignment for one pair; `left` / `right` are opaque frame objects. */
export function assignPair(previous, current, direction) {
  return direction === "counter-clockwise"
    ? { left: previous, right: current }
    : { left: current, right: previous };
}

/**
 * Streaming two-frame pairing window. `push` every decoded frame in
 * presentation order; every push after the first returns the completed pair.
 *
 * The output frame inherits the *older* frame's timestamp and the gap between
 * the two frames as duration, matching the native pipeline (which encodes the
 * pair under `front_time`).
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
   * @param {{timestamp: number, duration?: number}} frame Timestamps in
   * microseconds; frames must arrive in presentation order.
   * @returns {{left: object, right: object, timestamp: number, duration: number} | null}
   * the completed pair, or null while the window is still filling.
   */
  push(frame) {
    const previous = this._previous;
    this._previous = frame;
    if (!previous) return null;

    const duration =
      frame.timestamp > previous.timestamp
        ? frame.timestamp - previous.timestamp
        : (frame.duration ?? 0);
    return {
      ...assignPair(previous, frame, this.direction),
      timestamp: previous.timestamp,
      duration,
    };
  }

  /**
   * The last pushed frame while it is still waiting for its partner. The
   * owner must release it (close any underlying resources) at end of stream.
   */
  held() {
    return this._previous;
  }
}
