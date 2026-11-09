/**
 * Sweep the pipelined-emit FIFO depth on Chrome (headed) for one clip and
 * direction, via gpu-video/bench.html?captureDepth=N. Reports median
 * pairs/s per depth so the auto-mode depth choice can be tuned with data.
 *
 *   node bench/probe-depth.mjs [clip] [direction] [depths...] [key=value...]
 *   e.g. node bench/probe-depth.mjs bench/boomerang.mp4 auto 16 8 4 2 1
 *        node bench/probe-depth.mjs bench/boomerang.mp4 auto 16 motionBatch=16
 */
import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = 8130;
const args = process.argv.slice(2);
const clip = args[0] ?? "bench/boomerang.mp4";
const direction = args[1] ?? "auto";
const rest = args.slice(2);
const depthsQ = rest.filter((a) => /^\d+$/.test(a)).map(Number);
const extras = rest.filter((a) => a.includes("="));
const extrasQuery = Object.fromEntries(extras.map((a) => a.split(/=(.+)/)));
const ITERS = 3;

const browser = await chromium.launch({
  channel: "chrome",
  headless: false,
  args: ["--ignore-gpu-blocklist"],
});
const server = spawn(
  "python3",
  ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"],
  { cwd: root, stdio: "ignore" },
);
const median = (v) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)];
try {
  const page = await browser.newPage();
  for (const depth of depthsQ.length ? depthsQ : [16]) {
    const query = new URLSearchParams({
      clip: `../${clip}`,
      direction,
      iters: String(ITERS),
      autorun: "1",
      capture: "bytes",
      latency: "quality",
      captureDepth: String(depth),
      ...extrasQuery,
    });
    const t0 = Date.now();
    await page.goto(`http://localhost:${PORT}/gpu-video/bench.html?${query}`);
    await page.waitForFunction(() => window.__benchResults !== null, null, {
      timeout: 600_000,
    });
    const result = await page.evaluate(() => window.__benchResults);
    if (result.error) throw new Error(`depth ${depth}: ${result.error}`);
    const fps = median(result.iters.map((r) => r.fps));
    const motion = median(result.iters.map((r) => r.stageMs?.motion ?? 0));
    const wakes = median(result.iters.map((r) => r.stageMs?.pumpWakes ?? 0));
    const waits = median(result.iters.map((r) => r.stageMs?.pumpWaits ?? 0));
    const paired = median(result.iters.map((r) => r.stageMs?.pumpPaired ?? 0));
    console.log(
      `depth ${String(depth).padStart(2)}${extras.length ? " " + extras.join(",") : ""}: ${fps.toFixed(1)} pairs/s  motion ${motion.toFixed(0)} ms  wakes ${wakes} waits ${waits} paired ${paired}  (${((Date.now() - t0) / 1000).toFixed(0)}s)`,
    );
  }
} finally {
  await browser.close();
  server.kill();
}
