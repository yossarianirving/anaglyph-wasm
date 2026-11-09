/**
 * Driver for bench/probe-motion.html: GpuMotion-only throughput (no blend,
 * no encode) for a clip at several batch sizes.
 *
 *   node bench/probe-motion.mjs [clip] [batchK...]
 */
import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = 8131;
const args = process.argv.slice(2);
const clip = args[0] ?? "bench/boomerang.mp4";
const batches = args.slice(1).map(Number);
const batchesQ = batches.length ? batches : [8];

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
try {
  const page = await browser.newPage();
  for (const batchK of batchesQ) {
    const query = new URLSearchParams({
      clip: `../${clip}`,
      batch: String(batchK),
    });
    await page.goto(
      `http://localhost:${PORT}/bench/probe-motion.html?${query}`,
    );
    await page.waitForFunction(() => window.__result !== null, null, {
      timeout: 600_000,
    });
    const r = await page.evaluate(() => window.__result);
    if (r.error) throw new Error(`batch ${batchK}: ${r.error}`);
    console.log(
      `K ${String(batchK).padStart(2)}: ${r.fps.toFixed(0)} frames/s ` +
        `(submit ${r.submitCpuMs.toFixed(0)} ms, next ${r.nextCpuMs.toFixed(0)} ms, ` +
        `waits ${r.waitN} × ${(r.waitMs / Math.max(1, r.waitN)).toFixed(1)} ms = ${(r.waitMs / r.frames).toFixed(2)} ms/frame, ` +
        `map ${(r.mapMs / Math.max(1, r.mapN)).toFixed(1)} ms avg over ${r.mapN})`,
    );
  }
} finally {
  await browser.close();
  server.kill();
}
