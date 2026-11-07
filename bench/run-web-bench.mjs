/**
 * Benchmark runner: the browser pipeline (headless Chromium via
 * bench-video.html) plus the native CLI, over a fixed clip matrix.
 *
 * Usage: node bench/run-web-bench.mjs [--iters N] [--gpu] [--skip-native]
 *
 * Results land in bench/results/web-<timestamp>.json (and
 * bench/results/latest.md as a markdown summary). Read them with
 * `node bench/compare.mjs a.json b.json` to diff two runs.
 *
 * Headless caveat: unless you run this where Chromium can use real hardware
 * (`--gpu` plus a desktop GPU with WebGPU and hardware codecs), the browser
 * rows measure software decode/encode and SwiftShader-or-CPU blending, while
 * the native rows use whatever ffmpeg found (often VAAPI). Numbers are then
 * stage-shape comparisons, not hardware-vs-hardware. On real hardware, run
 * both and quote the meta blocks.
 */
import { execFileSync, spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = 8124;
const BASE = `http://localhost:${PORT}`;
const NATIVE = join(root, "..", "anaglyph-rs", "target", "release", "anaglyph-rs");

const argValue = (name, fallback) => {
  const index = process.argv.indexOf(name);
  return index > -1 ? process.argv[index + 1] : fallback;
};
const ITERS = parseInt(argValue("--iters", "3"), 10);
const WITH_GPU = process.argv.includes("--gpu");
const SKIP_NATIVE = process.argv.includes("--skip-native");

const CLIPS = ["bench/booth2s.mp4", "bench/booth2s-aac.mp4", "bench/boomerang.mp4"];

/** Web runs: CPU blend everywhere; `auto` direction costs motion analysis. */
const webRuns = [];
for (const clip of CLIPS) {
  webRuns.push({ clip, direction: "clockwise", backend: "cpu" });
  if (!clip.includes("aac")) webRuns.push({ clip, direction: "auto", backend: "cpu" });
}
if (WITH_GPU) {
  // The interesting comparison on real hardware: WebGPU blend vs CPU blend.
  for (const run of [...webRuns]) {
    webRuns.push({ ...run, backend: "auto" });
  }
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

async function waitForServer(retries = 30) {
  for (let i = 0; i < retries; i++) {
    try {
      await fetch(`${BASE}/bench-video.html`);
      return;
    } catch {
      await new Promise((r) => setTimeout(r, 200));
    }
  }
  throw new Error("static server did not come up");
}

function runNative(clip, direction) {
  const out = `/tmp/opencode/bench-native-${direction}-${clip.split("/").pop()}`;
  const started = process.hrtime.bigint();
  const stdout = execFileSync(
    NATIVE,
    ["--video", join(root, clip), "--out", out, "-d", direction, "-b", "cpu"],
    { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] },
  );
  const seconds = Number(process.hrtime.bigint() - started) / 1e9;
  const frames = Number(/encoded (\d+) frames/.exec(stdout)?.[1] ?? 0);
  return { frames, seconds, fps: seconds > 0 ? frames / seconds : null, output: out };
}

const server = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"], {
  cwd: root,
  stdio: "ignore",
});

let browser;
try {
  await waitForServer();
  browser = await chromium.launch({
    headless: true,
    // --gpu: full chromium (headless=new) without SwiftShader pinning, so a
    // machine with a real GPU actually uses it; the default headless shell
    // cannot reach WebGPU/hardware codecs at all.
    ...(WITH_GPU ? { channel: "chromium" } : {}),
    args: WITH_GPU
      ? ["--ignore-gpu-blocklist"]
      : ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
  });
  const page = await browser.newPage();
  page.on("pageerror", (err) => console.error(`[pageerror] ${err.message}`));

  const web = [];
  for (const run of webRuns) {
    const query = new URLSearchParams({ ...run, iters: String(ITERS), autorun: "1" });
    process.stdout.write(`web ${run.clip} ${run.direction}/${run.backend} … `);
    await page.goto(`${BASE}/bench-video.html?${query}`);
    await page.waitForFunction(() => window.__benchResults !== null, null, {
      timeout: 600_000,
    });
    const result = await page.evaluate(() => window.__benchResults);
    if (result.error) {
      console.log(`ERROR: ${result.error}`);
    } else {
      const fps = median(result.iters.map((r) => r.fps));
      console.log(`${fps.toFixed(1)} pairs/s (median of ${result.iters.length})`);
    }
    web.push(result);
  }

  let native = null;
  if (!SKIP_NATIVE) {
    native = [];
    for (const clip of CLIPS) {
      for (const direction of ["clockwise", "auto"]) {
        process.stdout.write(`native ${clip} ${direction} … `);
        const result = runNative(clip, direction);
        console.log(`${result.fps.toFixed(1)} frames/s (${result.frames} frames, wall)`);
        native.push({ clip, direction, ...result });
      }
    }
  }

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const resultsDir = join(root, "bench", "results");
  mkdirSync(resultsDir, { recursive: true });
  const jsonPath = join(resultsDir, `web-${stamp}.json`);
  const payload = {
    schema: 1,
    date: new Date().toISOString(),
    iters: ITERS,
    headless: !WITH_GPU,
    webMeta: web[0]?.meta ?? null,
    web: web.map((run) => ({
      clip: run.clip,
      direction: run.direction,
      backend: run.backend,
      blender: run.iters[0]?.blender ?? null,
      encoder: run.iters[0]?.encoder ?? null,
      decoder: run.iters[0]?.decoder ?? null,
      medianFps: run.iters.length ? median(run.iters.map((r) => r.fps)) : null,
      iterations: run.iters,
      decodeOnly: run.decodeOnly,
      blendOnly: run.blendOnly,
      error: run.error,
    })),
    native,
  };
  writeFileSync(jsonPath, JSON.stringify(payload, null, 2));
  console.log(`wrote ${jsonPath}`);

  const markdown = renderMarkdown(payload);
  writeFileSync(join(resultsDir, "latest.md"), markdown);
  console.log(`wrote ${join(resultsDir, "latest.md")}\n\n${markdown}`);
} finally {
  await browser?.close();
  server.kill();
}

function renderMarkdown(payload) {
  const lines = [];
  lines.push(`# Web video benchmark — ${payload.date.slice(0, 10)}`);
  lines.push("");
  lines.push(
    `${payload.iters} iterations per run (median shown); ` +
      (payload.headless
        ? "headless Chromium (software codecs; WebGPU adapter per meta)."
        : "headed/hardware run."),
  );
  lines.push(`Browser: ${payload.webMeta?.userAgent ?? "unknown"}`);
  lines.push(`WebGPU: ${payload.webMeta?.webgpu ?? "unknown"}`);
  lines.push("");
  lines.push("| clip | direction | backend | blend backend | median pairs/s | decode-only fps | blend-only fps |");
  lines.push("| --- | --- | --- | --- | --- | --- | --- |");
  for (const run of payload.web) {
    if (run.error) {
      lines.push(`| ${run.clip} | ${run.direction} | ${run.backend} | — | ERROR: ${run.error} | — | — |`);
      continue;
    }
    lines.push(
      `| ${run.clip} | ${run.direction} | ${run.backend} | ${run.blender ?? "?"} | ` +
        `${run.medianFps.toFixed(1)} | ${run.decodeOnly ? run.decodeOnly.fps.toFixed(1) : "—"} | ` +
        `${run.blendOnly ? run.blendOnly.fps.toFixed(1) : "—"} |`,
    );
  }
  if (payload.native) {
    lines.push("");
    lines.push("Native CLI (`-b cpu`, ffmpeg-selected encoder, wall-clock):");
    lines.push("");
    lines.push("| clip | direction | frames/s | frames |");
    lines.push("| --- | --- | --- | --- |");
    for (const run of payload.native) {
      lines.push(`| ${run.clip} | ${run.direction} | ${run.fps.toFixed(1)} | ${run.frames} |`);
    }
  }
  lines.push("");
  return lines.join("\n");
}
