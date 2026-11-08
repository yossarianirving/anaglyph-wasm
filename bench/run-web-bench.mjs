/**
 * Benchmark runner: the browser pipeline (headless Chromium, Firefox, or
 * GNOME Web/WebKitGTK, via bench-video.html) plus the native CLI, over a
 * fixed clip matrix.
 *
 * Usage: node bench/run-web-bench.mjs [--iters N] [--gpu] [--skip-native]
 *                                     [--browser chromium|firefox|webkit]
 *                                     [--latency quality|realtime] [--hw off]
 *                                     [--page bench-video.html]
 *                                     [--capture-bytes]
 *
 * `--page` selects the bench page under test; point it at
 * `gpu-video/bench.html` to benchmark the all-GPU pipeline (run both pages
 * in one command sequence to A/B against the wasm pipeline). On GPU pages
 * the backend column shows the encoder capture path, and `--capture-bytes`
 * adds the readback-encode variants next to the zero-copy canvas ones.
 *
 * Results land in bench/results/web-<timestamp>.json (and
 * bench/results/latest.md as a markdown summary). Read them with
 * `node bench/compare.mjs a.json b.json` to diff two runs.
 *
 * Firefox runs open-codec WebM clips: Playwright's Firefox build carries no
 * proprietary H.264/MP4 codecs, and the WebM set keeps the two browsers
 * comparable stage-for-stage (software codecs on both sides unless the host
 * exposes hardware ones to the browser).
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
import { chromium, firefox } from "playwright";

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
const BROWSER = argValue("--browser", "chromium");
const LATENCY = argValue("--latency", "quality");
// Restrict every run to one codec family ("avc" | "vp9" | "vp8" | "av1").
const CODEC = argValue("--codec", null);
// `--hw off` disables hardware decode AND encode preferences.
const HW = argValue("--hw", "on");
// Which bench page to drive (the GPU pipeline lives on its own page).
const PAGE = argValue("--page", "bench-video.html");
const GPU_PAGE = PAGE.startsWith("gpu-video");

// Firefox (and any browser compared against it) uses the open-codec WebM
// set; `--webm` forces the same set on every browser for apples-to-apples
// cross-browser runs. WebKitGTK decodes VP9/VP8 through GStreamer.
const WEBM_CLIPS = ["bench/booth2s-vp9.webm", "bench/boomerang-vp9.webm"];
const CLIPS =
  BROWSER === "firefox" || BROWSER === "webkit" || process.argv.includes("--webm")
    ? WEBM_CLIPS
    : ["bench/booth2s.mp4", "bench/booth2s-aac.mp4", "bench/boomerang.mp4"];

/** Web runs: CPU blend everywhere; `auto` direction costs motion analysis. */
const webRuns = [];
for (const clip of CLIPS) {
  if (GPU_PAGE) {
    // The all-GPU pipeline: capture path replaces the blend backend, and
    // it always runs its GPU variant (no CPU blend exists).
    webRuns.push({ clip, direction: "clockwise", backend: "capture=canvas", capture: "canvas" });
    if (!clip.includes("aac")) {
      webRuns.push({ clip, direction: "auto", backend: "capture=canvas", capture: "canvas" });
    }
    if (process.argv.includes("--capture-bytes")) {
      webRuns.push({ clip, direction: "clockwise", backend: "capture=bytes", capture: "bytes" });
      if (!clip.includes("aac")) {
        webRuns.push({ clip, direction: "auto", backend: "capture=bytes", capture: "bytes" });
      }
    }
    continue;
  }
  webRuns.push({ clip, direction: "clockwise", backend: "cpu" });
  if (!clip.includes("aac")) webRuns.push({ clip, direction: "auto", backend: "cpu" });
}
if (WITH_GPU && !GPU_PAGE) {
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

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Open a "page" that can run bench-video.html with autorun and hand back
 * `window.__benchResults`: Playwright for Chromium/Firefox, WebKitWebDriver
 * (GNOME Web, the iOS Safari engine family) for webkit. The classic WebDriver
 * protocol cannot await promises, so the WebKit shim polls instead.
 */
async function openPage() {
  if (BROWSER !== "webkit") {
    const browser =
      BROWSER === "firefox"
        ? await firefox.launch({ headless: true })
        : await chromium.launch({
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
    return {
      async runBench(url, timeoutMs) {
        await page.goto(url);
        await page.waitForFunction(() => window.__benchResults !== null, null, {
          timeout: timeoutMs,
        });
        return page.evaluate(() => window.__benchResults);
      },
      close: () => browser.close(),
    };
  }

  const DRIVER = "http://127.0.0.1:9922";
  const req = (method, path, body, timeoutMs = 30000) =>
    fetch(DRIVER + path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(timeoutMs),
    }).then((r) => r.json());

  let driverProc = null;
  try {
    await req("GET", "/status", undefined, 1500);
  } catch {
    driverProc = spawn("WebKitWebDriver", ["--host=127.0.0.1", "--port=9922", "--replace-on-new-session"], {
      stdio: "ignore",
    });
    for (let i = 0; i < 80; i++) {
      try {
        await req("GET", "/status", undefined, 1000);
        break;
      } catch {
        await sleep(250);
      }
    }
  }
  const sess = await req("POST", "/session", {
    capabilities: { alwaysMatch: {}, firstMatch: [{}] },
  });
  if (!sess.value?.sessionId) throw new Error(`WebKitWebDriver session failed: ${JSON.stringify(sess)}`);
  const sid = sess.value.sessionId;
  return {
    async runBench(url, timeoutMs) {
      await req("POST", `/session/${sid}/url`, { url }, 120000);
      const deadline = Date.now() + timeoutMs;
      while (Date.now() < deadline) {
        const r = await req(
          "POST",
          `/session/${sid}/execute/sync`,
          {
            script: "return window.__benchResults ? JSON.stringify(window.__benchResults) : null;",
            args: [],
          },
          20000,
        );
        if (typeof r.value === "string") return JSON.parse(r.value);
        await sleep(400);
      }
      throw new Error("bench page never produced results");
    },
    async close() {
      await req("DELETE", `/session/${sid}`, undefined, 5000).catch(() => {});
      driverProc?.kill();
    },
  };
}

async function waitForServer(retries = 30) {
  for (let i = 0; i < retries; i++) {
    try {
      await fetch(`${BASE}/${PAGE}`);
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

let page;
try {
  await waitForServer();
  page = await openPage();

  const web = [];
  for (const run of webRuns) {
    const query = new URLSearchParams({
      ...run,
      iters: String(ITERS),
      autorun: "1",
      latency: LATENCY,
      ...(GPU_PAGE ? { capture: run.capture } : {}),
      ...(CODEC ? { codec: CODEC } : {}),
      ...(HW === "off" ? { hwDecode: "0", hwEncode: "0" } : {}),
    });
    process.stdout.write(`web ${run.clip} ${run.direction}/${run.backend} … `);
    let result;
    try {
      result = await page.runBench(`${BASE}/${PAGE}?${query}`, 600_000);
    } catch (err) {
      result = { clip: run.clip, direction: run.direction, backend: run.backend, error: String(err) };
    }
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
    // The native CLI comparison always runs on the reference MP4 set so the
    // column means the same thing for every --browser choice.
    native = [];
    for (const clip of ["bench/booth2s.mp4", "bench/boomerang.mp4"]) {
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
    schema: 2,
    date: new Date().toISOString(),
    iters: ITERS,
    browser: BROWSER,
    latency: LATENCY,
    hw: HW,
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
      medianStageMs: run.iters.length
        ? Object.fromEntries(
            ["copy", "blend", "capture", "motion", "encodeWait", "decodeWait"].map((k) => [
              k,
              median(run.iters.map((r) => r.stageMs?.[k] ?? 0)),
            ]),
          )
        : null,
      iterations: run.iters,
      capture: run.iters[0]?.capture ?? null,
      decodeOnly: run.decodeOnly,
      blendOnly: run.blendOnly,
      encodeOnly: run.encodeOnly,
      motionOnly: run.motionOnly,
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
  await page?.close();
  server.kill();
}

function renderMarkdown(payload) {
  const lines = [];
  lines.push(`# Web video benchmark — ${payload.date.slice(0, 10)}`);
  lines.push("");
  lines.push(
    `${payload.iters} iterations per run (median shown); ` +
      (payload.headless
        ? `headless ${payload.browser} (software codecs; WebGPU adapter per meta).`
        : "headed/hardware run.") +
      ` Encoder latency: ${payload.latency}, hardware preference: ${payload.hw}.`,
  );
  lines.push(`Browser: ${payload.webMeta?.userAgent ?? "unknown"}`);
  lines.push(`WebGPU: ${payload.webMeta?.webgpu ?? "unknown"}`);
  lines.push("");
  lines.push("| clip | direction | backend | blend backend | median pairs/s | copy ms | blend ms | motion ms | enc-wait ms | dec-wait ms | decode-only fps | encode-only fps | blend-only fps | motion-only fps |");
  lines.push("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
  for (const run of payload.web) {
    if (run.error) {
      lines.push(`| ${run.clip} | ${run.direction} | ${run.backend} | — | ERROR: ${run.error} | — | — | — | — | — | — | — | — | — |`);
      continue;
    }
    const s = run.medianStageMs ?? {};
    lines.push(
      `| ${run.clip} | ${run.direction} | ${run.backend} | ${run.blender ?? "?"} | ` +
        `${run.medianFps.toFixed(1)} | ` +
        `${Math.round(s.copy ?? 0)} | ${Math.round(s.blend ?? 0)} | ${Math.round(s.motion ?? 0)} | ` +
        `${Math.round(s.encodeWait ?? 0)} | ${Math.round(s.decodeWait ?? 0)} | ` +
        `${run.decodeOnly ? run.decodeOnly.fps.toFixed(1) : "—"} | ` +
        `${run.encodeOnly && !run.encodeOnly.error ? run.encodeOnly.fps.toFixed(1) : "—"} | ` +
        `${run.blendOnly ? run.blendOnly.fps.toFixed(1) : "—"} | ` +
        `${run.motionOnly && !run.motionOnly.error ? run.motionOnly.fps.toFixed(1) : "—"} |`,
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
