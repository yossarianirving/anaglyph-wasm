/**
 * Does Firefox's VideoEncoder RGB->YUV round trip spill green on extreme
 * chroma content — and does the WASM pipeline show it too?
 *
 * Feeds a solid RGB(222,0,221) clip (green channel = 0, chroma at the
 * gamut boundary) through both browser pipelines on Firefox with the
 * default `color` blend, decodes the outputs back to RGB with ffmpeg,
 * and reports per-channel means. If green is lifted ~30 for BOTH the
 * wasm and GPU outputs, the spill lives in Firefox's encoder and the
 * GPU pipeline merely inherits it.
 */
import { spawn, spawnSync } from "node:child_process";
import { writeFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { firefox } from "playwright";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = 8129;
const BASE = `http://localhost:${PORT}`;
const CLIP = "/tmp/opencode/solid-magenta.mp4";
const OUT = "/tmp/opencode/greenspill";

function sh(cmd, args) {
  const r = spawnSync(cmd, args, { encoding: "utf8" });
  if (r.status !== 0) {
    throw new Error(`${cmd} failed (${r.status}):\n${r.stderr}`);
  }
  return r;
}

if (!existsSync(CLIP)) {
  sh("ffmpeg", [
    "-v", "error", "-y",
    "-f", "lavfi", "-i", "color=c=0xDE00DD:s=64x64:d=0.6:r=30",
    "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
    CLIP,
  ]);
}

async function runPage(browser, pagePath, outFile) {
  const context = await browser.newContext({ acceptDownloads: true });
  const page = await context.newPage();
  await page.goto(`${BASE}${pagePath}`, { waitUntil: "domcontentloaded" });
  await page.setInputFiles("#videoFile", CLIP);
  await page.waitForFunction(() => !document.getElementById("submit").disabled);
  await page.click("#submit");
  await page.waitForFunction(
    () =>
      document.getElementById("result").style.display === "block" ||
      document.getElementById("status").textContent.startsWith("Failed"),
    { timeout: 240_000 },
  );
  const status = await page.textContent("#status");
  if (status.startsWith("Failed")) throw new Error(`${pagePath}: ${status}`);
  const bytes = await page.evaluate(async () => {
    const response = await fetch(document.getElementById("download").href);
    const buffer = await response.arrayBuffer();
    let binary = "";
    const chunk = 0x8000;
    const view = new Uint8Array(buffer);
    for (let i = 0; i < view.length; i += chunk) {
      binary += String.fromCharCode(...view.subarray(i, i + chunk));
    }
    return btoa(binary);
  });
  writeFileSync(outFile, Buffer.from(bytes, "base64"));
  await context.close();
  return outFile;
}

function greenMean(file) {
  sh("ffmpeg", ["-v", "error", "-y", "-i", file, "-frames:v", "1", `${file}.ppm`]);
  const text = `${file}.ppm`;
  const buf = require("node:fs").readFileSync(text);
  const m = /^P6\s+(\d+)\s+(\d+)\s+255\s/.exec(buf.toString("latin1", 0, 64));
  const px = buf.subarray(m[0].length);
  let sum = [0, 0, 0];
  for (let i = 0; i < px.length; i++) sum[i % 3] += px[i];
  const n = px.length / 3;
  return sum.map((v) => +(v / n).toFixed(1));
}

const browser = await firefox.launch({
  headless: false,
  firefoxUserPrefs: {
    "dom.webgpu.enabled": true,
    "gfx.webrender.all": true,
  },
});
const server = spawn(
  "python3",
  ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"],
  { cwd: root, stdio: "ignore" },
);
try {
  for (const [name, pagePath] of [
    ["wasm", "/video.html"],
    ["gpu", "/gpu-video/video.html"],
  ]) {
    const file = join(OUT, `${name}-solid.mp4`);
    await runPage(browser, pagePath, file);
    const tags = spawnSync(
      "ffprobe",
      ["-v", "error", "-select_streams", "v:0", "-show_entries",
       "stream=color_range,color_space,color_matrix,color_primaries",
       "-of", "default=noprint_wrappers=1", file],
      { encoding: "utf8" },
    ).stdout.trim().replace(/\n/g, " ");
    console.log(`${name}: rgb mean ${greenMean(file)} | ${tags}`);
  }
} finally {
  await browser.close();
  server.kill();
}
