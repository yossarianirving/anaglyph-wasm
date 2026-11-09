/**
 * Encode-quality sweep: measures the three `encoderQuality` tiers
 * (standard/high/maximum, js/encoder-plan.js) through both pipelines on
 * both browsers, with hardware encode on and off. For every combination
 * it reports throughput, output size/bitrate, and — against the native
 * anaglyph-rs software-encode reference — PSNR and SSIM, so the quality
 * gain vs the throughput/size cost can be weighed per platform.
 *
 *   node bench/quality-sweep.mjs [--browser chrome|firefox]
 *                                [--page both|gpu|wasm]
 *                                [--hw both|on|off]
 *                                [--qualities standard,high,maximum]
 *                                [--iters N] [--clip path]
 *
 * Clips default per browser (boomerang: real content, blockiness is
 * visible; Firefox uses the VP9 WebM twin). Clockwise direction only —
 * auto adds motion-analysis noise to a pure encoder comparison. Stage
 * probes are skipped (stages=0); the sweep measures end-to-end pairs/s.
 *
 * Outputs + ground truth land in /tmp/opencode/qsweep/. The ground truth
 * is itself encoded (the CLI has no bitrate knob), so PSNR/SSIM measure
 * how much OUR encoder adds on top of the reference's own noise — which
 * is exactly the "re-encoding makes blockiness worse" effect.
 */
import { execFileSync, spawn, spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, firefox } from "playwright";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = 8134;
const OUT_DIR = "/tmp/opencode/qsweep";

const argValue = (name, fallback) => {
  const inline = process.argv.find((a) => a.startsWith(name + "="));
  if (inline !== undefined) return inline.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index > -1 ? process.argv[index + 1] : fallback;
};

const BROWSER = argValue("--browser", "chrome");
const PAGE = argValue("--page", "both");
const HW = argValue("--hw", "both");
const QUALITIES = argValue("--qualities", "standard,high,maximum").split(",");
const ITERS = parseInt(argValue("--iters", "2"), 10);
const NATIVE = join(root, "..", "anaglyph-rs", "target", "release", "anaglyph-rs");

if (!["chrome", "firefox"].includes(BROWSER)) {
  console.error("only --browser chrome|firefox supported");
  process.exit(2);
}
const clip =
  argValue("--clip", null) ??
  (BROWSER === "firefox" ? "bench/boomerang-vp9.webm" : "bench/boomerang.mp4");

const pages =
  PAGE === "both"
    ? [
        { label: "gpu", url: "gpu-video/bench.html" },
        { label: "wasm", url: "bench-video.html" },
      ]
    : PAGE === "gpu"
      ? [{ label: "gpu", url: "gpu-video/bench.html" }]
      : [{ label: "wasm", url: "bench-video.html" }];
const hwModes = HW === "both" ? [true, false] : [HW === "on"];

mkdirSync(OUT_DIR, { recursive: true });

/** Native CLI reference (clockwise, color type, software encode). */
const gtPath = join(
  OUT_DIR,
  `gt-${clip.split("/").pop()}.mp4`,
);
execFileSync(
  NATIVE,
  ["--video", join(root, clip), "--out", gtPath, "-d", "clockwise", "-t", "color",
    "-e", "software", "-b", "cpu"],
  { stdio: ["ignore", "pipe", "pipe"] },
);

const median = (v) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)];

/** ffmpeg PSNR average dB between file and ground truth. */
function psnrVs(file) {
  const r = spawnSync(
    "ffmpeg",
    ["-hide_banner", "-i", file, "-i", gtPath, "-lavfi", "psnr", "-f", "null", "-"],
    { encoding: "utf8", maxBuffer: 32 << 20 },
  );
  return parseFloat(/average:([0-9.inf]+)/.exec(r.stderr ?? "")?.[1] ?? "NaN");
}

/** ffmpeg SSIM "All" score between file and ground truth. */
function ssimVs(file) {
  const r = spawnSync(
    "ffmpeg",
    ["-hide_banner", "-i", file, "-i", gtPath, "-lavfi", "ssim", "-f", "null", "-"],
    { encoding: "utf8", maxBuffer: 32 << 20 },
  );
  return parseFloat(/All:([0-9.]+)/.exec(r.stderr ?? "")?.[1] ?? "NaN");
}

/** ffprobe average bitrate of the video stream (bps); 0 when absent. */
function videoBitrate(file) {
  const r = spawnSync(
    "ffprobe",
    ["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=bit_rate",
      "-of", "csv=p=0", file],
    { encoding: "utf8" },
  );
  return parseInt(r.stdout?.trim(), 10) || 0;
}

const browser =
  BROWSER === "firefox"
    ? await firefox.launch({
        headless: false,
        firefoxUserPrefs: {
          "dom.webgpu.enabled": true,
          "gfx.webrender.all": true,
        },
      })
    : await chromium.launch({
        channel: "chrome",
        headless: false,
        args: ["--ignore-gpu-blocklist"],
      });
const server = spawn(
  "python3",
  ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"],
  { cwd: root, stdio: "ignore" },
);

const rows = [];
try {
  for (const pageDef of pages) {
    for (const hw of hwModes) {
      for (const quality of QUALITIES) {
        const page = await browser.newPage();
        page.on("pageerror", (e) => console.log(`[pageerror] ${e}`));
        const query = new URLSearchParams({
          clip: pageDef.label === "gpu" ? `../${clip}` : clip,
          direction: "clockwise",
          iters: String(ITERS),
          autorun: "1",
          stages: "0",
          quality,
          ...(pageDef.label === "gpu" ? { capture: "bytes" } : {}),
          ...(hw ? {} : { hwEncode: "0" }),
        });
        const t0 = Date.now();
        let row;
        try {
          await page.goto(`http://localhost:${PORT}/${pageDef.url}?${query}`);
          await page.waitForFunction(() => window.__benchResults !== null, null, {
            timeout: 600_000,
          });
          const result = await page.evaluate(() => window.__benchResults);
          if (result.error) throw new Error(result.error);
          const fps = median(result.iters.map((r) => r.fps));
          const outPath = join(
            OUT_DIR,
            `${pageDef.label}-hw${hw ? "on" : "off"}-${quality}.out`,
          );
          const b64 = await page.evaluate(async () => {
            if (!window.__lastOutput) return null;
            const bytes = new Uint8Array(await window.__lastOutput.arrayBuffer());
            let s = "";
            for (let i = 0; i < bytes.length; i += 0x8000) {
              s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
            }
            return btoa(s);
          });
          if (!b64) throw new Error("page kept no output blob");
          const buf = Buffer.from(b64, "base64");
          writeFileSync(outPath, buf);
          row = {
            pipeline: pageDef.label,
            hw,
            quality,
            fps,
            bytes: buf.length,
            vbitrate: videoBitrate(outPath),
            psnr: psnrVs(outPath),
            ssim: ssimVs(outPath),
            encoder: result.iters[0]?.encoder ?? "?",
            seconds: (Date.now() - t0) / 1000,
          };
        } catch (err) {
          row = {
            pipeline: pageDef.label,
            hw,
            quality,
            error: String(err && err.message ? err.message : err),
          };
        }
        await page.close();
        rows.push(row);
        if (row.error) {
          console.log(`${pageDef.label} hw=${row.hw ? "on " : "off"} ${quality.padEnd(8)} ERROR ${row.error}`);
        } else {
          console.log(
            `${pageDef.label} hw=${row.hw ? "on " : "off"} ${quality.padEnd(8)} ` +
              `${row.fps.toFixed(0).padStart(5)} pairs/s  ${(row.bytes / 1e6).toFixed(2)} MB  ` +
              `${(row.vbitrate / 1e6).toFixed(2)} Mbps  PSNR ${row.psnr.toFixed(2)} dB  ` +
              `SSIM ${row.ssim.toFixed(4)}  [${row.encoder}]  (${row.seconds.toFixed(0)}s)`,
          );
        }
      }
    }
  }
} finally {
  await browser.close();
  server.kill();
}

writeFileSync(
  join(OUT_DIR, `sweep-${BROWSER}.json`),
  JSON.stringify({ browser: BROWSER, clip, qualities: QUALITIES, gt: gtPath, rows }, null, 2),
);
console.log(`\nwrote ${join(OUT_DIR, `sweep-${BROWSER}.json`)}`);
