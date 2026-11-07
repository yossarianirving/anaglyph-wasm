/**
 * End-to-end tests for the browser video pipeline.
 *
 * Serves the repo, drives video.html in headless Chromium across the sample
 * clips in bench/, and validates each result with ffprobe (codec, dimensions,
 * frame count, audio streams, rotation metadata).
 *
 * Usage: node bench/e2e-video.mjs [name-filter] [--headed] [--gpu]
 *
 * Headless Chromium runs software codecs and (depending on the build) has no
 * WebGPU adapter, so throughput here is a correctness signal only;
 * performance comes from bench-video.html on real hardware.
 *
 * Note on frame counts: booth2s.mp4 marks one trailing packet discardable;
 * ffmpeg's decoder drops it (native CLI: N-2 pairs) while WebCodecs decodes
 * it (browser: N-1 pairs). The extra frame is appended at the end and does
 * not shift any earlier pairing.
 */
import { execFileSync, spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = 8123;
const BASE = `http://localhost:${PORT}`;
const sample = (name) => join(root, "bench", name);

/** ffprobe a file, returning a compact multi-line summary (streams + video side data). */
function probe(file) {
  const streams = execFileSync(
    "ffprobe",
    [
      "-v",
      "error",
      "-count_frames",
      "-show_entries",
      "stream=index,codec_type,codec_name,width,height,nb_read_frames",
      "-of",
      "compact",
      file,
    ],
  ).toString();
  const side = execFileSync(
    "ffprobe",
    [
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-read_intervals",
      "%+#1",
      "-show_entries",
      "frame_side_data_list",
      "-of",
      "compact",
      file,
    ],
  ).toString();
  return streams + side;
}

const CASES = [
  {
    name: "silent sample",
    file: sample("booth2s.mp4"),
    expectAudio: false,
    check(report, output) {
      requireStream(output, "codec_type=video", "codec_name=h264");
      requireNoStream(output, "codec_type=audio");
    },
  },
  {
    name: "aac sample",
    file: sample("booth2s-aac.mp4"),
    expectAudio: true,
    check(report, output) {
      requireStream(output, "codec_type=audio", "codec_name=aac");
    },
  },
  {
    // Forward + reverse concat: exactly one reversal, verified against the
    // native CLI (`anaglyph-rs --video bench/boomerang.mp4 -d auto`):
    //   "reversal at 5.67s (frame 190): counter-clockwise -> clockwise"
    name: "boomerang auto",
    file: sample("boomerang.mp4"),
    direction: "auto",
    expectAudio: false,
    check(report) {
      if (report.direction !== "auto") {
        throw new Error(`expected auto direction, got "${report.direction}"`);
      }
      const reversals =
        report.reversals === "none detected" ? [] : report.reversals.split("; ");
      if (reversals.length !== 1) {
        throw new Error(
          `expected exactly 1 reversal (native CLI reference), got ${reversals.length}: "${report.reversals}"`,
        );
      }
      if (!/counter-clockwise -> clockwise/.test(reversals[0])) {
        throw new Error(`reversal direction mismatch, got "${reversals[0]}"`);
      }
      const seconds = Number(/([0-9.]+)s/.exec(reversals[0])?.[1]);
      if (!(Math.abs(seconds - 5.67) <= 0.1)) {
        throw new Error(`reversal time ${seconds}s deviates from 5.67s`);
      }
    },
  },
  {
    // Same clip with the eyes swapped: native `-d auto-reverse` reports the
    // mirrored reversal "clockwise -> counter-clockwise".
    name: "boomerang auto-reverse",
    file: sample("boomerang.mp4"),
    direction: "auto-reverse",
    expectAudio: false,
    check(report) {
      if (report.direction !== "auto (reversed)") {
        throw new Error(`expected "auto (reversed)" label, got "${report.direction}"`);
      }
      if (!/^5\.6\d+s clockwise -> counter-clockwise$/.test(report.reversals)) {
        throw new Error(`expected mirrored reversal at ~5.67s, got "${report.reversals}"`);
      }
    },
  },
  {
    // VP9 in WebM with odd coded dimensions (481x271): the pipeline crops to
    // even dimensions for H.264, and the WebM source itself must decode.
    name: "odd dims",
    file: sample("odd-dims.webm"),
    expectAudio: false,
    check(report, output) {
      if (report.width !== "481" || report.height !== "271") {
        throw new Error(`expected source dims 481x271, got ${report.width}x${report.height}`);
      }
      requireStream(output, "codec_type=video", "width=480", "height=270");
    },
  },
  {
    // 90-degree display matrix must survive demux -> remux untouched.
    name: "rotation",
    file: sample("rotated.mp4"),
    expectAudio: false,
    check(report, output) {
      if (!/rotation=90/.test(output)) {
        throw new Error("output lost the 90-degree display matrix");
      }
    },
  },
  {
    // Hardware toggles: decoding + blending forced to software must still
    // produce a valid conversion, and the report must say so.
    name: "software paths",
    file: sample("booth2s.mp4"),
    unchecks: ["hwDecode", "webgpuBlend"],
    expectAudio: false,
    check(report) {
      if (!/\(software-forced\)/.test(report.decoder)) {
        throw new Error(`expected software-forced decoder, got "${report.decoder}"`);
      }
      if (report.blender !== "CPU (fallback)") {
        throw new Error(`expected CPU blend fallback, got "${report.blender}"`);
      }
    },
  },
  {
    // The wasm SIMD CPU blend must be byte-identical to its scalar
    // reference (rounding, saturation, transparent-pixel rule), for every
    // anaglyph type and including tail pixels past the 4-pixel groups.
    name: "simd equals scalar",
    async custom(page) {
      const outcome = await page.evaluate(async () => {
        const wasm = await import("/pkg/anaglyph_wasm.js");
        await wasm.default();
        if (!wasm.blend_is_simd()) {
          return { ok: false, detail: "build does not use SIMD (blend_is_simd false)" };
        }
        let seed = 12345;
        const rand = () => {
          seed = (seed * 1103515245 + 12345) & 0x7fffffff;
          return seed % 256;
        };
        const fails = [];
        for (const [w, h] of [[64, 8], [37, 7], [3, 1], [0, 4]]) {
          const n = w * h * 4;
          const l = new Uint8Array(n);
          const r = new Uint8Array(n);
          for (let i = 0; i < n; i++) {
            l[i] = rand();
            r[i] = rand();
          }
          // Transparent pairs every 5th pixel (alpha 0 on both eyes) and
          // half-alpha pixels that are not transparent.
          for (let p = 0; p < w * h; p += 5) {
            l[p * 4 + 3] = 0;
            r[p * 4 + 3] = 0;
          }
          for (let p = 1; p < w * h; p += 7) l[p * 4 + 3] = 128;
          for (const type of ["color", "half-color", "grayscale", "optimized", "true"]) {
            const simd = wasm.cpu_blend_debug(l, r, type, false);
            const scalar = wasm.cpu_blend_debug(l, r, type, true);
            if (simd.length !== scalar.length) {
              fails.push(`${w}x${h} ${type}: length ${simd.length} != ${scalar.length}`);
              continue;
            }
            for (let i = 0; i < simd.length; i++) {
              if (simd[i] !== scalar[i]) {
                fails.push(
                  `${w}x${h} ${type}: byte ${i} (pixel ${i >> 2}) simd=${simd[i]} scalar=${scalar[i]}`,
                );
                break;
              }
            }
          }
        }
        return {
          ok: fails.length === 0,
          detail: fails.length === 0 ? "SIMD == scalar on all sizes/types" : fails.join("; "),
        };
      });
      return outcome;
    },
  },
  {
    // Opus is not a legal MP4 audio codec (and the browser picked an MP4
    // output): the run must succeed video-only with an explanatory note.
    name: "opus in mp4",
    file: sample("opus-in-mp4.mp4"),
    check(report, output) {
      if (!/cannot be stream-copied/i.test(report.audio)) {
        throw new Error(`expected a stream-copy refusal note, got "${report.audio}"`);
      }
      requireNoStream(output, "codec_type=audio");
    },
  },
  {
    name: "one frame",
    file: sample("one-frame.mp4"),
    expectError: /fewer than two frames/,
  },
  {
    name: "corrupt upload",
    file: sample("corrupt.mp4"),
    expectError: /Failed: .+/, // any clean pipeline/demux error (not a crash)
  },
  {
    // 1700-frame clip on software codecs gives the cancel click a multi-
    // second window; the UI must return to idle with a "Cancelled." status.
    name: "cancel",
    file: sample("long-boomerang.mp4"),
    cancelAfterMs: 300,
  },
];

function requireStream(probeOutput, ...needles) {
  const has = probeOutput
    .split("\n")
    .some((l) => /codec_type=/.test(l) && needles.every((n) => l.includes(n)));
  if (!has) {
    throw new Error(`ffprobe lacks a stream with all of: ${needles.join(", ")}`);
  }
}

function requireNoStream(probeOutput, needle) {
  const bad = probeOutput
    .split("\n")
    .some((l) => /codec_type=/.test(l) && l.includes(needle));
  if (bad) throw new Error(`ffprobe unexpectedly matched ${needle}`);
}

function fail(message) {
  console.error(`FAIL: ${message}`);
  process.exitCode = 1;
  throw new Error(message);
}

async function waitForServer(retries = 30) {
  for (let i = 0; i < retries; i++) {
    try {
      await fetch(BASE + "/video.html");
      return;
    } catch {
      await new Promise((r) => setTimeout(r, 200));
    }
  }
  fail("static server did not come up");
}

const server = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"], {
  cwd: root,
  stdio: "ignore",
});

let browser;
try {
  await waitForServer();
  let launchOptions = {
    headless: process.argv.includes("--headed") ? false : true,
    args: [
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--ignore-gpu-blocklist",
    ],
  };
  if (process.argv.includes("--gpu")) {
    // The default headless-shell build has no WebGPU; the full chromium
    // build (headless=new) renders WebGPU through SwiftShader, exercising
    // the wasm blend path that real GPUs use.
    launchOptions = { ...launchOptions, channel: "chromium" };
  }
  browser = await chromium.launch(launchOptions);

  const caseFilter = process.argv.slice(2).find((arg) => !arg.startsWith("--"));
  for (const testCase of CASES.filter(
    (c) => !caseFilter || c.name.includes(caseFilter),
  )) {
    const context = await browser.newContext({ acceptDownloads: true });
    const page = await context.newPage();
    const logs = [];
    page.on("console", (msg) => logs.push(`[${msg.type()}] ${msg.text()}`));
    page.on("pageerror", (err) => logs.push(`[pageerror] ${err.message}`));
    page.on("dialog", (dialog) => {
      logs.push(`[dialog] ${dialog.message()}`);
      dialog.dismiss();
    });

    await page.goto(`${BASE}/video.html`);
    await page.waitForFunction(
      () => !document.getElementById("caps").textContent.includes("Checking"),
      { timeout: 30_000 },
    );

    if (testCase.custom) {
      const outcome = await testCase.custom(page);
      if (!outcome.ok) fail(`${testCase.name}: ${outcome.detail}`);
      console.log(`ok(${testCase.name}): ${outcome.detail}`);
      await context.close();
      continue;
    }

    if (testCase.file) {
      await page.setInputFiles("#videoFile", testCase.file);
    }
    for (const id of testCase.unchecks ?? []) {
      await page.uncheck(`#${id}`);
    }
    if (testCase.direction) {
      await page.selectOption("#videoDirection", testCase.direction);
    }
    await page.waitForFunction(() => !document.getElementById("submit").disabled);
    await page.click("#submit");

    if (testCase.cancelAfterMs) {
      await page.waitForTimeout(testCase.cancelAfterMs);
      await page.click("#cancel");
    }

    // Terminal states: result shown, "Failed: ...", or "Cancelled."
    await page.waitForFunction(
      () =>
        document.getElementById("result").style.display === "block" ||
        document.getElementById("status").textContent.startsWith("Failed") ||
        document.getElementById("status").textContent.startsWith("Cancelled"),
      { timeout: 240_000 },
    );
    const statusText = await page.textContent("#status");

    if (testCase.cancelAfterMs) {
      if (!statusText.startsWith("Cancelled")) {
        fail(`${testCase.name}: expected "Cancelled.", got "${statusText}"`);
      }
      console.log(`ok(${testCase.name}): ${statusText}`);
      await context.close();
      continue;
    }

    if (testCase.expectError) {
      if (!statusText.startsWith("Failed")) {
        fail(`${testCase.name}: expected failure, got "${statusText}"`);
      }
      if (!testCase.expectError.test(statusText)) {
        fail(
          `${testCase.name}: "${statusText}" does not match ${testCase.expectError}`,
        );
      }
      console.log(`ok(${testCase.name}): ${statusText}`);
      await context.close();
      continue;
    }

    if (statusText.startsWith("Failed")) {
      console.log(logs.join("\n"));
      fail(`${testCase.name}: ${statusText}`);
    }

    const report = await page.evaluate(() => {
      const rows = [...document.querySelectorAll("#report tr")];
      return Object.fromEntries(
        rows.map((row) => [row.cells[0].textContent, row.cells[1].textContent]),
      );
    });
    console.log(`report(${testCase.name}):`, report);

    if (report.audio === undefined) fail(`${testCase.name}: missing audio row`);
    if (testCase.expectAudio && !report.audio.startsWith("aac")) {
      fail(`${testCase.name}: expected copied aac audio, got "${report.audio}"`);
    }
    if (
      testCase.expectAudio === false &&
      !/no audio|stream copy/.test(report.audio) &&
      !/cannot be stream-copied/i.test(report.audio)
    ) {
      fail(`${testCase.name}: unexpected audio report entry "${report.audio}"`);
    }
    if (parseInt(report.frames, 10) < 180) {
      fail(`${testCase.name}: only ${report.frames} frames encoded (expected ~189)`);
    }

    // Pull the produced file back out of the browser and validate it.
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
    const slug = testCase.name.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
    const outFile = join(root, "bench", `e2e-out-${slug}.mp4`);
    writeFileSync(outFile, Buffer.from(bytes, "base64"));
    const output = probe(outFile);
    // Mux-level sanity: ffmpeg must demux without complaints.
    execFileSync("ffmpeg", ["-v", "error", "-i", outFile, "-f", "null", "-"]);
    if (testCase.check) {
      try {
        testCase.check(report, output);
      } catch (err) {
        fail(`${testCase.name}: ${err.message}`);
      }
    }
    console.log(`ok(${testCase.name}): ${outFile}`);

    await context.close();
    if (logs.some((l) => l.startsWith("[error]") || l.startsWith("[pageerror]"))) {
      console.log(
        "console issues:\n" +
          logs.filter((l) => l.startsWith("[error]") || l.startsWith("[pageerror]")).join("\n"),
      );
    }
  }
  console.log(process.exitCode ? "E2E FAILED" : "E2E PASSED");
} finally {
  await browser?.close();
  server.kill();
}
