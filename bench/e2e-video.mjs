/**
 * End-to-end smoke test for the browser video pipeline.
 *
 * Serves the repo, drives video.html in headless Chromium through the two
 * bundled sample clips (silent and AAC), and validates each output file with
 * ffprobe (codec, frame count, audio stream presence, duration).
 *
 * Usage: node bench/e2e-video.mjs [name-filter] [--headed] [--gpu]
 *
 * Note: headless Chromium encodes/decodes on the CPU; the shipped
 * headless-shell build exposes no WebGPU at all (and full chromium here
 * falls back without a GPU process), so this run covers the CPU blend path.
 * The WebGPU blend path is exercised on real hardware via gif.html/bench.html
 * (same `blend_rgba_async` code) and `--headed` runs. Performance numbers
 * come from bench-video.html on real hardware.
 *
 * Note on frame counts: this input's container marks one trailing packet
 * discardable; ffmpeg's decoder drops it (native CLI: N-2 pairs) while
 * WebCodecs decodes it (browser: N-1 pairs). The extra frame is appended at
 * the end and does not shift any earlier pairing.
 */
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = 8123;
const BASE = `http://localhost:${PORT}`;

const CASES = [
  {
    name: "silent sample",
    file: join(root, "bench", "booth2s.mp4"),
    expectAudio: false,
  },
  {
    name: "aac sample",
    file: join(root, "bench", "booth2s-aac.mp4"),
    expectAudio: true,
  },
  {
    // Forward + reverse concat: exactly one reversal, verified against the
    // native CLI:
    //   anaglyph-rs --video bench/boomerang.mp4 -d auto
    //   -> "reversal at 5.67s (frame 190): counter-clockwise -> clockwise"
    name: "boomerang auto",
    file: join(root, "bench", "boomerang.mp4"),
    direction: "auto",
    expectAudio: false,
    check(report) {
      if (report.direction !== "auto") {
        throw new Error(`expected auto direction, got "${report.direction}"`);
      }
      const reversals =
        report.reversals === "none detected"
          ? []
          : report.reversals.split("; ");
      if (reversals.length !== 1) {
        throw new Error(
          `expected exactly 1 reversal (native CLI reference), got ${reversals.length}: "${report.reversals}"`,
        );
      }
      if (!/counter-clockwise -> clockwise/.test(reversals[0])) {
        throw new Error(
          `reversal direction mismatch, got "${reversals[0]}" (native: counter-clockwise -> clockwise at 5.67s)`,
        );
      }
      const seconds = Number(/([0-9.]+)s/.exec(reversals[0])?.[1]);
      if (!(Math.abs(seconds - 5.67) <= 0.1)) {
        throw new Error(
          `reversal time ${seconds}s deviates from native reference 5.67s`,
        );
      }
    },
  },
  {
    // Same clip with the eyes swapped: native `-d auto-reverse` reports the
    // mirrored reversal "clockwise -> counter-clockwise".
    name: "boomerang auto-reverse",
    file: join(root, "bench", "boomerang.mp4"),
    direction: "auto-reverse",
    expectAudio: false,
    check(report) {
      if (report.direction !== "auto (reversed)") {
        throw new Error(
          `expected "auto (reversed)" direction label, got "${report.direction}"`,
        );
      }
      if (!/^5\.6\d+s clockwise -> counter-clockwise$/.test(report.reversals)) {
        throw new Error(
          `expected mirrored reversal at ~5.67s, got "${report.reversals}"`,
        );
      }
    },
  },
];

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
    console.log(`caps: ${await page.textContent("#caps")}`);

    await testCase.prepare?.(page);
    if (testCase.file) {
      await page.setInputFiles("#videoFile", testCase.file);
    }
    if (testCase.direction) {
      await page.selectOption("#videoDirection", testCase.direction);
    }
    await page.waitForFunction(() => !document.getElementById("submit").disabled);
    await page.click("#submit");

    // Software blend + software codec on SwiftShader: allow generous time.
    await page.waitForFunction(
      () =>
        document.getElementById("result").style.display === "block" ||
        document.getElementById("status").textContent.startsWith("Failed"),
      { timeout: 240_000 },
    );
    const statusText = await page.textContent("#status");
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
    if (!testCase.expectAudio && !report.audio.includes("no audio")) {
      fail(`${testCase.name}: expected audio-free report, got "${report.audio}"`);
    }
    if (parseInt(report.frames, 10) < 180) {
      fail(`${testCase.name}: only ${report.frames} frames encoded (expected ~189)`);
    }
    if (testCase.check) {
      try {
        testCase.check(report);
      } catch (err) {
        fail(`${testCase.name}: ${err.message}`);
      }
    }

    // Pull the produced file back out of the browser for ffprobe.
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
    console.log(`wrote ${outFile}`);

    await context.close();
    if (logs.some((l) => l.startsWith("[error]") || l.startsWith("[pageerror]"))) {
      console.log("console issues:\n" + logs.filter((l) => l.startsWith("[error]") || l.startsWith("[pageerror]")).join("\n"));
    }
  }
  console.log(process.exitCode ? "E2E FAILED" : "E2E PASSED");
} finally {
  await browser?.close();
  server.kill();
}
