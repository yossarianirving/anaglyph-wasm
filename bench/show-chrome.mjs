/**
 * Live demo driver: opens gpu-video/video.html in a visible Chrome, runs one
 * conversion, then LEAVES THE BROWSER OPEN so the result can be played and
 * further conversions run by hand.
 *
 * Usage: node bench/show-chrome.mjs [clip] [direction]
 *   clip      basename under bench/ (default booth2s.mp4) or an absolute path
 *   direction clockwise | counter-clockwise | auto | auto-reverse (default clockwise)
 *
 * The process intentionally never exits (keeps the browser + static server
 * alive). Stop it with Ctrl-C / kill; Chrome closes with it.
 *
 * Chrome launches with --ignore-gpu-blocklist: this machine's Mesa driver is
 * on Chrome's blocklist, so a plain launch gets a null WebGPU adapter.
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = 8128;
const BASE = `http://localhost:${PORT}`;

const clipArg = process.argv[2] ?? "booth2s.mp4";
const direction = process.argv[3] ?? "clockwise";
const clip = clipArg.startsWith("/") ? clipArg : join(root, "bench", clipArg);
if (!existsSync(clip)) {
  console.error(`no such clip: ${clip}`);
  process.exit(2);
}

const server = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"], {
  cwd: root,
  stdio: "ignore",
});
for (let i = 0; i < 30; i++) {
  try {
    await fetch(`${BASE}/gpu-video/video.html`);
    break;
  } catch {
    await new Promise((r) => setTimeout(r, 200));
  }
}

const browser = await chromium.launch({
  channel: "chrome",
  headless: false,
  args: ["--ignore-gpu-blocklist", "--start-maximized"],
});
const context = await browser.newContext({ viewport: null, acceptDownloads: true });
const page = await context.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", String(e).slice(0, 300)));

await page.goto(`${BASE}/gpu-video/video.html`);
await page.waitForFunction(
  () => !document.getElementById("caps").textContent.includes("Checking"),
  { timeout: 30_000 },
);
console.log("caps:", (await page.textContent("#caps")).trim());

await page.setInputFiles("#videoFile", clip);
await page.selectOption("#videoDirection", direction);
await page.waitForFunction(() => !document.getElementById("submit").disabled);
console.log(`converting ${clipArg} (${direction}) …`);
await page.click("#submit");

await page.waitForFunction(
  () =>
    document.getElementById("result").style.display === "block" ||
    document.getElementById("status").textContent.startsWith("Failed"),
  null,
  { timeout: 240_000 },
);
console.log("status:", (await page.textContent("#status")).trim());
if ((await page.textContent("#status")).includes("Failed")) {
  console.log("console:", await page.evaluate(() => document.body.dataset.log ?? ""));
} else {
  const report = await page.$$eval("#report tr", (rows) =>
    rows.map((r) =>
      [...r.cells].map((c) => c.textContent.trim()).join(": "),
    ),
  );
  for (const line of report) console.log("  ", line);
  console.log("\nResult video is loaded in the page — press play in Chrome.");
}

console.log("Leaving Chrome open (convert more clips, tweak options, download).");
console.log("Press Ctrl-C in this terminal (or kill this pid) to close it.");
process.on("SIGINT", () => process.exit(0));
await new Promise(() => {}); // keep browser + server alive
