/** quick one-shot: load probe-motion.html and show console/errors */
import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = 8132;
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
  page.on("console", (m) => console.log(`[console.${m.type()}]`, m.text()));
  page.on("pageerror", (e) => console.log("[pageerror]", String(e)));
  await page.goto(
    `http://localhost:${PORT}/bench/probe-motion.html?clip=../bench/boomerang.mp4&batch=${process.argv[2] || 8}&mode=${process.argv[3] || "external"}`,
  );
  try {
    await page.waitForFunction(() => window.__result !== null, null, {
      timeout: 60_000,
    });
    console.log(JSON.stringify(await page.evaluate(() => window.__result)));
  } catch {
    console.log("TIMEOUT; out text:", await page.evaluate(() => document.getElementById("out").textContent));
  }
} finally {
  await browser.close();
  server.kill();
}
