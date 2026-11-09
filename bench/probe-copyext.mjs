import { spawn } from "node:child_process";
import { chromium } from "playwright";
const root = "/home/chris/Code/anaglyph-wasm";
const PORT = 8133;
const browser = await chromium.launch({ channel: "chrome", headless: false, args: ["--ignore-gpu-blocklist"] });
const server = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"], { cwd: root, stdio: "ignore" });
try {
  const page = await browser.newPage();
  page.on("console", (m) => console.log(`[.${m.type()}]`, m.text()));
  page.on("pageerror", (e) => console.log("[pageerror]", String(e)));
  await page.goto(`http://localhost:${PORT}/bench/probe-copyext.html?clip=../bench/booth2s.mp4`);
  await page.waitForFunction(() => window.__result !== null, null, { timeout: 60000 });
  console.log(JSON.stringify(await page.evaluate(() => window.__result)));
} finally { await browser.close(); server.kill(); }
