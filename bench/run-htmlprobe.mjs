import { chromium, firefox, webkit } from "playwright";
import { spawn } from "node:child_process";
const args = process.argv.slice(2);
const pageUrl = args.find((a) => !a.startsWith("--")) || "probe-canvas.html";
const browserName =
  args.find((a) => a.startsWith("--browser="))?.slice("--browser=".length) ?? "chrome";
const server = spawn("python3", ["-m", "http.server", "8126", "--bind", "127.0.0.1"], { cwd: "/home/chris/Code/anaglyph-wasm", stdio: "ignore" });
await new Promise((r) => setTimeout(r, 800));
const launchers = {
  chrome: () => chromium.launch({ channel: "chrome", headless: false, args: ["--ignore-gpu-blocklist"] }),
  firefox: () => firefox.launch({ headless: false, firefoxUserPrefs: { "dom.webgpu.enabled": true, "gfx.webrender.all": true } }),
  webkit: () => webkit.launch({ headless: false, args: ["--features=WebGPU"] }),
};
const browser = await launchers[browserName]();
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", String(e).slice(0, 300)));
page.on("console", (m) => { if (m.type() === "warning" || m.type() === "error") console.log("[" + m.type() + "]", m.text().slice(0, 250)); });
await page.goto(`http://localhost:8126/bench/${pageUrl}`);
await page.waitForFunction(() => !document.getElementById("out").textContent.startsWith("running"), { timeout: 120000 });
console.log(await page.textContent("#out"));
await browser.close(); server.kill();
