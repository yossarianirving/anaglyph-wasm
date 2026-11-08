import { firefox } from "playwright";
const browser = await firefox.launch({
  headless: false,
  firefoxUserPrefs: { "dom.webgpu.enabled": true, "gfx.webrender.all": true },
});
const page = await browser.newPage();
await page.goto("http://localhost:8130/");
const info = await page.evaluate(async () => {
  const out = { vendor: navigator.userAgent.includes("Firefox") ? "firefox" : "?" };
  if (!navigator.gpu) return { ...out, gpu: false };
  const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!a) return { ...out, gpu: true, adapter: null };
  out.adapter = "present";
  out.isFallback = a.isFallbackAdapter;
  try { out.info = await a.requestAdapterInfo; } catch (e) { out.info = "threw: " + e.message; }
  out.features = [...a.features].length;
  out.maxBuffer = a.limits.maxStorageBufferBindingSize;
  return out;
});
console.log(JSON.stringify(info, null, 2));
await browser.close();
