/**
 * Probe: how fast does GPUBuffer.mapAsync resolve after a submit on each
 * engine? The all-GPU pipeline gates each frame's canvas snapshot on one
 * mapAsync; if an engine only resolves map completions on a slow periodic
 * tick, per-frame throughput is capped regardless of GPU speed.
 *
 * Measures: (a) tight loop of submit+mapAsync (per-call latency, serial)
 *           (b) 8 outstanding submits, mapped in order (pipelined drain)
 *
 * Usage: node bench/probe-mapasync.mjs [--browser=chrome|firefox|webkit]
 */
import { chromium, firefox, webkit } from "playwright";

const browserName =
  process.argv.find((a) => a.startsWith("--browser="))?.slice("--browser=".length) ?? "chrome";
const launchers = {
  chrome: () => chromium.launch({ channel: "chrome", headless: false, args: ["--ignore-gpu-blocklist"] }),
  firefox: () =>
    firefox.launch({ headless: false, firefoxUserPrefs: { "dom.webgpu.enabled": true, "gfx.webrender.all": true } }),
  webkit: () => webkit.launch({ headless: false, args: ["--features=WebGPU"] }),
};

const browser = await launchers[browserName]();
const page = await browser.newPage();
// WebGPU needs a secure context; about:blank/data: are not one in Firefox.
const probeUrl = process.env.PROBE_URL || "http://localhost:8130/";
await page.goto(probeUrl);
const result = await page.evaluate(async () => {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { error: "no adapter" };
  const device = await adapter.requestDevice();
  const src = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_SRC });
  const buffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const oneRound = async () => {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, buffer, 0, 64);
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const t1 = performance.now();
    buffer.unmap();
    return t1 - t0;
  };

  // (a) serial latency, 20 rounds
  const serial = [];
  for (let i = 0; i < 20; i++) serial.push(await oneRound());

  // (b) pipelined: cannot use one buffer for concurrent maps; use 8 buffers.
  const bufs = Array.from({ length: 8 }, () =>
    device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
  );
  const t0 = performance.now();
  const subs = bufs.map((b) => {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, b, 0, 64);
    device.queue.submit([enc.finish()]);
    return b.mapAsync(GPUMapMode.READ); // map right after submit (allowed: map queues behind work)
  });
  await Promise.all(subs);
  const pipelined = performance.now() - t0;
  for (const b of bufs) b.unmap();

  const sorted = [...serial].sort((a, b) => a - b);
  return {
    serialMedianMs: sorted[Math.floor(sorted.length / 2)],
    serialMinMs: sorted[0],
    serialMaxMs: sorted[sorted.length - 1],
    pipelined8TotalMs: pipelined,
  };
});
console.log(browserName, JSON.stringify(result, null, 2));
await browser.close();
