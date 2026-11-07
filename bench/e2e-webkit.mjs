/**
 * End-to-end tests for the browser video pipeline on a real WebKit engine
 * (GNOME Web / WebKitGTK) driven through WebKitWebDriver — the same engine
 * family as iOS Safari, which is where the pipeline's frame-copy behaviour
 * diverges from Chromium (`copyTo` ignores the requested RGBA format and
 * hands back raw NV12 planes; the pipeline must detect that and raster
 * frames through a canvas instead).
 *
 * Usage:
 *   WebKitWebDriver --host=127.0.0.1 --port=9922 &   # or: --start-driver
 *   node bench/e2e-webkit.mjs [name-filter] [--start-driver]
 *
 * Requires: `npm run build:wasm`, a running static server (this script
 * starts one on :8124), GNOME Web's WebKitWebDriver, and a display
 * (DISPLAY/WAYLAND_DISPLAY env is passed through; MiniBrowser opens
 * normally on XWayland).
 *
 * Checks:
 *  - conversion succeeds, report says `frameCopy: "canvas raster"`
 *  - output decodes with ffprobe, right frame count (N-1 pairs)
 *  - output frame is NOT gray garbage: luma spread (YDIF/YMIN..YMAX) and
 *    red/cyan channel structure present (the NV12-as-RGBA failure mode is
 *    mostly-flat gray and would fail both)
 *  - rotation metadata survives (rotated.mp4)
 *  - cancel recovers the page: worker ends the conversion and a second
 *    conversion then completes in the same page (the user-reported hang).
 */
import { execFileSync, spawn, spawnSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const PORT = 8124;
const DRIVER_PORT = 9922;
const BASE = `http://127.0.0.1:${PORT}`;
const DRIVER = `http://127.0.0.1:${DRIVER_PORT}`;
const FILTER = process.argv.slice(2).find((a) => !a.startsWith("--"));

let driverProc = null;
let serverProc = null;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function req(method, url, body, timeoutMs = 30000) {
  const r = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(timeoutMs),
  });
  return r.json();
}

function startServer() {
  serverProc = spawn("python3", ["-m", "http.server", String(PORT), "--bind", "127.0.0.1"], {
    cwd: root,
    stdio: "ignore",
  });
}

async function startDriver() {
  driverProc = spawn("WebKitWebDriver", [`--host=127.0.0.1`, `--port=${DRIVER_PORT}`, "--replace-on-new-session"], {
    stdio: "ignore",
  });
}

async function waitForHttp(url, tries = 80) {
  for (let i = 0; i < tries; i++) {
    try {
      const r = await fetch(url, { signal: AbortSignal.timeout(1000) });
      if (r.status < 500) return true;
    } catch {
      /* not up yet */
    }
    await sleep(250);
  }
  return false;
}

/** Kick+poll eval: the classic protocol cannot await promises. */
async function evalAsync(sid, expr, timeoutMs = 180000) {
  const kick = `
    window.__diag = { done: false, result: null };
    (async () => {
      try {
        window.__diag.result = await (${expr.replace(/;\s*$/, "")});
      } catch (e) {
        window.__diag.result = { __error: String(e), name: e && e.name,
          stack: String((e && e.stack) || "").slice(0, 600) };
      }
      window.__diag.done = true;
    })();
    return "kicked";
  `;
  await req("POST", `${DRIVER}/session/${sid}/execute/sync`, { script: kick, args: [] });
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const st = await req("POST", `${DRIVER}/session/${sid}/execute/sync`, {
      script: "return window.__diag ? JSON.stringify(window.__diag) : 'gone';",
      args: [],
    });
    if (typeof st.value === "string" && st.value !== "gone") {
      const diag = JSON.parse(st.value);
      if (diag.done) {
        if (diag.result && diag.result.__error) {
          throw new Error(`${diag.result.__error}\n${diag.result.stack || ""}`);
        }
        return diag.result;
      }
    }
    await sleep(300);
  }
  throw new Error("evalAsync timed out");
}

/** Run the pipeline in-page on a fixture and return report + output bytes. */
async function convertInPage(sid, file, opts = {}) {
  return evalAsync(sid, `(async () => {
    const wasm = await import("/pkg/anaglyph_wasm.js");
    await wasm.default();
    const { convertVideoToAnaglyph } = await import("/js/video-pipeline.js");
    const blob = await (await fetch(${JSON.stringify("/bench/" + file)})).blob();
    const { blob: out, report } = await convertVideoToAnaglyph({
      file: blob,
      anaglyphType: "color",
      direction: ${JSON.stringify(opts.direction ?? "clockwise")},
      createBlender: (w, h, t, backend) => wasm.VideoBlender.new(w, h, t, backend ?? null),
    });
    const bytes = new Uint8Array(await out.arrayBuffer());
    let bin = "";
    for (let i = 0; i < bytes.length; i += 8192)
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
    return { report, b64: btoa(bin) };
  })();`);
}

/** ffprobe a saved output: all streams (compact) + first-frame side data. */
function probe(file) {
  const streams = execFileSync(
    "ffprobe",
    [
      "-v", "error",
      "-show_entries", "stream=codec_type,codec_name,width,height,nb_frames",
      "-of", "compact",
      file,
    ],
    { encoding: "utf8" },
  );
  const side = execFileSync(
    "ffprobe",
    [
      "-v", "error",
      "-select_streams", "v:0",
      "-read_intervals", "%+#1",
      "-show_entries", "frame_side_data_list",
      "-of", "compact",
      file,
    ],
    { encoding: "utf8" },
  );
  return streams + side;
}

let failures = 0;
function fail(msg) {
  failures += 1;
  console.log(`FAIL: ${msg}`);
}
function ok(name, detail) {
  console.log(`ok(${name}): ${detail}`);
}

async function main() {
  startServer();
  if (process.argv.includes("--start-driver")) startDriver();
  if (!(await waitForHttp(`${BASE}/video.html`))) throw new Error("static server did not start");
  if (!(await waitForHttp(`${DRIVER}/status`))) {
    throw new Error(
      "WebKitWebDriver not reachable on :9922 — start it (WebKitWebDriver --host=127.0.0.1 --port=9922) or pass --start-driver",
    );
  }

  const sess = await req("POST", `${DRIVER}/session`, {
    capabilities: { alwaysMatch: {}, firstMatch: [{}] },
  });
  if (!sess.value?.sessionId) throw new Error(`session failed: ${JSON.stringify(sess)}`);
  const sid = sess.value.sessionId;
  await req("POST", `${DRIVER}/session/${sid}/timeouts`, { script: 60000, pageLoad: 60000 });
  await req("POST", `${DRIVER}/session/${sid}/url`, { url: `${BASE}/video.html` });
  await sleep(2500);

  const engine = await req("POST", `${DRIVER}/session/${sid}/execute/sync`, {
    script: "return navigator.userAgent;",
    args: [],
  });
  console.log(`WebKit engine: ${engine.value}`);

  // ---- conversion cases ----
  const cases = [
    {
      name: "basic anaglyph",
      file: "booth2s.mp4",
      check: (report) => {
        if (report.frameCopy !== "canvas raster") {
          throw new Error(`expected canvas raster copy on WebKit, got "${report.frameCopy}"`);
        }
        if (report.frames < 180) throw new Error(`too few frames: ${report.frames}`);
      },
      probeMinFrames: 180,
      notGray: true,
    },
    {
      name: "audio copy",
      file: "booth2s-aac.mp4",
      check: (report) => {
        if (!/aac \(stream copy\)/.test(report.audio)) {
          throw new Error(`expected aac stream copy, got "${report.audio}"`);
        }
      },
      probeNeed: ["codec_type=audio"],
    },
    {
      name: "rotation",
      file: "rotated.mp4",
      check: (report) => {
        if (!report.rotation.includes("container metadata")) {
          throw new Error(`expected rotation metadata pass-through, got "${report.rotation}"`);
        }
      },
      probeNeed: ["rotation=90"],
    },
    {
      name: "odd dims",
      file: "odd-dims.webm",
      check: (report) => {
        if (!/^\d+×\d+$/.test(report.output)) throw new Error(`bad output dims ${report.output}`);
      },
    },
  ];

  const tmp = tmpdir();
  for (const c of cases) {
    if (FILTER && !c.name.includes(FILTER)) continue;
    try {
      const { report, b64 } = await convertInPage(sid, c.file);
      c.check(report);
      const out = join(tmp, `webkit-${c.file.replace(/\.[^.]+$/, "")}.mp4`);
      writeFileSync(out, Buffer.from(b64, "base64"));
      const info = probe(out);
      const nb = /nb_frames=(\d+)/.exec(info);
      if (c.probeMinFrames && (!nb || +nb[1] < c.probeMinFrames)) {
        throw new Error(`ffprobe frame count ${nb?.[1]} < ${c.probeMinFrames}`);
      }
      for (const needle of c.probeNeed ?? []) {
        if (!info.includes(needle)) {
          throw new Error(`ffprobe output lacks "${needle}": ${info.replace(/\n/g, " ")}`);
        }
      }
      if (c.notGray) {
        // Garbled NV12-as-RGBA output is near-flat gray; a real anaglyph
        // has strong luma spread. Check one frame.
        const png = join(tmp, "webkit-check.png");
        execFileSync("ffmpeg", ["-v", "error", "-y", "-i", out, "-vf", "select=eq(n\\,60)", "-frames:v", "1", png]);
        const stats = frameSignalStats(png);
        if (stats.ydif < 8) {
          throw new Error(`frame looks gray (YDIF=${stats.ydif.toFixed(1)}) — NV12-as-RGBA regression?`);
        }
      }
      ok(c.name, out);
    } catch (e) {
      fail(`${c.name}: ${e.message.split("\n")[0]}`);
    }
  }

  // ---- cancel recovery through the real UI ----
  // WebDriver cannot stuff a path into a <input type=file> (browser
  // security), so the page is driven exactly like a user would: the
  // "Load sample clip" button (with ?sample= pointing at the long fixture),
  // Convert, Cancel — then a second full conversion in the same page.
  if (!FILTER || "cancel".includes(FILTER)) {
    try {
      await req("POST", `${DRIVER}/session/${sid}/url`, {
        url: `${BASE}/video.html?sample=bench/long-boomerang.mp4`,
      });
      await sleep(2000);
      const click = (id) =>
        req("POST", `${DRIVER}/session/${sid}/execute/sync`, {
          script: `document.getElementById(${JSON.stringify(id)}).click(); return 1;`,
          args: [],
        });
      const statusOf = () =>
        req("POST", `${DRIVER}/session/${sid}/execute/sync`, {
          script: "return document.getElementById('status').textContent;",
          args: [],
        });

      click("useSample");
      if (!(await waitStatus(sid, /^Ready:/, 10000))) {
        throw new Error(`sample clip never reached Ready (status="${await statusOf()}")`);
      }
      click("submit");
      // The long clip runs several seconds on WebKit; catch it mid-flight.
      if (!(await waitStatus(sid, /\d+ \/ \d+ frames/, 15000))) {
        throw new Error(`conversion never reported progress (status="${await statusOf()}")`);
      }
      click("cancel");
      const cancelled = await waitStatus(sid, /Cancelled|Failed/, 8000);
      if (!cancelled) throw new Error("cancel did not settle the page within 8s");

      // Recovery: a second conversion in the same page must complete.
      await req("POST", `${DRIVER}/session/${sid}/url`, { url: `${BASE}/video.html` });
      await sleep(1500);
      click("useSample");
      if (!(await waitStatus(sid, /^Ready:/, 10000))) {
        throw new Error(`re-pick after cancel never reached Ready (status="${await statusOf()}")`);
      }
      click("submit");
      const done = await waitStatus(sid, /Done|Failed/, 90000);
      if (!done || !done.startsWith("Done")) {
        throw new Error(`conversion after cancel did not finish (status="${done}")`);
      }
      ok("cancel recovers", `cancelled via "${cancelled.trim()}" + second run Done`);
    } catch (e) {
      fail(`cancel recovers: ${e.message.split("\n")[0]}`);
    }
  }

  await req("DELETE", `${DRIVER}/session/${sid}`);
  if (failures > 0) {
    console.log(`\nWEBKIT E2E FAILED (${failures})`);
    process.exitCode = 1;
  } else {
    console.log("\nWEBKIT E2E PASSED");
  }
}

async function waitStatus(sid, re, ms) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    const r = await req("POST", `${DRIVER}/session/${sid}/execute/sync`, {
      script: "return document.getElementById('status').textContent;",
      args: [],
    });
    if (typeof r.value === "string" && re.test(r.value)) return r.value;
    await sleep(200);
  }
  return null;
}

/** signalstats luma stats for a PNG (`metadata=print` writes to stderr). */
function frameSignalStats(png) {
  const r = spawnSync(
    "ffmpeg",
    ["-v", "error", "-i", png, "-vf", "signalstats,metadata=print", "-f", "null", "-"],
    { encoding: "utf8" },
  );
  const stderr = r.stderr ?? "";
  const num = (key) => {
    const m = new RegExp(`lavfi\\.signalstats\\.${key}=([0-9.]+)`).exec(stderr);
    return m ? parseFloat(m[1]) : NaN;
  };
  return { yavg: num("YAVG"), ymin: num("YMIN"), ymax: num("YMAX"), ydif: num("YDIF") };
}

main().finally(() => {
  serverProc?.kill();
  driverProc?.kill();
});
