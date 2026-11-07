/**
 * Diff two run-web-bench result files (bench/results/*.json).
 *
 * Usage: node bench/compare.mjs BEFORE.json AFTER.json
 *
 * Prints median pairs/s per (clip, direction, backend) key with the delta,
 * plus the decode-only / blend-only bounds. Rows missing on either side are
 * shown with "—". Use this to quantify a performance change (run the bench
 * before and after, then compare).
 */
import { readFileSync } from "node:fs";

const [pathA, pathB] = process.argv.slice(2);
if (!pathA || !pathB) {
  console.error("usage: node bench/compare.mjs BEFORE.json AFTER.json");
  process.exit(2);
}

const a = JSON.parse(readFileSync(pathA, "utf8"));
const b = JSON.parse(readFileSync(pathB, "utf8"));

/** @param {any} payload */
function index(payload) {
  const map = new Map();
  for (const run of payload.web ?? []) {
    map.set(runKey(run), run);
  }
  return map;
}

const runKey = (run) => `${run.clip} | ${run.direction} | ${run.backend}`;

const mapA = index(a);
const mapB = index(b);
const keys = [...new Set([...mapA.keys(), ...mapB.keys()])].sort();

const fmt = (value, digits = 1) => (value == null ? "—" : value.toFixed(digits));

const rows = [];
for (const key of keys) {
  const runA = mapA.get(key);
  const runB = mapB.get(key);
  const fpsA = runA?.medianFps ?? null;
  const fpsB = runB?.medianFps ?? null;
  const delta = fpsA && fpsB ? ((fpsB - fpsA) / fpsA) * 100 : null;
  rows.push({
    key,
    fpsA,
    fpsB,
    delta,
    decodeA: runA?.decodeOnly?.fps ?? null,
    decodeB: runB?.decodeOnly?.fps ?? null,
    blendA: runA?.blendOnly?.fps ?? null,
    blendB: runB?.blendOnly?.fps ?? null,
  });
}

console.log(`A: ${pathA} (${a.date ?? "?"})`);
console.log(`B: ${pathB} (${b.date ?? "?"})\n`);
console.log(
  "| run | A pairs/s | B pairs/s | Δ | decode A→B | blend A→B |",
);
console.log("| --- | --- | --- | --- | --- | --- |");
for (const row of rows) {
  const delta = row.delta == null ? "—" : `${row.delta >= 0 ? "+" : ""}${row.delta.toFixed(1)}%`;
  console.log(
    `| ${row.key} | ${fmt(row.fpsA)} | ${fmt(row.fpsB)} | ${delta} | ` +
      `${fmt(row.decodeA)} → ${fmt(row.decodeB)} | ${fmt(row.blendA)} → ${fmt(row.blendB)} |`,
  );
}

if (a.native && b.native) {
  console.log("\nNative CLI reference:");
  const nativeKey = (run) => `${run.clip} | ${run.direction}`;
  const nativeMap = new Map(a.native.map((run) => [nativeKey(run), run.fps]));
  for (const run of b.native) {
    const before = nativeMap.get(nativeKey(run));
    if (before != null) {
      const delta = ((run.fps - before) / before) * 100;
      console.log(`  ${nativeKey(run)}: ${before.toFixed(1)} → ${run.fps.toFixed(1)} (${delta >= 0 ? "+" : ""}${delta.toFixed(1)}%)`);
    }
  }
}
