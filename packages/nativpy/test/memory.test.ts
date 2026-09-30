import { describe, expect, it } from "vitest";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// D-023: with experimental Node-API, basic finalizers run synchronously in GC,
// so a synchronous loop no longer accumulates dead result buffers. A clean
// child process with --expose-gc gives deterministic GC control.
const addonPath = resolve(dirname(fileURLToPath(import.meta.url)), "../../../build/Release/nativpy.node");

const script = `
const addon = require(${JSON.stringify(addonPath)});
const base = addon.ones([100000], "float64");
for (let i = 0; i < 1000; i++) addon.binary("add", base, base);
const beforeGc = addon.memoryStats().buffers;
globalThis.gc();
const afterGc = addon.memoryStats().buffers;   // still synchronous: no event-loop turn
process.stdout.write(JSON.stringify({ beforeGc, afterGc }));
`;

describe("buffer release (D-023)", () => {
  it.runIf(existsSync(addonPath))("frees dead result buffers during a synchronous GC", () => {
    const r = spawnSync(process.execPath, ["--expose-gc", "-e", script], { encoding: "utf8" });
    expect(r.status, r.stderr).toBe(0);
    const { beforeGc, afterGc } = JSON.parse(r.stdout) as { beforeGc: number; afterGc: number };
    // `base` (and at most one temporary still referenced) remains live.
    expect(afterGc).toBeLessThanOrEqual(2);
    expect(beforeGc).toBeGreaterThanOrEqual(afterGc);
  });
});

describe("native handle construction guard", () => {
  it.runIf(existsSync(addonPath))("rejects direct JS construction of NativeNDArray", () => {
    const req = createRequire(import.meta.url);
    const native = req(addonPath) as { NativeNDArray: new (...args: unknown[]) => unknown };
    for (const arg of [undefined, "__nativpy_internal__", {}, null]) {
      expect(() => new native.NativeNDArray(arg)).toThrow(/cannot be constructed directly/);
    }
  });
});
