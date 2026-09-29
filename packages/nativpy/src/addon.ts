import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/** Handle to a native NDArray. Internal: not part of the public API. */
export interface NativeNDArray {
  shape(): number[];
  strides(): number[];
  offset(): number;
  dtype(): string;
  ndim(): number;
  size(): number;
  itemsize(): number;
  nbytes(): number;
  flags(): { cContiguous: boolean; fContiguous: boolean; ownData: boolean; writeable: boolean };
  toList(): unknown;
  toTypedArray(): ArrayBufferView;
  view(shape: number[], strides: number[], offset: number): NativeNDArray;
  reshape(shape: number[]): NativeNDArray;
  copy(): NativeNDArray;
  astype(dtype: string): NativeNDArray;
  sharesMemory(other: NativeNDArray): boolean;
  getItem(index: number[]): number | boolean;
}

export interface NativeAddon {
  napiVersion: number;
  dtypes: Record<string, { itemsize: number; alignment: number; kind: string }>;
  promoteTypes(a: string, b: string): string;
  NativeNDArray: new (...args: never[]) => NativeNDArray;
  empty(shape: number[], dtype: string): NativeNDArray;
  zeros(shape: number[], dtype: string): NativeNDArray;
  fromNested(data: unknown, dtype: string): NativeNDArray;
  fromTypedArray(data: ArrayBufferView, shape: number[], dtype: string): NativeNDArray;
  memoryStats(): { buffers: number; bytes: number };
}

// Resolution order (DECISIONS D-007): NATIVPY_ADDON_PATH, then the repo build.
function candidatePaths(): string[] {
  const here = dirname(fileURLToPath(import.meta.url));
  const paths: string[] = [];
  const env = process.env["NATIVPY_ADDON_PATH"];
  if (env) paths.push(resolve(env));
  // src/ or dist/ -> packages/nativpy -> repo root
  paths.push(resolve(here, "../../../build/Release/nativpy.node"));
  paths.push(resolve(here, "../../../build/Debug/nativpy.node"));
  return paths;
}

function loadAddon(): NativeAddon {
  const require = createRequire(import.meta.url);
  const tried = candidatePaths();
  for (const p of tried) {
    if (existsSync(p)) return require(p) as NativeAddon;
  }
  throw new Error(
    "nativpy: native addon not found. Build it with `pnpm build:native` " +
      "(requires CMake, Ninja and a C++20 compiler). Tried:\n  " +
      tried.join("\n  "),
  );
}

export const addon: NativeAddon = loadAddon();
