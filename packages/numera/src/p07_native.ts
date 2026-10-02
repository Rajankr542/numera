// Typed access to the P7 native kernels (`addon.p07`, native/bindings/p07_binding.cpp).
import { nativeModule, type NativeNDArray } from "./addon.js";
import { wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";

export interface P07Native {
  logspace(sre: number, sim: number, ere: number, eim: number, isComplex: boolean, num: number,
    endpoint: boolean, base: number, dtype: string | null): NativeNDArray;
  geomspace(sre: number, sim: number, ere: number, eim: number, isComplex: boolean, num: number,
    endpoint: boolean, dtype: string): NativeNDArray;
  tri(n: number, m: number, k: number, dtype: string): NativeNDArray;
  tril(a: NativeNDArray, k: number): NativeNDArray;
  triu(a: NativeNDArray, k: number): NativeNDArray;
  diag(a: NativeNDArray, k: number): NativeNDArray;
  vander(a: NativeNDArray, n: number | null, increasing: boolean): NativeNDArray;
  indices(dims: number[], dtype: string): NativeNDArray;
  gridAxis(start: number, step: number, n: number, dtype: string): NativeNDArray;
  mgrid(starts: number[], steps: number[], sizes: number[], dtype: string): NativeNDArray;
  triIndices(n: number, m: number, k: number, upper: boolean): NativeNDArray[];
  fillDiagonal(a: NativeNDArray, values: NativeNDArray, wrap: boolean): undefined;
  concatenate(arrays: NativeNDArray[], axis: number): NativeNDArray;
  fromstring(s: string, dtype: string, count: number, sep: string): NativeNDArray;
}

/** @internal */
export const native = nativeModule<P07Native>("p07");

/** @internal */
export const wrapP07 = (f: (n: P07Native) => NativeNDArray): NDArray =>
  wrapNative(() => NDArray._wrap(f(native)));

