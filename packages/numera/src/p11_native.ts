// Typed access to the P11 native kernels (`addon.p11`, native/bindings/p11_binding.cpp).
import { nativeModule, type NativeNDArray } from "./addon.js";
import { array } from "./creation.js";
import { wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
import type { ArrayLike } from "./ufunc.js";

export interface P11Native {
  cholesky(a: NativeNDArray, upper: boolean): NativeNDArray;
  slogdet(a: NativeNDArray): { sign: NativeNDArray; logabsdet: NativeNDArray };
  matrixPower(a: NativeNDArray, n: number): NativeNDArray;
  pinv(a: NativeNDArray, rcond: NativeNDArray, hermitian: boolean): NativeNDArray;
  matrixRank(
    a: NativeNDArray,
    tol: NativeNDArray | null,
    rtol: NativeNDArray | null,
    hermitian: boolean,
  ): NativeNDArray;
  cond(a: NativeNDArray, p: number | string | null): NativeNDArray;
  einsum(operands: NativeNDArray[], terms: string[], steps: [number[], string][]): NativeNDArray;
}

/** @internal */
export const native = (): P11Native => nativeModule<P11Native>("p11");
/** @internal */
export const toArray = (a: ArrayLike): NDArray => (a instanceof NDArray ? a : array(a));
/** @internal */
export const w = (f: (n: P11Native) => NativeNDArray): NDArray =>
  wrapNative(() => NDArray._wrap(f(native())));
