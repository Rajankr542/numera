import { nativeModule, type NativeNDArray } from "./addon.js";

type N = NativeNDArray;

declare module "./addon.js" {
  interface NativeNDArray {
    /** Owner of the data (NumPy `base`), null if this array owns it (D-060). */
    base(): NativeNDArray | null;
    setWriteable(flag: boolean): undefined;
  }
}

/** Native table `addon.p03` (native/bindings/p03_binding.cpp, D-056). */
export interface P03Native {
  // P3-2 (D-060)
  fill(a: N, value: N): undefined;
  tobytes(a: N, order: string | undefined): Uint8Array;
  viewAs(a: N, dtype: string): N;
  byteswap(a: N, inplace: boolean): N;
  matrixTranspose(a: N): N;
  flatAssign(a: N, positions: N | null, values: N): undefined;
  // P3-3 (D-061)
  iterPlan(ops: N[], order: string | undefined): { shape: number[]; axes: number[]; flipped: boolean[] };
  // P3-4 (D-062)
  finfo(dtype: string): Record<string, number> & { dtype: string };
  iinfo(dtype: string): { bits: number; min: bigint; max: bigint };
  minScalarType(a: N): string;
  // P3-5 (D-063)
  formatFloat(a: N, opts: Record<string, unknown>): string;
  leadingTrailing(a: N, edgeitems: number): N;
  formatElements(
    a: N,
    opts: { precision: number | null; floatmode: string; suppress: boolean; sign: string; nanstr: string; infstr: string },
  ): string[];
  scalarStr(a: N): string;
}

export const p03native = nativeModule<P03Native>("p03");
