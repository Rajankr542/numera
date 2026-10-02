import { nativeModule, type NativeNDArray } from "./addon.js";

/** Native table `addon.p06` (native/bindings/p06_binding.cpp, D-056). Internal. */
type N = NativeNDArray;
export interface P06Native {
  concatenate(arrays: N[], axis: number | null, dtype: string | null, casting: string | null, out: N | null): N;
  stack(arrays: N[], axis: number, dtype: string | null, casting: string | null, out: N | null): N;
  splitAt(a: N, indices: number[], axis: number): N[];
  splitSections(a: N, sections: number, axis: number, equal: boolean): N[];
  unstack(a: N, axis: number): N[];
}

export const native = nativeModule<P06Native>("p06");
