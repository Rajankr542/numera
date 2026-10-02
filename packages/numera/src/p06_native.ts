import { nativeModule, type NativeNDArray } from "./addon.js";

/** Native table `addon.p06` (native/bindings/p06_binding.cpp, D-056). Internal. */
type N = NativeNDArray;
export interface P06Native {
  concatenate(arrays: N[], axis: number | null, dtype: string | null, casting: string | null, out: N | null): N;
  stack(arrays: N[], axis: number, dtype: string | null, casting: string | null, out: N | null): N;
  splitAt(a: N, indices: number[], axis: number): N[];
  splitSections(a: N, sections: number, axis: number, equal: boolean): N[];
  unstack(a: N, axis: number): N[];
  tile(a: N, reps: number[]): N;
  repeat(a: N, repeats: number[], axis: number | null): N;
  resize(a: N, shape: number[]): N;
  resizeInplace(a: N, shape: number[]): N;
  insertAlong(a: N, axis: number, positions: number[], values: N): N;
  deleteAlong(a: N, axis: number, keep: boolean[]): N;
  trimZeros(a: N, front: boolean, back: boolean, axes: boolean[]): N;
  bufferRefs(a: N): number;
  flip(a: N, axes: number[] | null): N;
  roll(a: N, shifts: number[]): N;
  copyto(dst: N, src: N, casting: string | null, where: N | null): void;
  allFinite(a: N): boolean;
  pad(a: N, mode: string, width: number[], values: N | null, statLength: number[] | null, odd: boolean): N;
}

export const native = nativeModule<P06Native>("p06");
