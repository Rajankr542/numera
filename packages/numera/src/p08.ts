// Parity milestone P8 public functions (D-056).
// Functions exported from `p08` are spread into the default `np` object by
// index.ts; also add named exports there (only the index.ts block for P8).
// Native kernels live in native/bindings/p08_binding.cpp (`addon.p08`).
// NDArray methods: add them with declaration merging, e.g.
//   declare module "./ndarray.js" { interface NDArray { foo(): NDArray } }
//   NDArray.prototype.foo = function () { ... };
// (see AGENTS.md "Parallel milestones").

export const p08 = {} as const;
