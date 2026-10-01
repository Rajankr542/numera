// Parity milestone P5 public functions (D-056).
// Functions exported from `p05` are spread into the default `np` object by
// index.ts; also add named exports there (only the index.ts block for P5).
// Native kernels live in native/bindings/p05_binding.cpp (`addon.p05`).
// NDArray methods: add them with declaration merging, e.g.
//   declare module "./ndarray.js" { interface NDArray { foo(): NDArray } }
//   NDArray.prototype.foo = function () { ... };
// (see AGENTS.md "Parallel milestones").

export const p05 = {} as const;
