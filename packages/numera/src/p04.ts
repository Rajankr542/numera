// Parity milestone P4 public functions (D-056).
// Functions exported from `p04` are spread into the default `np` object by
// index.ts; also add named exports there (only the index.ts block for P4).
// Native kernels live in native/bindings/p04_binding.cpp (`addon.p04`).
// NDArray methods: add them with declaration merging, e.g.
//   declare module "./ndarray.js" { interface NDArray { foo(): NDArray } }
//   NDArray.prototype.foo = function () { ... };
// (see AGENTS.md "Parallel milestones").

export const p04 = {} as const;
