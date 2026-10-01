// Parity milestone P9 public functions (D-056).
// Functions exported from `p09` are spread into the default `np` object by
// index.ts; also add named exports there (only the index.ts block for P9).
// Native kernels live in native/bindings/p09_binding.cpp (`addon.p09`).
// NDArray methods: add them with declaration merging, e.g.
//   declare module "./ndarray.js" { interface NDArray { foo(): NDArray } }
//   NDArray.prototype.foo = function () { ... };
// (see AGENTS.md "Parallel milestones").

export const p09 = {} as const;
