// Parity milestone P14 public functions (D-056).
// Functions exported from `p14` are spread into the default `np` object by
// index.ts; also add named exports there (only the index.ts block for P14).
// Native kernels live in native/bindings/p14_binding.cpp (`addon.p14`).
// NDArray methods: add them with declaration merging, e.g.
//   declare module "./ndarray.js" { interface NDArray { foo(): NDArray } }
//   NDArray.prototype.foo = function () { ... };
// (see AGENTS.md "Parallel milestones").

export const p14 = {} as const;
