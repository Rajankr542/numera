// Parity milestone P15 public functions (D-056).
// Functions exported from `p15` are spread into the default `np` object by
// index.ts; also add named exports there (only the index.ts block for P15).
// Native kernels live in native/bindings/p15_binding.cpp (`addon.p15`).
// NDArray methods: add them with declaration merging, e.g.
//   declare module "./ndarray.js" { interface NDArray { foo(): NDArray } }
//   NDArray.prototype.foo = function () { ... };
// (see AGENTS.md "Parallel milestones").

export const p15 = {} as const;
