// Parity milestone P11 public functions (D-056).
// Functions exported from `p11` are spread into the default `np` object by
// index.ts; also add named exports there (only the index.ts block for P11).
// Native kernels live in native/bindings/p11_binding.cpp (`addon.p11`).
// NDArray methods: add them with declaration merging, e.g.
//   declare module "./ndarray.js" { interface NDArray { foo(): NDArray } }
//   NDArray.prototype.foo = function () { ... };
// (see AGENTS.md "Parallel milestones").

export const p11 = {} as const;
