// P16B API reference entries (D-190).
// np.ma masked arrays. Every example is executed by packages/numera/test/docs_site.test.ts.

/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "ma",
    title: "Masked arrays (np.ma)",
    intro: "The `np.ma` module provides a MaskedArray class that wraps an NDArray with an optional boolean mask. Masked elements are excluded from arithmetic and reductions. `np.ma.nomask` is `false`; `np.ma.masked` is a masked-scalar sentinel.",
    entries: [
      {
        name: "ma",
        sig: "np.ma",
        desc: "The masked-array sub-module. Exposes `MaskedArray`, constructors, mask utilities, and all ufuncs that propagate the mask.",
        returns: "object",
        example: `np.ma.nomask; // => false
np.ma.masked_array([1, 2, 3], { mask: [false, true, false] }).count(); // => 2`,
      },
    ],
  },
];
