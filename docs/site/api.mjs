// API reference content for the numera documentation site (DECISIONS D-030).
// Rendered by scripts/build-docs.mjs into packages/numera/docs/index.html.
// Every `example` is executed by packages/numera/test/docs_site.test.ts:
// a line ending in `// => <json>` must evaluate to that value (NDArray results
// are compared via `toArray()`), so keep examples real and deterministic.
// Text fields accept `code` spans only.

/**
 * @typedef {{ name: string, type: string, desc: string }} Arg
 * @typedef {{ name: string, sig: string, desc: string, args?: Arg[],
 *   returns: string, example: string, notes?: string }} Entry
 * @typedef {{ id: string, title: string, intro?: string, entries: Entry[] }} Category
 */

import { categories as p03 } from "./parts/p03.mjs";
import { categories as p04 } from "./parts/p04.mjs";
import { categories as p05 } from "./parts/p05.mjs";
import { categories as p06 } from "./parts/p06.mjs";
import { categories as p07 } from "./parts/p07.mjs";
import { categories as p08 } from "./parts/p08.mjs";
import { categories as p09 } from "./parts/p09.mjs";
import { categories as p10 } from "./parts/p10.mjs";
import { categories as p11 } from "./parts/p11.mjs";
import { categories as p12 } from "./parts/p12.mjs";
import { categories as p13 } from "./parts/p13.mjs";
import { categories as p14 } from "./parts/p14.mjs";
import { categories as p15 } from "./parts/p15.mjs";
import { categories as p16a } from "./parts/p16a.mjs";
import { categories as p16d } from "./parts/p16d.mjs";
import { categories as p16c } from "./parts/p16c.mjs";

const shape = { name: "shape", type: "number | number[]", desc: "Dimensions of the new array." };
const dtypeOpt = { name: "[options.dtype]", type: "DTypeLike", desc: "Element type, e.g. `\"float32\"` or `np.int32`. Default `float64`." };
const arrayArg = (name = "a") => ({ name, type: "ArrayLike", desc: "An `NDArray`, nested JS array or scalar." });
const reduceOpts = [
  { name: "[options.axis]", type: "number | number[] | null", desc: "Axis or axes to reduce. Omitted reduces all axes." },
  { name: "[options.keepdims]", type: "boolean", desc: "Keep reduced axes with length 1. Default `false`." },
];

/** @type {Category[]} */
const baseCategories = [
  {
    id: "creation",
    title: "Array creation",
    entries: [
      {
        name: "array",
        sig: "np.array(data, [options])",
        desc: "Creates an array from nested JS arrays, numbers, booleans or bigints. The dtype is inferred like NumPy: all booleans → `bool`, all integers → `int64`, otherwise `float64`.",
        args: [
          { name: "data", type: "NestedArray | NDArray", desc: "Values to copy into the array." },
          { name: "[options.dtype]", type: "DTypeLike", desc: "Force a dtype instead of inferring it." },
        ],
        returns: "NDArray",
        example: `const a = np.array([[1, 2], [3, 4]]);
a.shape;       // => [2, 2]
a.dtype.name;  // => "int64"
np.array([1, 2], { dtype: "float32" }).dtype.name; // => "float32"`,
      },
      {
        name: "asarray",
        sig: "np.asarray(data, [options])",
        desc: "Like `array`, but returns the input unchanged (no copy) if it is already an `NDArray` of the requested dtype.",
        args: [
          { name: "data", type: "NestedArray | NDArray", desc: "Input data." },
          dtypeOpt,
        ],
        returns: "NDArray",
        example: `const a = np.array([1, 2, 3]);
np.asarray(a) === a; // => true`,
      },
      {
        name: "zeros",
        sig: "np.zeros(shape, [options])",
        desc: "Array filled with zeros.",
        args: [shape, dtypeOpt],
        returns: "NDArray",
        example: `np.zeros([2, 3]); // => [[0, 0, 0], [0, 0, 0]]`,
      },
      {
        name: "ones",
        sig: "np.ones(shape, [options])",
        desc: "Array filled with ones.",
        args: [shape, dtypeOpt],
        returns: "NDArray",
        example: `np.ones(3, { dtype: "int32" }); // => [1, 1, 1]`,
      },
      {
        name: "empty",
        sig: "np.empty(shape, [options])",
        desc: "Array whose contents are uninitialized. It is faster than `zeros` when you are going to overwrite every element anyway.",
        args: [shape, dtypeOpt],
        returns: "NDArray",
        example: `np.empty([2, 2]).shape; // => [2, 2]`,
      },
      {
        name: "full",
        sig: "np.full(shape, fillValue, [options])",
        desc: "Array filled with `fillValue`. Without `dtype`, the dtype is inferred from the value.",
        args: [shape, { name: "fillValue", type: "number | boolean | bigint", desc: "Value for every element." }, dtypeOpt],
        returns: "NDArray",
        example: `np.full([2, 2], 7); // => [[7, 7], [7, 7]]
np.full(2, 1.5).dtype.name; // => "float64"`,
      },
      {
        name: "zerosLike",
        sig: "np.zerosLike(a, [options])",
        desc: "Zeros with the same shape and dtype as `a`. The same family has `onesLike(a)`, `emptyLike(a)` and `fullLike(a, fillValue)`.",
        args: [{ name: "a", type: "NDArray", desc: "Template array." }, dtypeOpt],
        returns: "NDArray",
        example: `const a = np.array([[1, 2], [3, 4]]);
np.zerosLike(a);      // => [[0, 0], [0, 0]]
np.fullLike(a, 9);    // => [[9, 9], [9, 9]]
np.onesLike(a).dtype.name; // => "int64"`,
      },
      {
        name: "arange",
        sig: "np.arange([start], stop, [step], [options])",
        desc: "Evenly spaced values in the half-open interval `[start, stop)`. With one argument, it is `stop`. The result is `int64` when every argument is an integer, and `float64` otherwise. `options` is always the 4th argument: `np.arange(0, 5, 1, { dtype: \"float32\" })`.",
        args: [
          { name: "[start]", type: "number", desc: "Start of the interval. Default `0`." },
          { name: "stop", type: "number", desc: "End of the interval (excluded)." },
          { name: "[step]", type: "number", desc: "Spacing between values. Default `1`." },
          dtypeOpt,
        ],
        returns: "NDArray",
        example: `np.arange(5);        // => [0, 1, 2, 3, 4]
np.arange(2, 10, 3); // => [2, 5, 8]
np.arange(1, 0, -0.25); // => [1, 0.75, 0.5, 0.25]`,
      },
      {
        name: "linspace",
        sig: "np.linspace(start, stop, [num=50], [options])",
        desc: "`num` evenly spaced samples from `start` to `stop`.",
        args: [
          { name: "start", type: "number", desc: "First value." },
          { name: "stop", type: "number", desc: "Last value (included unless `endpoint` is `false`)." },
          { name: "[num=50]", type: "number", desc: "Number of samples." },
          { name: "[options.endpoint]", type: "boolean", desc: "Include `stop`. Default `true`." },
          dtypeOpt,
        ],
        returns: "NDArray",
        example: `np.linspace(0, 1, 5); // => [0, 0.25, 0.5, 0.75, 1]
np.linspace(0, 1, 4, { endpoint: false }); // => [0, 0.25, 0.5, 0.75]`,
      },
      {
        name: "eye",
        sig: "np.eye(n, [m=n], [options])",
        desc: "2-D array with ones on a diagonal and zeros elsewhere.",
        args: [
          { name: "n", type: "number", desc: "Number of rows." },
          { name: "[m]", type: "number", desc: "Number of columns. Default `n`." },
          { name: "[options.k]", type: "number", desc: "Diagonal offset: `0` main, positive above, negative below." },
          dtypeOpt,
        ],
        returns: "NDArray",
        example: `np.eye(2);                        // => [[1, 0], [0, 1]]
np.eye(2, 3, { k: 1, dtype: "int32" }); // => [[0, 1, 0], [0, 0, 1]]`,
      },
      {
        name: "identity",
        sig: "np.identity(n, [options])",
        desc: "Square identity matrix, the same as `eye(n)`.",
        args: [{ name: "n", type: "number", desc: "Size." }, dtypeOpt],
        returns: "NDArray",
        example: `np.identity(3).toArray()[1]; // => [0, 1, 0]`,
      },
      {
        name: "fromTypedArray",
        sig: "np.fromTypedArray(data, [shape], [options])",
        desc: "Copies a TypedArray (or Node `Buffer`) into an array. The dtype defaults to the element type, e.g. `Float32Array` → `float32`. An explicit dtype reinterprets the raw bytes.",
        args: [
          { name: "data", type: "ArrayBufferView", desc: "Source data (copied)." },
          { name: "[shape]", type: "number | number[]", desc: "Target shape. Default: 1-D." },
          dtypeOpt,
        ],
        returns: "NDArray",
        example: `const a = np.fromTypedArray(new Int32Array([1, 2, 3, 4]), [2, 2]);
a.dtype.name; // => "int32"
a;            // => [[1, 2], [3, 4]]`,
      },
    ],
  },
  {
    id: "ndarray",
    title: "NDArray",
    intro: "Every function returns an `NDArray`, a handle to data held in native memory. Read the values with `toArray()`, `item()` or `toTypedArray()`.",
    entries: [
      {
        name: "attributes",
        sig: "a.shape · a.ndim · a.size · a.dtype · a.strides · a.itemSize · a.nbytes · a.flags",
        desc: "Read-only array metadata. `strides` are in bytes. `flags` holds `cContiguous`, `fContiguous`, `writeable` and `ownData`.",
        returns: "number[] | number | DType | ArrayFlags",
        example: `const a = np.zeros([2, 3], { dtype: "float32" });
a.shape;    // => [2, 3]
a.ndim;     // => 2
a.size;     // => 6
a.itemSize; // => 4
a.nbytes;   // => 24
a.strides;  // => [12, 4]`,
      },
      {
        name: "a.toArray",
        sig: "a.toArray()",
        desc: "Copies the data into nested JS arrays. `int64`/`uint64` values beyond 2^53 lose precision; use `toTypedArray()` for those. Complex arrays throw `NotImplementedError`; use `toTypedArray()` to read them.",
        returns: "NestedArray",
        example: `np.arange(4).reshape(2, 2).toArray(); // => [[0, 1], [2, 3]]`,
      },
      {
        name: "a.item",
        sig: "a.item([...index])",
        desc: "Returns a single element as a JS number or boolean. With no index, the array must have exactly one element.",
        args: [{ name: "[...index]", type: "number[]", desc: "One integer per axis (negative counts from the end)." }],
        returns: "number | boolean",
        example: `const a = np.array([[1, 2], [3, 4]]);
a.item(1, 0);      // => 3
np.sum(a).item();  // => 10`,
      },
      {
        name: "a.toTypedArray",
        sig: "a.toTypedArray()",
        desc: "Copies the data into a TypedArray of the matching element type (`BigInt64Array` for int64). Complex arrays come out interleaved as `[re, im, re, im, ...]`.",
        returns: "ArrayBufferView",
        example: `const t = np.array([1, 2, 3], { dtype: "float32" }).toTypedArray();
t instanceof Float32Array; // => true`,
      },
      {
        name: "a.reshape",
        sig: "a.reshape(shape) / a.reshape(...dims)",
        desc: "Same data with a new shape. Returns a view when the layout allows it. One dimension may be `-1` and is then inferred.",
        args: [{ name: "shape", type: "number[] | ...number", desc: "New dimensions." }],
        returns: "NDArray",
        example: `np.arange(6).reshape(2, 3); // => [[0, 1, 2], [3, 4, 5]]
np.arange(6).reshape([-1, 2]).shape; // => [3, 2]`,
      },
      {
        name: "a.T",
        sig: "a.T · a.transpose([...axes])",
        desc: "Transposed view with no copy. `transpose()` reverses the axes, or permutes them in the order given.",
        returns: "NDArray",
        example: `const a = np.array([[1, 2, 3], [4, 5, 6]]);
a.T; // => [[1, 4], [2, 5], [3, 6]]
np.zeros([2, 3, 4]).transpose(1, 0, 2).shape; // => [3, 2, 4]`,
      },
      {
        name: "a.astype",
        sig: "a.astype(dtype)",
        desc: "Copy converted to another dtype. Float → int truncates toward zero, as in NumPy.",
        args: [{ name: "dtype", type: "DTypeLike", desc: "Target dtype." }],
        returns: "NDArray",
        example: `np.array([1.7, -2.7]).astype("int32"); // => [1, -2]`,
      },
      {
        name: "a.copy",
        sig: "a.copy()",
        desc: "Deep, C-contiguous copy of the array. `ravel()` flattens to a view when possible; `flatten()` always copies.",
        returns: "NDArray",
        example: `const a = np.array([[1, 2], [3, 4]]);
const b = a.copy();
np.mayShareMemory(a, b); // => false
a.T.flatten();           // => [1, 3, 2, 4]`,
      },
      {
        name: "a.toString",
        sig: "String(a)",
        desc: "NumPy-style representation, also used by `console.log`.",
        returns: "string",
        example: `String(np.array([1, 2])); // => "array([1, 2])"`,
      },
    ],
  },
  {
    id: "shape",
    title: "Shape manipulation",
    intro: "All of these return views (no data copy) unless noted. Each one exists as a free function (`np.transpose(a)`) and, for most, as a method (`a.transpose()`).",
    entries: [
      {
        name: "reshape",
        sig: "np.reshape(a, shape)",
        desc: "Function form of `a.reshape(shape)`.",
        args: [{ name: "a", type: "NDArray", desc: "Input array." }, { name: "shape", type: "number | number[]", desc: "New shape; one entry may be `-1`." }],
        returns: "NDArray",
        example: `np.reshape(np.arange(4), [2, 2]); // => [[0, 1], [2, 3]]`,
      },
      {
        name: "transpose",
        sig: "np.transpose(a, [axes])",
        desc: "Permutes the axes. By default the order is reversed.",
        args: [{ name: "a", type: "NDArray", desc: "Input array." }, { name: "[axes]", type: "number[]", desc: "New axis order." }],
        returns: "NDArray",
        example: `np.transpose(np.zeros([2, 3, 4])).shape;            // => [4, 3, 2]
np.transpose(np.zeros([2, 3, 4]), [0, 2, 1]).shape; // => [2, 4, 3]`,
      },
      {
        name: "squeeze",
        sig: "np.squeeze(a, [axis])",
        desc: "Removes length-1 axes: all of them, or only the ones given.",
        args: [{ name: "a", type: "NDArray", desc: "Input array." }, { name: "[axis]", type: "number | number[]", desc: "Axes to remove (each must have length 1)." }],
        returns: "NDArray",
        example: `np.squeeze(np.zeros([1, 3, 1])).shape;    // => [3]
np.squeeze(np.zeros([1, 3, 1]), 0).shape; // => [3, 1]`,
      },
      {
        name: "expandDims",
        sig: "np.expandDims(a, axis)",
        desc: "Inserts length-1 axes at the given positions.",
        args: [{ name: "a", type: "NDArray", desc: "Input array." }, { name: "axis", type: "number | number[]", desc: "Positions of the new axes in the result." }],
        returns: "NDArray",
        example: `np.expandDims(np.zeros([3]), 0).shape;       // => [1, 3]
np.expandDims(np.zeros([3]), [0, 2]).shape; // => [1, 3, 1]`,
      },
      {
        name: "swapAxes",
        sig: "np.swapAxes(a, axis1, axis2)",
        desc: "Swaps two axes.",
        args: [{ name: "a", type: "NDArray", desc: "Input array." }, { name: "axis1, axis2", type: "number", desc: "The axes to swap." }],
        returns: "NDArray",
        example: `np.swapAxes(np.zeros([2, 3, 4]), 0, 2).shape; // => [4, 3, 2]`,
      },
      {
        name: "moveAxis",
        sig: "np.moveAxis(a, source, destination)",
        desc: "Moves axes to new positions; the other axes keep their relative order.",
        args: [
          { name: "a", type: "NDArray", desc: "Input array." },
          { name: "source", type: "number | number[]", desc: "Original positions." },
          { name: "destination", type: "number | number[]", desc: "New positions." },
        ],
        returns: "NDArray",
        example: `np.moveAxis(np.zeros([2, 3, 4]), 0, -1).shape; // => [3, 4, 2]`,
      },
      {
        name: "ravel",
        sig: "np.ravel(a)",
        desc: "Flattens to 1-D. Returns a view when possible, otherwise a copy.",
        args: [{ name: "a", type: "NDArray", desc: "Input array." }],
        returns: "NDArray",
        example: `np.ravel(np.array([[1, 2], [3, 4]])); // => [1, 2, 3, 4]`,
      },
      {
        name: "broadcastTo",
        sig: "np.broadcastTo(a, shape)",
        desc: "Read-only view of `a` broadcast to `shape`, without copying.",
        args: [arrayArg(), { name: "shape", type: "number | number[]", desc: "Target shape." }],
        returns: "NDArray",
        example: `np.broadcastTo([1, 2], [2, 2]); // => [[1, 2], [1, 2]]`,
      },
      {
        name: "broadcastShapes",
        sig: "np.broadcastShapes(...shapes)",
        desc: "The shape that the given shapes broadcast to. Throws `BroadcastError` if they are incompatible.",
        args: [{ name: "...shapes", type: "number | number[]", desc: "Input shapes." }],
        returns: "number[]",
        example: `np.broadcastShapes([2, 1], [3]); // => [2, 3]`,
      },
      {
        name: "seterr",
        sig: "np.seterr({ all?, divide?, over?, under?, invalid? })",
        desc: "Sets how floating-point errors are handled: `\"ignore\"`, `\"warn\"` (a Node `RuntimeWarning`), `\"raise\"` (`FloatingPointError`) or `\"print\"`. Returns the previous settings.",
        args: [{ name: "settings", type: "object", desc: "Modes per category; `all` sets every category." }],
        returns: "object",
        example: `const old = np.seterr({ all: "ignore" });
np.seterr(old).divide; // => "ignore"`,
      },
      {
        name: "geterr",
        sig: "np.geterr()",
        desc: "The current floating-point error settings.",
        returns: "object",
        example: `np.geterr().under; // => "ignore"`,
      },
      {
        name: "errstate",
        sig: "np.errstate(settings, fn)",
        desc: "Runs `fn` synchronously with the given error settings and restores the previous ones afterwards.",
        args: [
          { name: "settings", type: "object", desc: "As for `seterr`." },
          { name: "fn", type: "() => T", desc: "Function to run." },
        ],
        returns: "T",
        example: `np.errstate({ divide: "ignore" }, () => np.divide([1], [0]).toArray()[0] === Infinity); // => true`,
      },
    ],
  },
  {
    id: "indexing",
    title: "Indexing",
    intro: "JavaScript has no `a[1:3, ::2]` syntax, so indexing uses methods. A slice is a tuple `[start, stop, step]` in which `null` means omitted. Basic indices (integers, slices, `newaxis`, `ellipsis`) return **views**; integer-array and boolean-mask indices return **copies**, as in NumPy.",
    entries: [
      {
        name: "a.get",
        sig: "a.get(...index)",
        desc: "NumPy `a[i, j, ...]`, with one spec per axis: an integer, a slice tuple, `np.newaxis`, `np.ellipsis`, an integer `NDArray` or a boolean mask. A full integer index returns a 0-d array; call `.item()` to get a JS value.",
        args: [{ name: "...index", type: "IndexSpec[]", desc: "One spec per axis." }],
        returns: "NDArray",
        example: `const b = np.arange(12).reshape(3, 4);
b.get(1);                 // => [4, 5, 6, 7]
b.get(1, -1).item();      // => 7
b.get([0, 2], 1);         // => [1, 5]
b.get(np.ellipsis, 0);    // => [0, 4, 8]
b.get(np.newaxis).shape;  // => [1, 3, 4]
b.get(np.array([2, 0]));  // => [[8, 9, 10, 11], [0, 1, 2, 3]]`,
      },
      {
        name: "a.slice",
        sig: "a.slice(index)",
        desc: "NumPy `a[index]` from a single list. A flat list of numbers/`null` (length 1–3) is one slice tuple, so `a.slice([0, 5])` is `a[0:5]`. Otherwise it is one spec per axis, so `a.slice([[0, 2], [null, null, 2]])` is `a[0:2, ::2]`.",
        args: [{ name: "index", type: "SliceTuple | IndexSpec[]", desc: "Slice tuple or per-axis specs." }],
        returns: "NDArray",
        example: `const b = np.arange(12).reshape(3, 4);
b.slice([[0, 2], [null, null, 2]]); // => [[0, 2], [4, 6]]
np.arange(5).slice([null, null, -1]); // => [4, 3, 2, 1, 0]
const m = np.array([1, -2, 3, -4]);
m.get(np.array([true, false, true, false])); // => [1, 3]`,
      },
      {
        name: "a.set",
        sig: "a.set(index, value)",
        desc: "NumPy `a[index] = value`. `value` broadcasts to the selection and is cast to `a`'s dtype. Writes go through to views. Throws `ValueError` on read-only arrays (for example `broadcastTo` results).",
        args: [
          { name: "index", type: "IndexSpec | IndexSpec[]", desc: "Per-axis specs (like `get`), or a single spec." },
          { name: "value", type: "NDArray | NestedArray", desc: "Values to write." },
        ],
        returns: "void",
        example: `const c = np.zeros(4);
c.set([[0, 2]], [7, 8]);   // c[0:2] = [7, 8]
c.set(3, 1);               // c[3] = 1
c;                         // => [7, 8, 0, 1]
const d = np.zeros([2, 2]);
d.set([np.ellipsis], 5);   // d[...] = 5
d;                         // => [[5, 5], [5, 5]]`,
      },
      {
        name: "where",
        sig: "np.where(condition, x, y) / np.where(condition)",
        desc: "Chooses elements from `x` where `condition` is true and from `y` elsewhere, broadcasting all three. With only `condition`, it is the same as `nonzero`.",
        args: [
          { name: "condition", type: "NDArray | NestedArray", desc: "Boolean selector." },
          { name: "[x], [y]", type: "NDArray | NestedArray", desc: "Values for true / false (give both or neither)." },
        ],
        returns: "NDArray | NDArray[]",
        example: `np.where([true, false, true], [1, 2, 3], [0, 0, 0]); // => [1, 0, 3]`,
      },
      {
        name: "nonzero",
        sig: "np.nonzero(a)",
        desc: "Indices of the non-zero elements, as one `int64` array per dimension.",
        args: [{ name: "a", type: "NDArray | NestedArray", desc: "Input array." }],
        returns: "NDArray[]",
        example: `np.nonzero([0, 3, 0, 4])[0]; // => [1, 3]
np.nonzero([[1, 0], [0, 1]]).map((ix) => ix.toArray()); // => [[0, 1], [0, 1]]`,
      },
      {
        name: "take",
        sig: "np.take(a, indices, [axis])",
        desc: "Takes elements by index along an axis. Without `axis`, the array is indexed as if flattened.",
        args: [
          { name: "a", type: "NDArray | NestedArray", desc: "Source array." },
          { name: "indices", type: "NDArray | NestedArray", desc: "Integer indices." },
          { name: "[axis]", type: "number | null", desc: "Axis to take along. Default: the flattened array." },
        ],
        returns: "NDArray",
        example: `const a = np.array([[1, 2], [3, 4]]);
np.take(a, [3, 0]);    // => [4, 1]
np.take(a, [1], 1);    // => [[2], [4]]`,
      },
    ],
  },

  {
    id: "math",
    title: "Math (ufuncs)",
    intro: "Element-wise functions with NumPy broadcasting and type promotion. Operands can be `NDArray`s, nested arrays or scalars. JS scalars are \"weak\", so `int32 array + 1` stays `int32`. Pass `{ out }` to write into an existing array (NumPy `out=`): the result is cast to `out.dtype` under `same_kind`, and `out` itself is returned.",
    entries: [
      {
        name: "add",
        sig: "np.add(a, b, { out? }) · np.subtract · np.multiply · np.divide",
        desc: "Element-wise `+`, `-`, `*`, `/`. `divide` is true division, so integer inputs give `float64`.",
        args: [arrayArg("a"), arrayArg("b")],
        returns: "NDArray",
        example: `np.add([[1], [2]], [10, 20]); // => [[11, 21], [12, 22]]
np.subtract([5, 7], 2);       // => [3, 5]
np.multiply([1, 2, 3], 2);    // => [2, 4, 6]
np.divide([1, 3], 2);         // => [0.5, 1.5]`,
      },
      {
        name: "floorDivide",
        sig: "np.floorDivide(a, b, { out? }) · np.mod(a, b, { out? })",
        desc: "Python-style floor division and modulo: the result of `mod` takes the sign of the divisor.",
        args: [arrayArg("a"), arrayArg("b")],
        returns: "NDArray",
        example: `np.floorDivide([7, -7], 2); // => [3, -4]
np.mod([7, -7], 3);         // => [1, 2]`,
      },
      {
        name: "power",
        sig: "np.power(a, b, { out? })",
        desc: "Element-wise `a ** b`.",
        args: [arrayArg("a"), arrayArg("b")],
        returns: "NDArray",
        example: `np.power([1, 2, 3], 2);  // => [1, 4, 9]
np.power(2, [0.5, -1]).toArray()[1]; // => 0.5`,
      },
      {
        name: "abs",
        sig: "np.abs(a, { out? }) · np.negative(a, { out? })",
        desc: "Absolute value and negation. For complex input `abs` returns the magnitude in the matching real dtype (`complex128` → `float64`).",
        args: [arrayArg()],
        returns: "NDArray",
        example: `np.abs([-1, 2, -3]);  // => [1, 2, 3]
np.negative([1, -2]); // => [-1, 2]
np.abs([np.complex(3, 4)]); // => [5]`,
      },
      {
        name: "real",
        sig: "np.real(a) · np.imag(a) · np.conj(a) · np.conjugate(a) · np.angle(z, deg?) · np.iscomplex(a) · np.isreal(a) · np.iscomplexobj(a) · np.isrealobj(a)",
        desc: "Complex helpers. `real` and `imag` return views of the components (also available as `a.real` and `a.imag`), so writing to them changes `a`. For real input, `real` is `a` itself and `imag` is read-only zeros. `conj` negates the imaginary part. `angle` is `atan2(im, re)`, in degrees if `deg` is true. `iscomplex` and `isreal` test `imag != 0` element by element. `iscomplexobj` and `isrealobj` test the dtype.",
        args: [arrayArg(), { name: "[deg]", type: "boolean", desc: "`angle` only: return degrees. Default `false`." }],
        returns: "NDArray (boolean for iscomplexobj / isrealobj)",
        example: `const z = np.array([np.complex(1, 2), np.complex(3, -4)]);
np.real(z);     // => [1, 3]
np.imag(z);     // => [2, -4]
np.imag(np.conj(z)); // => [-2, 4]
np.angle([np.complex(0, 1)], true); // => [90]
np.iscomplex([np.complex(1, 0), np.complex(1, 1)]); // => [false, true]
np.isreal([1, 2]);     // => [true, true]
np.iscomplexobj(z);    // => true
np.isrealobj([1, 2]);  // => true`,
      },
      {
        name: "sqrt",
        sig: "np.sqrt(a, { out? }) · np.exp(a, { out? }) · np.log(a, { out? })",
        desc: "Square root, `e^x` and the natural log. Integer inputs are promoted to `float64`; out-of-domain inputs give `NaN`, as in NumPy.",
        args: [arrayArg()],
        returns: "NDArray",
        example: `np.sqrt([4, 9]);    // => [2, 3]
np.exp([0]);        // => [1]
np.log([1]);        // => [0]
Number.isNaN(np.sqrt(-1).item()); // => true`,
      },
      {
        name: "matmul",
        sig: "np.matmul(a, b) · np.dot(a, b)",
        desc: "Matrix product (`@`), with batched broadcasting over leading axes. `dot` follows NumPy's `dot` rules, including scalars and 1-D inputs. Backed by BLAS.",
        args: [arrayArg("a"), arrayArg("b")],
        returns: "NDArray",
        example: `np.matmul([[1, 2], [3, 4]], [[5], [6]]); // => [[17], [39]]
np.dot([1, 2, 3], [4, 5, 6]).item();      // => 32`,
      },
      {
        name: "outer",
        sig: "np.outer(a, b) · np.inner(a, b)",
        desc: "Outer product of two vectors (flattened), and the inner product over the last axes.",
        args: [arrayArg("a"), arrayArg("b")],
        returns: "NDArray",
        example: `np.outer([1, 2], [3, 4]);        // => [[3, 4], [6, 8]]
np.inner([1, 2], [3, 4]).item(); // => 11`,
      },
    ],
  },

  {
    id: "reduce",
    title: "Reductions",
    intro: "Each one exists as a function (`np.sum(a, opts)`) and as a method (`a.sum(opts)`). The result is always an `NDArray` (0-d when every axis is reduced); use `.item()` to get a JS number.",
    entries: [
      {
        name: "sum",
        sig: "np.sum(a, [options]) · np.prod(a, [options])",
        desc: "Sum or product of elements. Small integer types accumulate in `int64`/`uint64`, as in NumPy.",
        args: [
          arrayArg(),
          ...reduceOpts,
          { name: "[options.dtype]", type: "DTypeLike", desc: "Accumulator/result dtype." },
          { name: "[options.initial]", type: "number", desc: "Starting value." },
        ],
        returns: "NDArray",
        example: `const a = np.array([[1, 2, 3], [4, 5, 6]]);
np.sum(a).item();                        // => 21
np.sum(a, { axis: 0 });                  // => [5, 7, 9]
a.sum({ axis: 1, keepdims: true });      // => [[6], [15]]
np.prod([1, 2, 3, 4]).item();            // => 24`,
      },
      {
        name: "max",
        sig: "np.max(a, [options]) · np.min(a, [options])",
        desc: "Largest or smallest element. `amax` and `amin` are aliases. Empty reductions without `initial` throw `ValueError`.",
        args: [arrayArg(), ...reduceOpts, { name: "[options.initial]", type: "number", desc: "Value included in the reduction." }],
        returns: "NDArray",
        example: `const a = np.array([[1, 2, 3], [4, 5, 6]]);
a.max({ axis: 1 });      // => [3, 6]
np.min(a).item();        // => 1
np.amax([-1, 5]).item(); // => 5`,
      },
      {
        name: "mean",
        sig: "np.mean(a, [options])",
        desc: "Arithmetic mean. Integer inputs produce `float64`.",
        args: [arrayArg(), ...reduceOpts, { name: "[options.dtype]", type: "DTypeLike", desc: "Accumulator/result dtype." }],
        returns: "NDArray",
        example: `np.mean([[1, 2], [3, 4]], { axis: 0 }); // => [2, 3]`,
      },
      {
        name: "std",
        sig: "np.std(a, [options]) · np.var(a, [options])",
        desc: "Standard deviation and variance. `ddof` is the delta degrees of freedom, so `ddof: 1` gives the sample estimate. `np.var` is also exported by name as `variance`.",
        args: [arrayArg(), ...reduceOpts, { name: "[options.ddof]", type: "number", desc: "Divisor is `N - ddof`. Default `0`." }],
        returns: "NDArray",
        example: `np.var([1, 2, 3, 4]).item();              // => 1.25
np.std([2, 4, 4, 4, 5, 5, 7, 9]).item();   // => 2
np.var([1, 2, 3, 4], { ddof: 1 }).item();  // => 1.6666666666666667`,
      },
      {
        name: "argmax",
        sig: "np.argmax(a, [options]) · np.argmin(a, [options])",
        desc: "`int64` index of the first maximum/minimum. Without `axis`, the index is into the flattened array.",
        args: [
          arrayArg(),
          { name: "[options.axis]", type: "number | null", desc: "A single axis. Default: flattened." },
          { name: "[options.keepdims]", type: "boolean", desc: "Keep the reduced axis." },
        ],
        returns: "NDArray",
        example: `const a = np.array([[1, 9, 3], [7, 2, 8]]);
np.argmax(a).item();        // => 1
np.argmin(a, { axis: 1 });  // => [0, 1]`,
      },
    ],
  },

  {
    id: "linalg",
    title: "np.linalg",
    intro: "Dense linear algebra on LAPACK: Apple Accelerate on macOS, and a portable built-in backend elsewhere. The functions take the last two axes as the matrix, and inputs with extra leading axes are processed as batches, as in NumPy. Singular or non-convergent inputs throw `LinAlgError`. Float results are shown rounded; they are exact up to rounding.",
    entries: [
      {
        name: "linalg.det",
        sig: "np.linalg.det(a)",
        desc: "Determinant of a square matrix, or of each matrix in a batch.",
        args: [arrayArg()],
        returns: "NDArray",
        example: `np.linalg.det([[3, 1], [1, 2]]).item(); // => 5`,
      },
      {
        name: "linalg.inv",
        sig: "np.linalg.inv(a)",
        desc: "Matrix inverse. Throws `LinAlgError` for singular matrices.",
        args: [arrayArg()],
        returns: "NDArray",
        example: `np.linalg.inv([[4, 7], [2, 6]]); // => [[0.6, -0.7], [-0.2, 0.4]]`,
      },
      {
        name: "linalg.solve",
        sig: "np.linalg.solve(a, b)",
        desc: "Solves `a @ x = b` for `x`.",
        args: [arrayArg("a"), arrayArg("b")],
        returns: "NDArray",
        example: `np.linalg.solve([[3, 1], [1, 2]], [9, 8]); // => [2, 3]`,
      },
      {
        name: "linalg.eig",
        sig: "np.linalg.eig(a) · np.linalg.eigvals(a)",
        desc: "Eigen-decomposition of a general square matrix. Results are always complex: `complex64` for `float32` or `complex64` input, otherwise `complex128`. `eigvals` returns only the eigenvalues, computed without eigenvectors as in NumPy.",
        args: [arrayArg()],
        returns: "{ eigenvalues: NDArray, eigenvectors: NDArray }",
        example: `const { eigenvalues } = np.linalg.eig([[2, 0], [0, 3]]);
eigenvalues.dtype.name; // => "complex128"`,
      },
      {
        name: "linalg.eigh",
        sig: "np.linalg.eigh(a) · np.linalg.eigvalsh(a)",
        desc: "Eigen-decomposition of a symmetric/Hermitian matrix (real or complex), using the lower triangle. Eigenvalues are real and sorted in ascending order. `eigvalsh` computes eigenvalues only.",
        args: [arrayArg()],
        returns: "{ eigenvalues: NDArray, eigenvectors: NDArray }",
        example: `np.linalg.eigh([[2, 1], [1, 2]]).eigenvalues; // => [1, 3]
np.linalg.eigvalsh([[2, 1], [1, 2]]);         // => [1, 3]`,
      },
      {
        name: "linalg.svd",
        sig: "np.linalg.svd(a, [options])",
        desc: "Singular value decomposition `a = U @ diag(S) @ Vh`. `S` is in descending order.",
        args: [
          arrayArg(),
          { name: "[options.fullMatrices]", type: "boolean", desc: "Return square `U`/`Vh`. Default `true`." },
          { name: "[options.computeUV]", type: "boolean", desc: "If `false`, `U` and `Vh` are `null`. Default `true`." },
        ],
        returns: "{ U: NDArray | null, S: NDArray, Vh: NDArray | null }",
        example: `np.linalg.svd([[3, 0], [0, 4]]).S; // => [4, 3]
np.linalg.svd([[1, 2], [3, 4], [5, 6]], { fullMatrices: false }).U.shape; // => [3, 2]`,
      },
      {
        name: "linalg.qr",
        sig: "np.linalg.qr(a, [mode=\"reduced\"])",
        desc: "QR factorization `a = Q @ R`. `mode` is `\"reduced\"`, `\"complete\"` or `\"r\"`; with `\"r\"`, `Q` is `null`.",
        args: [arrayArg(), { name: "[mode]", type: "\"reduced\" | \"complete\" | \"r\"", desc: "Output shapes. Default `\"reduced\"`." }],
        returns: "{ Q: NDArray | null, R: NDArray }",
        example: `np.linalg.qr([[1, 2], [3, 4], [5, 6]]).R.shape; // => [2, 2]
np.linalg.qr([[1, 2], [3, 4]], "r").Q;          // => null`,
      },
      {
        name: "linalg.lstsq",
        sig: "np.linalg.lstsq(a, b, [rcond])",
        desc: "Least-squares solution of `a @ x ≈ b`, computed with SVD.",
        args: [
          arrayArg("a"), arrayArg("b"),
          { name: "[rcond]", type: "number | null", desc: "Cut-off for small singular values. Default `eps * max(M, N)`." },
        ],
        returns: "{ x: NDArray, residuals: NDArray, rank: number, s: NDArray }",
        example: `// fit y = m*x + c
const r = np.linalg.lstsq([[0, 1], [1, 1], [2, 1], [3, 1]], [-1, 0.2, 0.9, 2.1]);
r.x;    // => [1, -0.95]
r.rank; // => 2`,
      },
      {
        name: "linalg.norm",
        sig: "np.linalg.norm(a, [options])",
        desc: "Vector or matrix norm. The default is the 2-norm of the flattened input.",
        args: [
          arrayArg(),
          { name: "[options.ord]", type: "number | \"fro\" | \"nuc\" | null", desc: "Norm order (`Infinity`, `1`, `2`, `\"fro\"`, ...)." },
          { name: "[options.axis]", type: "number | [number, number] | null", desc: "Vector axis or matrix axes." },
          { name: "[options.keepdims]", type: "boolean", desc: "Keep reduced axes." },
        ],
        returns: "NDArray",
        example: `np.linalg.norm([3, 4]).item();                     // => 5
np.linalg.norm([[1, -2], [3, 4]], { ord: 1 }).item(); // => 6
np.linalg.norm([[3, 4], [6, 8]], { axis: 1 });       // => [5, 10]`,
      },
      {
        name: "linalg.backend",
        sig: "np.linalg.backend()",
        desc: "Name of the active native LAPACK backend: `\"accelerate\"` (macOS) or `\"fallback\"`.",
        returns: "string",
        example: `typeof np.linalg.backend(); // => "string"`,
      },
    ],
  },

  {
    id: "fft",
    title: "np.fft",
    intro: "Discrete Fourier transforms on the pocketfft algorithm (the same one NumPy uses). Complex results are `complex128`. Read them with `toTypedArray()`, which returns an interleaved `Float64Array` `[re0, im0, re1, im1, ...]`. The 1-D functions accept either positional `(a, n, axis, norm)` or an options object `{ n, axis, norm }`.",
    entries: [
      {
        name: "fft.fft",
        sig: "np.fft.fft(a, [n], [axis=-1], [norm]) · np.fft.ifft(...)",
        desc: "1-D complex DFT and its inverse.",
        args: [
          arrayArg(),
          { name: "[n]", type: "number", desc: "Transform length; the input is zero-padded or truncated to fit." },
          { name: "[axis=-1]", type: "number", desc: "Axis to transform." },
          { name: "[norm]", type: "\"backward\" | \"ortho\" | \"forward\"", desc: "Scaling convention. Default `\"backward\"`." },
        ],
        returns: "NDArray (complex128)",
        example: `const f = np.fft.fft([1, 2, 3, 4]);
f.dtype.name;                       // => "complex128"
Array.from(f.toTypedArray());       // => [10, 0, -2, 2, -2, 0, -2, -2]
Array.from(np.fft.ifft(f).toTypedArray()); // => [1, 0, 2, 0, 3, 0, 4, 0]
Array.from(np.fft.fft([1, 1, 1, 1], { norm: "ortho" }).toTypedArray()); // => [2, 0, 0, 0, 0, 0, 0, 0]`,
      },
      {
        name: "fft.rfft",
        sig: "np.fft.rfft(a, [n], [axis], [norm]) · np.fft.irfft(...)",
        desc: "DFT of real input, which returns the `n / 2 + 1` non-negative frequencies. `irfft` inverts it back to a real array; its default length is `2 * (m - 1)`.",
        args: [arrayArg(), { name: "[n], [axis], [norm]", type: "", desc: "As for `fft`." }],
        returns: "NDArray",
        example: `const spec = np.fft.rfft([1, 2, 3, 4]);
spec.shape;           // => [3]
np.fft.irfft(spec);   // => [1, 2, 3, 4]`,
      },
      {
        name: "fft.fftn",
        sig: "np.fft.fftn(a, [options]) · ifftn · fft2 · ifft2",
        desc: "N-D transforms over `axes` (by default all of them, or the last `s.length`). `fft2`/`ifft2` default to the last two axes.",
        args: [
          arrayArg(),
          { name: "[options.s]", type: "number[]", desc: "Output lengths per transformed axis." },
          { name: "[options.axes]", type: "number[]", desc: "Axes to transform." },
          { name: "[options.norm]", type: "string", desc: "Scaling convention." },
        ],
        returns: "NDArray (complex128)",
        example: `np.fft.fft2([[1, 2], [3, 4]]).shape; // => [2, 2]
Array.from(np.fft.fft2([[1, 2], [3, 4]]).toTypedArray()).filter((_, i) => i % 2 === 0); // => [10, -2, -4, 0]`,
      },
      {
        name: "fft.fftfreq",
        sig: "np.fft.fftfreq(n, [d=1]) · np.fft.rfftfreq(n, [d=1])",
        desc: "Sample frequencies for `fft`/`rfft` outputs of length `n`, with sample spacing `d`.",
        args: [
          { name: "n", type: "number", desc: "Window length." },
          { name: "[d=1]", type: "number", desc: "Sample spacing (1 / sample rate)." },
        ],
        returns: "NDArray",
        example: `np.fft.fftfreq(4);       // => [0, 0.25, -0.5, -0.25]
np.fft.rfftfreq(4, 0.5); // => [0, 0.5, 1]`,
      },
    ],
  },

  {
    id: "random",
    title: "np.random",
    intro: "Random number generation. For the same seed, the streams are bit-for-bit identical to NumPy's: `defaultRng` is PCG64 with SeedSequence, and the legacy functions use MT19937. Without `size`, samplers return a JS number; with `size`, they return an `NDArray`.",
    entries: [
      {
        name: "random.defaultRng",
        sig: "np.random.defaultRng([seed])",
        desc: "Creates a `Generator` (PCG64). This is the recommended API. Without a seed it uses OS entropy.",
        args: [{ name: "[seed]", type: "number | bigint | number[] | Generator", desc: "Seed, or an existing Generator (returned as-is)." }],
        returns: "Generator",
        example: `const rng = np.random.defaultRng(42);
rng.random(3); // => [0.7739560485559633, 0.4388784397520523, 0.8585979199113825]
typeof np.random.defaultRng(42).random(); // => "number"`,
      },
      {
        name: "Generator.integers",
        sig: "rng.integers(low, [high], [size], [dtype=\"int64\"], [endpoint=false])",
        desc: "Random integers in `[low, high)`, or `[low, high]` when `endpoint` is set. If `high` is omitted, the range is `[0, low)`.",
        args: [
          { name: "low, [high]", type: "number | bigint", desc: "Bounds." },
          { name: "[size]", type: "number | number[]", desc: "Output shape." },
          { name: "[dtype]", type: "DTypeLike", desc: "Integer dtype." },
          { name: "[endpoint]", type: "boolean", desc: "Include `high`." },
        ],
        returns: "NDArray | number",
        example: `np.random.defaultRng(42).integers(0, 10, 5); // => [0, 7, 6, 4, 4]`,
      },
      {
        name: "Generator.normal",
        sig: "rng.normal([loc=0], [scale=1], [size]) · rng.standardNormal([size]) · rng.uniform([low=0], [high=1], [size])",
        desc: "Gaussian and uniform samples. Every sampler also accepts an options object, e.g. `rng.normal({ loc, scale, size })`.",
        args: [
          { name: "[loc], [scale]", type: "number", desc: "Mean and standard deviation." },
          { name: "[low], [high]", type: "number", desc: "Uniform bounds, `[low, high)`." },
          { name: "[size]", type: "number | number[]", desc: "Output shape." },
        ],
        returns: "NDArray | number",
        example: `np.random.defaultRng(42).normal(0, 1, 2);  // => [0.30471707975443135, -1.0399841062404955]
np.random.defaultRng(42).uniform(5, 10, 2); // => [8.869780242779816, 7.194392198760261]
np.random.defaultRng(1).standardNormal([2, 3]).shape; // => [2, 3]`,
      },
      {
        name: "Generator.choice",
        sig: "rng.choice(a, [size], [replace=true])",
        desc: "Random sample from an array, or from `arange(a)` when `a` is an integer. The `p` weights argument is not supported yet and throws `NotImplementedError`.",
        args: [
          { name: "a", type: "number | ArrayLike", desc: "Population." },
          { name: "[size]", type: "number | number[]", desc: "Output shape." },
          { name: "[replace]", type: "boolean", desc: "Sample with replacement." },
        ],
        returns: "NDArray | number",
        example: `np.random.defaultRng(1).choice([10, 20, 30], 2, false); // => [10, 20]`,
      },
      {
        name: "Generator.permutation",
        sig: "rng.permutation(x, [axis=0]) · rng.shuffle(x, [axis=0])",
        desc: "`permutation` returns a shuffled copy (or a permutation of `arange(x)` for an integer). `shuffle` permutes a writeable `NDArray` in place.",
        args: [{ name: "x", type: "number | NDArray | NestedArray", desc: "Input." }, { name: "[axis]", type: "number", desc: "Axis to permute." }],
        returns: "NDArray | void",
        example: `np.random.defaultRng(0).permutation(5); // => [2, 4, 3, 0, 1]`,
      },
      {
        name: "random.seed",
        sig: "np.random.seed(s) · rand(...dims) · randn(...dims) · randint(low, [high], [size]) · random([size])",
        desc: "Legacy global `RandomState` API (MT19937), matching `numpy.random.*`. It also includes `random`, `randomSample`, `normal`, `uniform`, `standardNormal`, `choice`, `shuffle` and `permutation`. `new np.random.RandomState(seed)` gives an independent instance.",
        args: [{ name: "s", type: "number | number[]", desc: "Seed for the global state." }],
        returns: "void / NDArray | number",
        example: `np.random.seed(0);
np.random.rand(2);          // => [0.5488135039273248, 0.7151893663724195]
np.random.seed(0);
np.random.randint(0, 10, 3); // => [5, 0, 3]`,
      },
    ],
  },

  {
    id: "dtype",
    title: "Data types",
    intro: "Supported: `bool`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float16`, `float32`, `float64`, `complex64` and `complex128`. Each one is available as `np.<name>`, and anywhere a `DTypeLike` is accepted you can pass the name as a string. NumPy aliases `float`, `double`, `single`, `half`, `int`, `complex` and `bool_` are also accepted.",
    entries: [
      {
        name: "dtype",
        sig: "np.dtype(like)",
        desc: "Resolves a name or `DType` to the canonical `DType` singleton, which has `name`, `kind` (`b`/`i`/`u`/`f`/`c`), `itemSize` and `alignment`. Unknown names throw `DTypeError`.",
        args: [{ name: "like", type: "DTypeLike", desc: "Name, alias or DType." }],
        returns: "DType",
        example: `const d = np.dtype("single");
d.name;              // => "float32"
d.kind;              // => "f"
d.itemSize;          // => 4
d === np.float32;    // => true`,
      },
      {
        name: "promoteTypes",
        sig: "np.promoteTypes(a, b)",
        desc: "The smallest dtype that both inputs can be safely cast to, using NumPy's promotion table.",
        args: [{ name: "a, b", type: "DTypeLike", desc: "Input dtypes." }],
        returns: "DType",
        example: `np.promoteTypes("int32", "float32").name; // => "float64"
np.promoteTypes("uint8", "int8").name;    // => "int16"`,
      },
      {
        name: "canCast",
        sig: "np.canCast(from, to, casting = \"safe\")",
        desc: "Whether `from` can be cast to `to` under a NumPy casting rule: `\"no\"`, `\"equiv\"`, `\"safe\"` (the default), `\"same_kind\"` or `\"unsafe\"`. `from` can be a dtype or an array, in which case only its dtype is used. JS numbers raise `DTypeError`, as Python scalars do in NumPy 2. An unknown rule raises `ValueError`.",
        args: [
          { name: "from", type: "DTypeLike | NDArray", desc: "Source dtype, or an array whose dtype is used." },
          { name: "to", type: "DTypeLike", desc: "Target dtype." },
          { name: "casting", type: "string", desc: "Casting rule." },
        ],
        returns: "boolean",
        example: `np.canCast("int8", "int16");                   // => true
np.canCast("float64", "float32");              // => false
np.canCast("float64", "float32", "same_kind"); // => true
np.canCast("int64", "uint8", "same_kind");     // => false`,
      },
    ],
  },
  {
    id: "utilities",
    title: "Utilities",
    entries: [
      {
        name: "mayShareMemory",
        sig: "np.mayShareMemory(a, b)",
        desc: "`true` if the two arrays may share memory, i.e. one is a view of the other or both view the same buffer.",
        args: [{ name: "a, b", type: "NDArray", desc: "Arrays to compare." }],
        returns: "boolean",
        example: `const a = np.arange(6);
np.mayShareMemory(a, a.reshape(2, 3)); // => true
np.mayShareMemory(a, a.copy());        // => false`,
      },
      {
        name: "copy",
        sig: "np.copy(a, { order? })",
        desc: "A copy of `a`. `order` is `\"K\"` (default, keep the layout), `\"A\"`, `\"C\"` or `\"F\"`.",
        args: [arrayArg(), { name: "order", type: "string", desc: "Memory order of the copy." }],
        returns: "NDArray",
        example: `np.copy(np.ones([2, 3], { order: "F" })).strides; // => [8, 16]`,
      },
      {
        name: "ascontiguousarray",
        sig: "np.ascontiguousarray(a, { dtype? })",
        desc: "`a` as a C-contiguous array with at least one dimension; no copy when it already is.",
        args: [arrayArg()],
        returns: "NDArray",
        example: `np.ascontiguousarray(np.arange(6).reshape(2, 3).T).strides; // => [16, 8]`,
      },
      {
        name: "asfortranarray",
        sig: "np.asfortranarray(a, { dtype? })",
        desc: "`a` as an F-contiguous (column-major) array with at least one dimension; no copy when it already is.",
        args: [arrayArg()],
        returns: "NDArray",
        example: `np.asfortranarray(np.arange(6).reshape(2, 3)).strides; // => [8, 16]`,
      },
      {
        name: "memoryStats",
        sig: "np.memoryStats()",
        desc: "Count and total size of the native buffers currently alive. Native memory is freed when the JS objects are garbage-collected.",
        returns: "{ buffers: number, bytes: number }",
        example: `const s = np.memoryStats();
typeof s.bytes; // => "number"`,
      },
      {
        name: "lib.stride_tricks.asStrided",
        sig: "np.lib.stride_tricks.asStrided(a, [shape], [strides])",
        desc: "View of the same memory with arbitrary shape and byte strides. It is unchecked, as in NumPy: wrong strides can read outside the buffer's logical data.",
        args: [
          { name: "a", type: "NDArray", desc: "Base array." },
          { name: "[shape]", type: "number[]", desc: "View shape." },
          { name: "[strides]", type: "number[]", desc: "Strides in bytes." },
        ],
        returns: "NDArray",
        example: `const a = np.arange(0, 5, 1, { dtype: "int32" });
np.lib.stride_tricks.asStrided(a, [3, 3], [4, 4]); // => [[0, 1, 2], [1, 2, 3], [2, 3, 4]]`,
      },
    ],
  },
  {
    id: "errors",
    title: "Errors",
    intro: "Native errors never reach JS as raw C++ exceptions. Every failure is an instance of `np.NativpyError`, with a stable `code` string and a NumPy-style message. Subclasses: `ShapeError`, `DTypeError`, `IndexError`, `BroadcastError`, `ValueError`, `MemoryError`, `NotImplementedError`, `LinAlgError` and `FloatingPointError`.",
    entries: [
      {
        name: "NativpyError",
        sig: "err instanceof np.ShapeError · err.code",
        desc: "Catch a specific subclass, or `NativpyError` for any library error.",
        returns: "—",
        example: `try {
  np.zeros(2).reshape(3);
} catch (err) {
  err instanceof np.ShapeError;    // => true
  err instanceof np.NativpyError;  // => true
  err.code;                        // => "NATIVPY_SHAPE_ERROR"
  err.message;                     // => "cannot reshape array of size 2 into shape (3,)"
}`,
      },
      {
        name: "LinAlgError",
        sig: "np.LinAlgError · np.BroadcastError · np.IndexError",
        desc: "Common failures and the error types they raise.",
        returns: "—",
        example: `const code = (f) => { try { f(); } catch (e) { return e.name; } };
code(() => np.linalg.inv([[1, 2], [2, 4]])); // => "LinAlgError"
code(() => np.add([1, 2], [1, 2, 3]));       // => "BroadcastError"
code(() => np.array([1, 2]).get(5));         // => "IndexError"`,
      },
      {
        name: "FloatingPointError",
        sig: "np.FloatingPointError",
        desc: "Raised for a floating-point error whose `np.seterr` mode is `\"raise\"`.",
        returns: "—",
        example: `const code = (f) => { try { f(); } catch (e) { return e.name; } };
np.errstate({ divide: "raise" }, () => code(() => np.divide([1], [0]))); // => "FloatingPointError"`,
      },
    ],
  },
];


// Per-milestone parts (D-056): entries join the category with the same id,
// other categories are appended in milestone order.
function mergeParts(base, parts) {
  const out = base.map((c) => ({ ...c, entries: [...c.entries] }));
  for (const part of parts) {
    for (const cat of part) {
      const hit = out.find((c) => c.id === cat.id);
      if (hit) hit.entries.push(...cat.entries);
      else out.push({ ...cat, entries: [...cat.entries] });
    }
  }
  return out;
}

/** @type {Category[]} */
export const categories = mergeParts(baseCategories, [p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16a, p16c, p16d]);
