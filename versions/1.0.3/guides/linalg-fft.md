# Linear algebra and FFT

## Matrix products

```js
const A = np.array([[1, 2], [3, 4]]);
np.matmul(A, [[1], [1]]);          // => [[3], [7]]
A.dot([1, 1]);                     // => [3, 7]
np.dot([1, 2], [3, 4]).item();     // => 11
np.outer([1, 2], [1, 10]);         // => [[1, 10], [2, 20]]
np.cross([1, 0, 0], [0, 1, 0]);    // => [0, 0, 1]
np.kron([1, 2], [1, 1]);           // => [1, 1, 2, 2]
np.einsum("ij,jk->ik", [[1, 2], [3, 4]], [[1], [1]]); // => [[3], [7]]
np.einsum("ii", [[1, 2], [3, 4]]).item();             // => 5
```

`np.matmul` broadcasts the batch dimensions, that is, everything except the last two axes. 1-D operands are treated as vectors, as in NumPy.

## np.linalg

Every routine accepts batched (stacked) input of shape `[..., M, N]`:

```js
np.linalg.det([[3, 1], [1, 2]]).item();        // => 5.000000000000001
np.linalg.inv([[2, 0], [0, 4]]);               // => [[0.5, 0], [0, 0.25]]
np.linalg.solve([[3, 1], [1, 2]], [9, 8]);     // => [2, 3]
np.linalg.norm([3, 4]).item();                 // => 5
np.linalg.matrixRank([[1, 2], [2, 4]]).item(); // => 1
np.linalg.matrixPower([[1, 1], [0, 1]], 3);    // => [[1, 3], [0, 1]]
np.linalg.cholesky([[4, 0], [0, 9]]);          // => [[2, 0], [0, 3]]
np.linalg.det(np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])); // => [1, 4]
```

## Decompositions

Results that NumPy returns as named tuples are objects with the same field names:

```js
np.linalg.eigh([[2, 1], [1, 2]]).eigenvalues;  // => [1, 3]
np.linalg.svd([[3, 0], [0, 4]]).S;             // => [4, 3]
Object.keys(np.linalg.svd([[1, 2], [3, 4]]));  // => ["U", "S", "Vh"]
Object.keys(np.linalg.qr([[1, 2], [3, 4]]));   // => ["Q", "R"]
np.linalg.eig([[2, 0], [0, 3]]).eigenvalues.dtype.name; // => "complex128"
```

| Function | Returns |
| --- | --- |
| `eig(a)` | `{ eigenvalues, eigenvectors }` (always complex) |
| `eigh(a)` | `{ eigenvalues, eigenvectors }` (ascending, real eigenvalues) |
| `svd(a, { fullMatrices, computeUv, hermitian })` | `{ U, S, Vh }` |
| `qr(a, { mode })` | `{ Q, R }` |
| `slogdet(a)` | `{ sign, logabsdet }` |
| `lstsq(a, b)` | `{ x, residuals, rank, s }` |

## Errors

A singular matrix raises `np.LinAlgError`:

```js
let err;
try { np.linalg.inv([[1, 2], [2, 4]]); } catch (e) { err = e; }
err instanceof np.LinAlgError; // => true
err.message;                   // => "Singular matrix"
```

## Backends

On macOS, `np.linalg` and large float matrix products use Apple Accelerate (LAPACK and BLAS). Elsewhere they use numera's portable C++ implementation. `np.linalg.backend()` reports which backend is active. The differential tests run on both backends. Because eigenvector signs and phases can differ between backends (they can also differ between NumPy builds), the tests compare decompositions by reconstruction rather than element by element.

## FFT

`np.fft` has the full set: `fft`, `ifft`, `rfft`, `irfft`, `hfft`, `ihfft`, the 2-D and n-D variants, `fftfreq`, `rfftfreq`, `fftshift` and `ifftshift`. The `n`, `axis`, `norm`, `s` and `axes` options work as in NumPy.

```js
const X = np.fft.fft([1, 2, 3, 4]);
X.dtype.name;                         // => "complex128"
np.abs(X);                            // => [10, 2.8284271247461903, 2, 2.8284271247461903]
X.item(1);                            // => {"re": -2, "im": 2}
X.toArray().map(String);              // => ["(10+0j)", "(-2+2j)", "(-2+0j)", "(-2-2j)"]
np.fft.irfft(np.fft.rfft([1, 2, 3, 4])); // => [1, 2, 3, 4]
np.fft.fftfreq(4);                    // => [0, 0.25, -0.5, -0.25]
np.fft.fftshift([0, 1, 2, 3]);        // => [2, 3, 0, 1]
np.fft.fft([1, 2, 3, 4], 8).shape;    // => [8]
```

Complex results come back from `toArray()` and `item()` as `Complex` objects with `re` and `im` fields. `toTypedArray()` returns interleaved `[re, im, re, im, …]` values. Use `np.real`, `np.imag`, `np.abs` and `np.angle` to split them:

```js
const z = np.array([np.complex(1, 2)]);
np.real(z);  // => [1]
np.imag(z);  // => [2]
np.conj(z).toArray().map(String); // => ["(1-2j)"]
```
