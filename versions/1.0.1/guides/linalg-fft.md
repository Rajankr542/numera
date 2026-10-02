# Linear algebra and FFT

## Matrix products

```js
np.matmul([[1, 2], [3, 4]], [[1], [1]]);  // => [[3], [7]]
np.dot([1, 2], [3, 4]).item();            // => 11
np.inner([1, 2], [3, 4]).item();          // => 11
np.outer([1, 2], [1, 10]);                // => [[1, 10], [2, 20]]
np.matmul(np.ones([5, 2, 3]), np.ones([3, 4])).shape; // => [5, 2, 4]
```

## np.linalg

{{version}} has `det`, `inv`, `solve`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `svd`, `qr`, `lstsq`, `norm`, `matmul` and `backend`.

```js
np.linalg.det([[3, 1], [1, 2]]).item();        // => 5
np.linalg.inv([[2, 0], [0, 4]]);               // => [[0.5, 0], [0, 0.25]]
np.linalg.solve([[3, 1], [1, 2]], [9, 8]);     // => [2, 3]
np.linalg.norm([3, 4]).item();                 // => 5
np.linalg.eigh([[2, 1], [1, 2]]).eigenvalues;  // => [1, 3]
np.linalg.svd([[3, 0], [0, 4]]).S;             // => [4, 3]
Object.keys(np.linalg.qr([[1, 2], [3, 4]]));   // => ["Q", "R"]
Object.keys(np.linalg.lstsq([[1, 0], [0, 1], [1, 1]], [1, 2, 3])); // => ["x", "residuals", "rank", "s"]
np.linalg.eig([[2, 0], [0, 3]]).eigenvalues.dtype.name; // => "complex128"
```

A singular matrix raises `np.LinAlgError`:

```js
let err;
try { np.linalg.inv([[1, 2], [2, 4]]); } catch (e) { err = e; }
err instanceof np.LinAlgError; // => true
err.message;                   // => "Singular matrix"
```

On macOS, `np.linalg` uses Apple Accelerate. Elsewhere it uses numera's portable C++ implementation. `np.linalg.backend()` reports which.

## FFT

{{version}} has `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `fftfreq` and `rfftfreq`.

```js
const X = np.fft.fft([1, 2, 3, 4]);
X.dtype.name;                            // => "complex128"
Array.from(X.toTypedArray());            // => [10, 0, -2, 2, -2, 0, -2, -2]
np.fft.irfft(np.fft.rfft([1, 2, 3, 4])); // => [1, 2, 3, 4]
np.fft.fftfreq(4);                       // => [0, 0.25, -0.5, -0.25]
```

> **Note**: In 1.0.1, complex elements cannot be converted with `toArray()` or `item()`, and `np.abs`/`np.real`/`np.imag` do not accept complex arrays. Read complex results with `toTypedArray()`, which interleaves `re` and `im`.
