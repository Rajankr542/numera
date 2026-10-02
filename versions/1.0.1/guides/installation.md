# Installation

## Install the package

```bash
npm install @cyfora/numera
```

```bash
pnpm add @cyfora/numera
```

```bash
yarn add @cyfora/numera
```

The package ships a precompiled C++ addon for each supported platform. When you `import "@cyfora/numera"`, it loads the matching binary. You do **not** need a compiler, CMake or Python.

## Supported platforms

| Platform | Architectures | Node.js |
| --- | --- | --- |
| macOS 13.3+ | arm64 (Apple Silicon), x64 (Intel) | {{node}} |
| Linux, glibc ≥ 2.28 (Ubuntu 20.04+, Debian 10+, RHEL 8+) | x64, arm64 | {{node}} |

Windows and Alpine/musl Linux have no prebuilt binaries yet. The prebuilds use the stable Node-API, so one binary works on every supported Node.js version.

> **Note**: numera is a native Node.js addon. It does not run in browsers, Deno or Bun.

## Importing

numera is an ES module. The default export is the `np` namespace, the same object you would get from `import numpy as np`:

```js
import np from "@cyfora/numera";

np.arange(4);   // => [0, 1, 2, 3]
```

Named imports work too:

```js
import { array, zeros, linalg } from "@cyfora/numera";

zeros([2]);                        // => [0, 0]
linalg.det(array([[1, 2], [3, 4]])).item(); // => -2
```

In a CommonJS project, use a dynamic import: `const { default: np } = await import("@cyfora/numera")`.

## Check the installation

```js
import np from "@cyfora/numera";

np.linalg.solve([[3, 1], [1, 2]], [9, 8]); // => [2, 3]
typeof np.linalg.backend();                // => "string"
```

`np.linalg.backend()` returns `"accelerate"` on macOS, where linear algebra runs on Apple Accelerate, and `"fallback"` elsewhere, where it uses numera's portable C++ implementation.

## TypeScript

Type declarations are included, so you do not need an `@types` package. Every function has a typed signature, and errors are exported as classes.
