import { addon } from "./addon.js";

/**
 * A complex scalar (DECISIONS D-033). Complex arrays return these from
 * `toArray()` and `item()`. As input, anything shaped `{ re, im }` is accepted.
 */
export class Complex {
  readonly re: number;
  readonly im: number;
  constructor(re: number, im = 0) {
    this.re = re;
    this.im = im;
    Object.freeze(this);
  }
  /** NumPy-style repr, e.g. `(1+2j)`, `-1.5j`, `(nan+infj)`. */
  toString(): string {
    const f = (x: number): string =>
      Number.isNaN(x) ? "nan" : x === Infinity ? "inf" : x === -Infinity ? "-inf" : Object.is(x, -0) ? "-0" : String(x);
    if (this.re === 0 && !Object.is(this.re, -0)) return `${f(this.im)}j`;
    const sign = this.im < 0 || Object.is(this.im, -0) ? "-" : "+";
    return `(${f(this.re)}${sign}${f(Math.abs(this.im))}j)`;
  }
  toJSON(): { re: number; im: number } {
    return { re: this.re, im: this.im };
  }
}

/** A `{ re, im }` value accepted wherever a complex element is expected. */
export type ComplexLike = Complex | { readonly re: number; readonly im?: number };

/** np.complex(re, im): a complex scalar. */
export function complex(re: number, im = 0): Complex {
  return new Complex(re, im);
}

/** True for `Complex` and `{ re: number }` objects (D-033). */
export function isComplexLike(v: unknown): v is ComplexLike {
  return typeof v === "object" && v !== null && !Array.isArray(v) && typeof (v as { re?: unknown }).re === "number";
}

addon.setComplexClass(Complex);
