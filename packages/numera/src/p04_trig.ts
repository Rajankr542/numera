import { binaryUfunc, unaryUfunc, type BinaryUfunc, type UnaryUfunc } from "./ufunc.js";

// P4-1: trigonometric and hyperbolic ufuncs (native rows in ufunc_math.cpp).

export const sin: UnaryUfunc = unaryUfunc("sin");
export const cos: UnaryUfunc = unaryUfunc("cos");
export const tan: UnaryUfunc = unaryUfunc("tan");
export const arcsin: UnaryUfunc = unaryUfunc("arcsin");
export const arccos: UnaryUfunc = unaryUfunc("arccos");
export const arctan: UnaryUfunc = unaryUfunc("arctan");
export const sinh: UnaryUfunc = unaryUfunc("sinh");
export const cosh: UnaryUfunc = unaryUfunc("cosh");
export const tanh: UnaryUfunc = unaryUfunc("tanh");
export const arcsinh: UnaryUfunc = unaryUfunc("arcsinh");
export const arccosh: UnaryUfunc = unaryUfunc("arccosh");
export const arctanh: UnaryUfunc = unaryUfunc("arctanh");
export const arctan2: BinaryUfunc = binaryUfunc("arctan2");
export const hypot: BinaryUfunc = binaryUfunc("hypot");
export const deg2rad: UnaryUfunc = unaryUfunc("deg2rad");
export const rad2deg: UnaryUfunc = unaryUfunc("rad2deg");
export const radians: UnaryUfunc = unaryUfunc("radians");
export const degrees: UnaryUfunc = unaryUfunc("degrees");

// Array-API aliases (same objects, as in NumPy 2).
export const asin = arcsin;
export const acos = arccos;
export const atan = arctan;
export const asinh = arcsinh;
export const acosh = arccosh;
export const atanh = arctanh;
export const atan2 = arctan2;

