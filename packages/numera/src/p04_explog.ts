import { binaryUfunc, unaryUfunc, type BinaryUfunc, type UnaryUfunc } from "./ufunc.js";

// P4-2: exponential / logarithm family (native rows in ufunc_math.cpp).

export const exp2: UnaryUfunc = unaryUfunc("exp2");
export const expm1: UnaryUfunc = unaryUfunc("expm1");
export const log2: UnaryUfunc = unaryUfunc("log2");
export const log10: UnaryUfunc = unaryUfunc("log10");
export const log1p: UnaryUfunc = unaryUfunc("log1p");
export const logaddexp: BinaryUfunc = binaryUfunc("logaddexp");
export const logaddexp2: BinaryUfunc = binaryUfunc("logaddexp2");
export const cbrt: UnaryUfunc = unaryUfunc("cbrt");
export const square: UnaryUfunc = unaryUfunc("square");
export const reciprocal: UnaryUfunc = unaryUfunc("reciprocal");
