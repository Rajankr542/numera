// P5 bitwise ufuncs (D-082).
import { binaryUfunc, unaryUfunc, type BinaryUfunc, type UnaryUfunc } from "./ufunc.js";

/** NumPy bitwise_and: element-wise `&` of integers/bools (identity all ones). */
export const bitwiseAnd: BinaryUfunc = binaryUfunc("bitwiseAnd");
/** NumPy bitwise_or: element-wise `|` of integers/bools. */
export const bitwiseOr: BinaryUfunc = binaryUfunc("bitwiseOr");
/** NumPy bitwise_xor: element-wise `^` of integers/bools. */
export const bitwiseXor: BinaryUfunc = binaryUfunc("bitwiseXor");
/** NumPy invert: element-wise `~` (logical NOT for bool). */
export const invert: UnaryUfunc = unaryUfunc("invert");
/** NumPy bitwise_not (the same ufunc as invert). */
export const bitwiseNot: UnaryUfunc = invert;
/** NumPy bitwise_invert (the same ufunc as invert). */
export const bitwiseInvert: UnaryUfunc = invert;
/** NumPy left_shift: `a << b`; counts outside [0, bits) give 0. */
export const leftShift: BinaryUfunc = binaryUfunc("leftShift");
/** NumPy right_shift: arithmetic `a >> b`; counts outside [0, bits) give 0 or -1. */
export const rightShift: BinaryUfunc = binaryUfunc("rightShift");
/** NumPy bitwise_left_shift (the same ufunc as leftShift). */
export const bitwiseLeftShift: BinaryUfunc = leftShift;
/** NumPy bitwise_right_shift (the same ufunc as rightShift). */
export const bitwiseRightShift: BinaryUfunc = rightShift;
/** NumPy bitwise_count: number of 1 bits in `|x|` (uint8 result). */
export const bitwiseCount: UnaryUfunc = unaryUfunc("bitwiseCount");
