/**
 * Typed errors raised by nativpy (PLAN §28, DECISIONS D-006).
 * Native errors arrive as plain JS Errors with `name` set to the kind; they are
 * re-thrown as instances of these classes by `translateNativeError`.
 */
export class NativpyError extends Error {
  readonly code: string;
  constructor(message: string, code = "NATIVPY_ERROR") {
    super(message);
    this.name = new.target.name;
    this.code = code;
  }
}

export class ShapeError extends NativpyError {}
export class DTypeError extends NativpyError {}
export class IndexError extends NativpyError {}
export class BroadcastError extends NativpyError {}
export class ValueError extends NativpyError {}
export class MemoryError extends NativpyError {}
export class NotImplementedError extends NativpyError {}
/** numpy.linalg.LinAlgError (DECISIONS D-018). */
export class LinAlgError extends NativpyError {}
/** numpy FloatingPointError: raised under np.seterr "raise" (D-054). */
export class FloatingPointError extends NativpyError {}

const byName: Record<string, new (message: string, code?: string) => NativpyError> = {
  ShapeError,
  DTypeError,
  IndexError,
  BroadcastError,
  ValueError,
  MemoryError,
  NotImplementedError,
  LinAlgError,
  FloatingPointError,
  NativpyError,
};

/** Converts an error thrown by the native addon into a typed nativpy error. */
export function translateNativeError(err: unknown): unknown {
  if (err instanceof NativpyError) return err;
  if (err instanceof Error) {
    const ctor = byName[err.name];
    const code = (err as { code?: unknown }).code;
    if (ctor !== undefined && typeof code === "string" && code.startsWith("NATIVPY")) {
      return new ctor(err.message, code);
    }
  }
  return err;
}

/** Runs `fn`, converting native errors into typed nativpy errors. */
export function wrapNative<T>(fn: () => T): T {
  try {
    return fn();
  } catch (err) {
    throw translateNativeError(err);
  }
}
