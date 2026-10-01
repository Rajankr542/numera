import { addon } from "./addon.js";
import { ValueError, wrapNative } from "./errors.js";

/**
 * NumPy floating-point error handling (np.seterr / np.geterr / np.errstate, D-054).
 * Modes: "ignore", "warn" (a Node `RuntimeWarning` via `process.emitWarning`),
 * "raise" (`FloatingPointError`) and "print" (stdout). "call"/"log" and
 * `seterrcall` are not supported.
 */
export type ErrMode = "ignore" | "warn" | "raise" | "print";

export interface ErrState {
  divide: ErrMode;
  over: ErrMode;
  under: ErrMode;
  invalid: ErrMode;
}

/** `all` sets every category; the named ones override it. */
export type ErrSettings = Partial<ErrState> & { all?: ErrMode };

const MODES = new Set(["ignore", "warn", "raise", "print"]);
const KEYS = new Set(["all", "divide", "over", "under", "invalid"]);

/** NumPy geterr: the current settings. */
export function geterr(): ErrState {
  return addon.getErr() as ErrState;
}

/** NumPy seterr: updates the settings and returns the previous ones. */
export function seterr(settings: ErrSettings = {}): ErrState {
  const old = geterr();
  const next: Record<string, string> = {};
  for (const [k, v] of Object.entries(settings)) {
    if (!KEYS.has(k)) throw new ValueError(`seterr() got an unexpected keyword argument '${k}'`);
    if (v === undefined) continue;
    if (!MODES.has(v as string)) throw new ValueError(`invalid error mode '${String(v)}' for ${k}`);
  }
  for (const k of ["divide", "over", "under", "invalid"] as const) {
    const v = settings[k] ?? settings.all;
    if (v !== undefined) next[k] = v;
  }
  wrapNative(() => addon.setErr(next));
  return old;
}

/**
 * NumPy errstate: runs `fn` synchronously with the given settings and
 * restores the previous ones afterwards (also when `fn` throws). Settings
 * are global, so they do not follow `await` inside `fn`.
 */
export function errstate<T>(settings: ErrSettings, fn: () => T): T {
  const old = seterr(settings);
  try {
    return fn();
  } finally {
    seterr(old);
  }
}
