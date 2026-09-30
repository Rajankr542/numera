// Next release version from Conventional Commits (DECISIONS D-029).
// Usage: node scripts/next-version.mjs <latest|beta|alpha> [--published '<json array>'] [--bump auto|patch|minor|major]
// Prints JSON {release, version, distTag, bump, base, commits}; in GitHub Actions
// also writes release/version/dist_tag to $GITHUB_OUTPUT.
//
// Rules: commits since the newest stable tag vX.Y.Z merged into HEAD decide the
// bump: "type!:" or "BREAKING CHANGE" -> major, feat -> minor, fix/perf/revert
// -> patch; anything else (docs, chore, test, ci, build, refactor, style) -> no
// release. "latest" publishes X.Y.Z; beta/alpha publish X.Y.Z-<channel>.N with N
// one past the highest tag or npm version already taken for that base.
import { execFileSync } from "node:child_process";
import { appendFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

export const CHANNELS = ["latest", "beta", "alpha"];
const LEVELS = ["none", "patch", "minor", "major"];
const STABLE = /^v?(\d+)\.(\d+)\.(\d+)$/;

/** Bump level of one commit: "none" | "patch" | "minor" | "major". */
export function commitLevel(subject, body = "") {
  if (/^chore\(release\)/.test(subject)) return "none";
  const m = /^(\w+)(\([^)]*\))?(!)?:/.exec(subject);
  if (m?.[3] || /^BREAKING[ -]CHANGE:/m.test(body)) return "major";
  if (!m) return "none";
  if (m[1] === "feat") return "minor";
  if (["fix", "perf", "revert"].includes(m[1])) return "patch";
  return "none";
}

function bump([a, b, c], level) {
  if (level === "major") return [a + 1, 0, 0];
  if (level === "minor") return [a, b + 1, 0];
  return [a, b, c + 1];
}

/**
 * @param {{channel: string, tags: string[], commits: {subject: string, body?: string}[], published?: string[]}} o
 * `tags`: stable tags merged into HEAD plus all prerelease tags; `commits`: those since the newest stable tag.
 */
export function computeNext({ channel, tags, commits, published = [], force = null }) {
  if (!CHANNELS.includes(channel)) throw new Error(`channel must be one of ${CHANNELS.join(", ")}`);
  if (force !== null && !LEVELS.slice(1).includes(force)) throw new Error("force must be patch, minor or major");
  const stable = tags
    .map((t) => STABLE.exec(t))
    .filter(Boolean)
    .map((m) => [Number(m[1]), Number(m[2]), Number(m[3])])
    .sort((x, y) => x[0] - y[0] || x[1] - y[1] || x[2] - y[2]);
  const last = stable.at(-1) ?? null;
  const fromCommits = commits.reduce((acc, c) => {
    const l = commitLevel(c.subject, c.body);
    return LEVELS.indexOf(l) > LEVELS.indexOf(acc) ? l : acc;
  }, "none");
  const level = force ?? fromCommits;
  const base = last ? last.join(".") : null;
  const none = { release: false, version: "", distTag: channel, bump: level, base, commits: commits.length };
  if (level === "none" && last) return none;
  const next = (last ? bump(last, level) : [1, 0, 0]).join(".");
  if (channel === "latest") return { ...none, release: true, version: next };
  const taken = new Set([...tags.map((t) => t.replace(/^v/, "")), ...published]);
  const prefix = `${next}-${channel}.`;
  let n = 0;
  for (const v of taken) {
    if (v.startsWith(prefix) && /^\d+$/.test(v.slice(prefix.length))) n = Math.max(n, Number(v.slice(prefix.length)) + 1);
  }
  return { ...none, release: true, version: `${prefix}${n}` };
}

function git(args) {
  return execFileSync("git", args, { encoding: "utf8" }).trim();
}

/** Reads tags and commits from the git repository in the current directory. */
export function readGit() {
  const tags = git(["tag", "--merged", "HEAD", "--list", "v*"]).split("\n").filter(Boolean);
  const stable = tags.filter((t) => STABLE.test(t)).sort((x, y) => {
    const [a, b] = [x, y].map((t) => STABLE.exec(t).slice(1).map(Number));
    return a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
  });
  const range = stable.length ? [`${stable.at(-1)}..HEAD`] : ["HEAD"];
  const log = git(["log", "--format=%s%x1f%b%x1e", ...range]);
  const commits = log
    .split("\x1e")
    .map((e) => e.trim())
    .filter(Boolean)
    .map((e) => {
      const [subject, body = ""] = e.split("\x1f");
      return { subject: subject.trim(), body };
    });
  // Base = stable tags reachable from HEAD; prerelease numbers also count tags
  // made on other branches, so a number is never reused.
  const prereleases = git(["tag", "--list", "v*"]).split("\n").filter((t) => t && !STABLE.test(t));
  return { tags: [...new Set([...stable, ...prereleases])], commits };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const channel = args[0];
  const pubIdx = args.indexOf("--published");
  const raw = pubIdx >= 0 ? args[pubIdx + 1] : "[]";
  const parsed = raw ? JSON.parse(raw) : [];
  const published = Array.isArray(parsed) ? parsed : [parsed];
  const bumpIdx = args.indexOf("--bump");
  const bumpArg = bumpIdx >= 0 ? args[bumpIdx + 1] : "auto";
  const force = !bumpArg || bumpArg === "auto" ? null : bumpArg;
  const result = computeNext({ channel, published, force, ...readGit() });
  console.log(JSON.stringify(result));
  if (process.env.GITHUB_OUTPUT) {
    appendFileSync(
      process.env.GITHUB_OUTPUT,
      `release=${result.release}\nversion=${result.version}\ndist_tag=${result.distTag}\n`,
    );
  }
}
