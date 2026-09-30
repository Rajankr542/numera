import { describe, expect, it } from "vitest";
// @ts-expect-error -- plain ESM release script without type declarations
import { commitLevel, computeNext } from "../../../scripts/next-version.mjs";

const c = (subject: string, body = "") => ({ subject, body });

describe("release: commitLevel (Conventional Commits, D-029)", () => {
  it("maps commit types to bump levels", () => {
    expect(commitLevel("fix: x")).toBe("patch");
    expect(commitLevel("perf(reduce): x")).toBe("patch");
    expect(commitLevel("feat(fft): x")).toBe("minor");
    expect(commitLevel("feat!: x")).toBe("major");
    expect(commitLevel("refactor(api)!: x")).toBe("major");
    expect(commitLevel("fix: x", "BREAKING CHANGE: y")).toBe("major");
    expect(commitLevel("docs: x")).toBe("none");
    expect(commitLevel("chore: x")).toBe("none");
    expect(commitLevel("test: x")).toBe("none");
    expect(commitLevel("chore(release): @cyfora/numera@1.0.1")).toBe("none");
    expect(commitLevel("Merge branch 'beta'")).toBe("none");
  });
});

describe("release: computeNext", () => {
  const tags = ["v1.0.0"];

  it("publishes nothing when only docs/chore commits landed", () => {
    const r = computeNext({ channel: "latest", tags, commits: [c("docs: a"), c("chore: b")] });
    expect(r.release).toBe(false);
  });

  it("uses the highest bump of all commits for latest", () => {
    expect(computeNext({ channel: "latest", tags, commits: [c("fix: a")] }).version).toBe("1.0.1");
    expect(computeNext({ channel: "latest", tags, commits: [c("fix: a"), c("feat: b")] }).version).toBe("1.1.0");
    expect(computeNext({ channel: "latest", tags, commits: [c("feat: b"), c("fix!: c")] }).version).toBe("2.0.0");
  });

  it("picks the newest stable tag, ignoring prerelease tags as a base", () => {
    const r = computeNext({ channel: "latest", tags: ["v1.0.0", "v1.2.0", "v1.10.0", "v1.11.0-beta.0"], commits: [c("fix: a")] });
    expect(r.base).toBe("1.10.0");
    expect(r.version).toBe("1.10.1");
  });

  it("numbers prereleases per channel past existing tags and npm versions", () => {
    const commits = [c("feat: a")];
    expect(computeNext({ channel: "beta", tags, commits }).version).toBe("1.1.0-beta.0");
    expect(computeNext({ channel: "beta", tags: [...tags, "v1.1.0-beta.0"], commits }).version).toBe("1.1.0-beta.1");
    expect(
      computeNext({ channel: "beta", tags, commits, published: ["1.0.0", "1.1.0-beta.0", "1.1.0-beta.3"] }).version,
    ).toBe("1.1.0-beta.4");
    expect(computeNext({ channel: "alpha", tags: [...tags, "v1.1.0-beta.2"], commits }).version).toBe("1.1.0-alpha.0");
    expect(computeNext({ channel: "alpha", tags, commits, published: ["1.1.0-alphabet.9"] }).version).toBe(
      "1.1.0-alpha.0",
    );
  });

  it("starts at 1.0.0 with no tags", () => {
    expect(computeNext({ channel: "latest", tags: [], commits: [c("docs: a")] }).version).toBe("1.0.0");
    expect(computeNext({ channel: "alpha", tags: [], commits: [] }).version).toBe("1.0.0-alpha.0");
  });

  it("lets a manual run force a bump level", () => {
    expect(computeNext({ channel: "latest", tags, commits: [c("docs: a")], force: "patch" }).version).toBe("1.0.1");
    expect(computeNext({ channel: "beta", tags, commits: [c("fix: a")], force: "major" }).version).toBe("2.0.0-beta.0");
    expect(() => computeNext({ channel: "latest", tags, commits: [], force: "none" })).toThrow(/force/);
  });

  it("rejects unknown channels", () => {
    expect(() => computeNext({ channel: "next", tags, commits: [] })).toThrow(/channel/);
  });
});
