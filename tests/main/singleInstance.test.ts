import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { filesAskedFor } from "../../src/main/singleInstance";

describe("the files a second launch was asked to open", () => {
  let folder: string;
  beforeEach(() => {
    folder = mkdtempSync(join(tmpdir(), "incarnamind-argv-"));
  });
  afterEach(() => rmSync(folder, { recursive: true, force: true }));

  test("are the arguments that are files, by absolute path or from where the launch ran", () => {
    const report = join(folder, "Quarterly report.pdf");
    writeFileSync(report, "x");
    writeFileSync(join(folder, "notes.md"), "x");
    expect(filesAskedFor(["/Applications/IncarnaMind", report, "notes.md"], folder)).toEqual([
      report,
      join(folder, "notes.md"),
    ]);
  });

  test("leave out switches, folders and what isn't there", () => {
    mkdirSync(join(folder, "Reports"));
    expect(
      filesAskedFor(
        ["exe", "--no-sandbox", "--user-data-dir=/tmp/x", "Reports", folder, "gone.pdf"],
        folder,
      ),
    ).toEqual([]);
  });

  test("keep Chinese names whole", () => {
    const file = join(folder, "财务报表 2026.xlsx");
    writeFileSync(file, "x");
    expect(filesAskedFor(["exe", file], folder)).toEqual([file]);
  });
});
