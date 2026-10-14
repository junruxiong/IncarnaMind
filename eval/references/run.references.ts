/**
 * The reference check (#711) against Crossref itself: each of the 50
 * references in references.json checked as the app checks one, and its mark
 * compared with what it should be. Reports, for "not found" and "doesn't
 * match", precision (of the references marked so, how many should be) and
 * recall (of those that should be, how many are), and every reference marked
 * otherwise than expected. Writes eval/results/<time>-references.md.
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "vitest";
import type { ReferenceCheck, ReferenceCheckMark } from "../../src/core/api";
import { checkReference } from "../../src/core/referenceChecks/check";
import { createCrossrefClient } from "../../src/core/referenceChecks/crossref";
import { citedReference } from "../../src/core/referenceChecks/match";
import { createSourceHttp } from "../../src/core/sources/http";
import { createPace } from "../../src/core/sources/pace";

const root = fileURLToPath(new URL("../..", import.meta.url));

interface Expected {
  id: string;
  reference: string;
  expected: Exclude<ReferenceCheckMark, "not-checked">;
  note: string;
}

const percent = (hits: number, total: number) =>
  total === 0 ? "–" : `${hits}/${total} (${Math.round((100 * hits) / total)}%)`;

test("references against Crossref", async () => {
  const set = JSON.parse(readFileSync(join(root, "eval/references/references.json"), "utf8")) as {
    references: Expected[];
  };
  const crossref = createCrossrefClient(createSourceHttp(createPace()));
  const signal = new AbortController().signal;
  const results: { item: Expected; check: ReferenceCheck }[] = [];
  for (const item of set.references) {
    let check: ReferenceCheck;
    try {
      check = await checkReference(citedReference(item.reference), crossref, signal);
    } catch (error) {
      check = {
        reference: item.reference,
        mark: "not-checked",
        record: null,
        differences: [],
        reason: "service-error",
      };
      console.error(`${item.id}: ${error instanceof Error ? error.message : String(error)}`);
    }
    results.push({ item, check });
    console.log(`${item.id}: expected ${item.expected}, marked ${check.mark}`);
  }

  const lines = [
    `# Reference check against Crossref, ${new Date().toISOString()}`,
    "",
    "| Mark | Precision | Recall |",
    "|---|---|---|",
  ];
  for (const mark of ["not-found", "mismatch"] as const) {
    const marked = results.filter(({ check }) => check.mark === mark);
    const should = results.filter(({ item }) => item.expected === mark);
    const right = marked.filter(({ item }) => item.expected === mark).length;
    lines.push(`| ${mark} | ${percent(right, marked.length)} | ${percent(right, should.length)} |`);
  }
  const found = results.filter(({ item }) => item.expected === "found");
  lines.push(
    "",
    `Real references marked found: ${percent(found.filter(({ check }) => check.mark === "found").length, found.length)}. Not checked: ${results.filter(({ check }) => check.mark === "not-checked").length}.`,
    "",
    "Marked otherwise than expected:",
    "",
    ...results
      .filter(({ item, check }) => item.expected !== check.mark)
      .map(
        ({ item, check }) =>
          `- ${item.id} (${item.note}): expected ${item.expected}, marked ${check.mark}${check.record ? `; Crossref: “${check.record.title}”, ${check.record.year ?? "no year"}, ${check.record.authors.slice(0, 2).join("; ")}` : ""}${check.differences.length > 0 ? `; differs in ${check.differences.map((each) => each.field).join(", ")}` : ""}`,
      ),
    "",
  );
  const report = lines.join("\n");
  mkdirSync(join(root, "eval/results"), { recursive: true });
  const file = join(
    root,
    "eval/results",
    `${new Date().toISOString().replace(/[:.]/g, "-")}-references.md`,
  );
  writeFileSync(file, report);
  console.log(`\n${report}\nWritten to ${file}`);
});
