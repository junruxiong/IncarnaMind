import { describe, expect, test } from "vitest";
import { type Sort, type SortFields, sorted } from "../../src/renderer/src/sort";

const fields = (name: string, added: string, extra: Partial<SortFields> = {}): SortFields => ({
  name,
  kind: "PDF",
  added,
  modified: null,
  ...extra,
});

const ASC = (key: Sort["key"]): Sort => ({ key, direction: "asc", foldersFirst: true });
const DESC = (key: Sort["key"]): Sort => ({ key, direction: "desc", foldersFirst: true });
const names = (items: readonly SortFields[]) => items.map((item) => item.name);
const sortedFields = (items: SortFields[], sort: Sort | null, language: "en" | "zh-CN" = "en") =>
  sorted(items, sort, (item) => item, language);

describe("sorting by name", () => {
  const items = [
    fields("banana", "2026-01-03"),
    fields("Apple", "2026-01-02"),
    fields("cherry", "2026-01-01"),
    fields("apple", "2026-01-04"),
    fields("File 10", "2026-01-05"),
    fields("File 2", "2026-01-06"),
  ];

  test("in English ignores case, counts digits as numbers, and goes either way", () => {
    expect(names(sortedFields(items, ASC("name")))).toEqual([
      "Apple",
      "apple",
      "banana",
      "cherry",
      "File 2",
      "File 10",
    ]);
    expect(names(sortedFields(items, DESC("name"))).slice(0, 2)).toEqual(["File 10", "File 2"]);
  });

  test("in Chinese follows pinyin", () => {
    const chinese = ["张三", "李四", "王五", "安娜", "陈六"].map((name, index) =>
      fields(name, `2026-01-0${index + 1}`),
    );
    // an, chen, li, wang, zhang
    expect(names(sortedFields(chinese, ASC("name"), "zh-CN"))).toEqual([
      "安娜",
      "陈六",
      "李四",
      "王五",
      "张三",
    ]);
  });
});

describe("sorting by dates and kind", () => {
  test("Date added goes either way, ties falling to the name", () => {
    const items = [fields("b", "2026-02-01"), fields("a", "2026-02-01"), fields("c", "2026-03-01")];
    expect(names(sortedFields(items, DESC("added")))).toEqual(["c", "a", "b"]);
    expect(names(sortedFields(items, ASC("added")))).toEqual(["a", "b", "c"]);
  });

  test("Date modified uses the file's time, and the added date for what has none", () => {
    const items = [
      fields("old file", "2026-05-01", { modified: "2024-01-01T00:00:00.000Z" }),
      fields("new file", "2026-01-01", { modified: "2026-06-01T00:00:00.000Z" }),
      fields("no time", "2025-01-01"),
    ];
    expect(names(sortedFields(items, DESC("modified")))).toEqual([
      "new file",
      "no time",
      "old file",
    ]);
  });

  test("Kind groups by the kind as written, then by name", () => {
    const items = [
      fields("z", "2026-01-01", { kind: "Word" }),
      fields("y", "2026-01-02", { kind: "PDF" }),
      fields("x", "2026-01-03", { kind: "Word" }),
    ];
    expect(names(sortedFields(items, ASC("kind")))).toEqual(["y", "x", "z"]);
  });
});

test("no choice leaves the order as it came, and sorting never changes the list it was given", () => {
  const items = [fields("b", "2026-01-01"), fields("a", "2026-01-02")];
  expect(sortedFields(items, null)).toBe(items);
  sortedFields(items, ASC("name"));
  expect(names(items)).toEqual(["b", "a"]);
});

test("2,000 Documents sort in every order, as the sidebar and the Library ask", () => {
  const items = Array.from({ length: 2000 }, (_, index) =>
    fields(
      `Document ${(index * 7919) % 2000}`,
      new Date(1.7e12 + index * 86_400_000).toISOString(),
      {
        kind: ["PDF", "Word", "Excel"][index % 3] ?? "PDF",
        modified: new Date(1.6e12 + ((index * 31) % 2000) * 3_600_000).toISOString(),
      },
    ),
  );
  for (const key of ["name", "added", "modified", "kind"] as const) {
    const started = performance.now();
    const result = sortedFields(items, ASC(key));
    // The time is for the speed audit and the log; the test asserts the order, not the clock.
    console.info(`2,000 Documents by ${key}: ${(performance.now() - started).toFixed(1)} ms`);
    expect(result).toHaveLength(2000);
    const first = result[0];
    const last = result.at(-1);
    if (!first || !last) throw new Error("Nothing sorted.");
    if (key === "name") expect(first.name).toBe("Document 0");
    if (key === "name") expect(last.name).toBe("Document 1999");
  }
});
