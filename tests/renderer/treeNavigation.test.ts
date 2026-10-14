import { describe, expect, test } from "vitest";
import { startsWith } from "../../src/renderer/src/matcher";
import {
  firstChildOf,
  jumpTo,
  parentOf,
  type TreeRow,
  TYPE_AHEAD_MS,
  typedAfter,
} from "../../src/renderer/src/treeNavigation";

/*
 * The sidebar's tree by keyboard (docs/agents/interaction.md, rule 7): which
 * row ← steps out to, which row → steps into, and where a typed name jumps.
 */

const row = (name: string, level: number, expanded?: boolean): TreeRow => ({
  level,
  expanded,
  name,
});

// Finance (open) holds a Mind and two Documents; Research is closed; Not in a Folder is open.
const TREE: TreeRow[] = [
  row("Finance", 1, true),
  row("Shortlist", 2),
  row("Board minutes", 2),
  row("Budget 2026", 2),
  row("Research", 1, false),
  row("Not in a Folder", 1, true),
  row("Field notes", 2),
  row("Tide tables", 2),
];

describe("stepping out with ←", () => {
  test("goes to the nearest row a level up, however far above it is", () => {
    expect(parentOf(TREE, 1)).toBe(0);
    expect(parentOf(TREE, 3)).toBe(0);
    expect(parentOf(TREE, 7)).toBe(5);
  });

  test("a row at the top, or no row, has none to go to", () => {
    expect(parentOf(TREE, 0)).toBeNull();
    expect(parentOf(TREE, 4)).toBeNull();
    expect(parentOf(TREE, 40)).toBeNull();
    expect(parentOf([], 0)).toBeNull();
  });

  test("passes over rows that are as deep as this one", () => {
    const deep = [row("Linked", 1, true), row("2026", 2, true), row("Q1", 3), row("Q2", 3)];
    expect(parentOf(deep, 3)).toBe(1);
    expect(parentOf(deep, 1)).toBe(0);
  });
});

describe("stepping in with →", () => {
  test("an open row goes to the one below it, a level deeper", () => {
    expect(firstChildOf(TREE, 0)).toBe(1);
    expect(firstChildOf(TREE, 5)).toBe(6);
  });

  test("a closed row, a row with nothing to open and an open one with nothing in it go nowhere", () => {
    expect(firstChildOf(TREE, 4)).toBeNull();
    expect(firstChildOf(TREE, 1)).toBeNull();
    expect(firstChildOf([row("Empty", 1, true), row("Next", 1, false)], 0)).toBeNull();
    expect(firstChildOf([row("Last", 1, true)], 0)).toBeNull();
    expect(firstChildOf(TREE, 40)).toBeNull();
  });
});

describe("typing a name", () => {
  test("one letter goes to the next row that starts with it, whatever its case, and wraps round", () => {
    expect(jumpTo(TREE, 0, "b")).toBe(2);
    expect(jumpTo(TREE, 2, "B")).toBe(3);
    expect(jumpTo(TREE, 3, "b")).toBe(2);
    expect(jumpTo(TREE, 6, "f")).toBe(0);
  });

  test("several letters stay on a row that still matches, and go on to the next that does", () => {
    expect(jumpTo(TREE, 2, "bo")).toBe(2);
    expect(jumpTo(TREE, 3, "bo")).toBe(2);
    expect(jumpTo(TREE, 0, "bud")).toBe(3);
    expect(jumpTo(TREE, 0, "no")).toBe(5);
  });

  test("the same letter again goes round the rows that start with it", () => {
    const names = [row("Alpha", 1), row("Beta", 1), row("Apple", 1), row("Avocado", 1)];
    expect(jumpTo(names, 0, "a")).toBe(2);
    expect(jumpTo(names, 2, "aa")).toBe(3);
    expect(jumpTo(names, 3, "aaa")).toBe(0);
  });

  test("a name that starts with the same letter twice is found by typing it", () => {
    const names = [row("Alpha", 1), row("Aardvark", 1), row("Apple", 1)];
    expect(jumpTo(names, 0, "aa")).toBe(1);
    expect(jumpTo(names, 2, "aa")).toBe(1);
  });

  test("nothing starting with it, or nothing typed, goes nowhere", () => {
    expect(jumpTo(TREE, 0, "z")).toBeNull();
    expect(jumpTo(TREE, 0, "xyz")).toBeNull();
    expect(jumpTo(TREE, 0, "")).toBeNull();
    expect(jumpTo([], 0, "a")).toBeNull();
  });

  test("a name must start with it: letters inside a name don't count", () => {
    expect(jumpTo(TREE, 0, "m")).toBeNull();
    expect(jumpTo(TREE, 0, "2026")).toBeNull();
  });
});

describe("typing a Chinese name", () => {
  const NAMES = [row("财务", 1), row("研究", 1), row("合同审查", 1), row("Notes", 1)];

  test("goes to the row that starts with the characters typed", () => {
    expect(jumpTo(NAMES, 0, "研")).toBe(1);
    expect(jumpTo(NAMES, 0, "合同")).toBe(2);
    expect(jumpTo(NAMES, 2, "合同")).toBe(2);
    expect(jumpTo(NAMES, 2, "合")).toBe(2);
  });

  test("traditional characters find simplified names", () => {
    expect(jumpTo(NAMES, 1, "財")).toBe(0);
    expect(jumpTo(NAMES, 1, "財務")).toBe(0);
  });

  test("the pinyin, or its initials, of a Chinese name finds it, as in ⌘K", () => {
    expect(jumpTo(NAMES, 3, "cai")).toBe(0);
    expect(jumpTo(NAMES, 3, "caiw")).toBe(0);
    expect(jumpTo(NAMES, 0, "yan")).toBe(1);
    expect(jumpTo(NAMES, 0, "hts")).toBe(2);
    expect(startsWith("cw", "财务")).toBe(true);
    expect(startsWith("wu", "财务")).toBe(false);
  });

  test("a Chinese letter in a mixed name is read as it is written", () => {
    const mixed = [row("Q3 财务报告", 1), row("报告", 1)];
    expect(jumpTo(mixed, 0, "q3 财")).toBe(0);
    expect(jumpTo(mixed, 0, "报")).toBe(1);
  });
});

describe("how long the letters stay together", () => {
  test("keys within the pause join, a longer pause starts a new name", () => {
    expect(typedAfter("", "b", Number.POSITIVE_INFINITY)).toBe("b");
    expect(typedAfter("b", "o", 120)).toBe("bo");
    expect(typedAfter("bo", "a", TYPE_AHEAD_MS)).toBe("boa");
    expect(typedAfter("boa", "f", TYPE_AHEAD_MS + 1)).toBe("f");
  });
});
