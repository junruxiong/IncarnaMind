import { describe, expect, test } from "vitest";
import { type ChangeCounts, changeFact } from "../../src/renderer/src/changeFacts";
import { type MessageKey, type MessageParams, translate } from "../../src/shared/i18n";

const t =
  (language: "en" | "zh-CN") =>
  (key: MessageKey, params?: MessageParams): string =>
    translate(language, key, params);

const row = (counts: Partial<ChangeCounts>): ChangeCounts => ({
  created: false,
  changed: 0,
  wrote: 0,
  tables: 0,
  ...counts,
});

describe("What an Answer did in a Mind, in words (#880), counted by paragraph (#1028)", () => {
  test.each([
    [
      row({ changed: 12 }),
      "Changed 12 paragraphs in “Q3”",
      "Changed 12 paragraphs",
      "修改了 12 个段落",
    ],
    [row({ changed: 1 }), "Changed 1 paragraph in “Q3”", "Changed 1 paragraph", "修改了 1 个段落"],
    [row({ wrote: 1 }), "Wrote 1 paragraph in “Q3”", "Wrote 1 paragraph", "写入了 1 个段落"],
    [
      row({ created: true, wrote: 3 }),
      "Created “Q3” and wrote 3 paragraphs",
      "Wrote 3 paragraphs",
      "写入了 3 个段落",
    ],
    [
      row({ changed: 4, tables: 1 }),
      "Changed 4 paragraphs and added a table in “Q3”",
      "Changed 4 paragraphs and added a table",
      "修改了 4 个段落，添加了 1 个表格",
    ],
    [row({ tables: 2 }), "Added 2 tables to “Q3”", "Added 2 tables", "添加了 2 个表格"],
  ] as const)(
    "on the card and on the Mind's line, the same counts: %o",
    (counts, card, line, zhLine) => {
      expect(changeFact(t("en"), counts, "Q3")).toBe(card);
      expect(changeFact(t("en"), counts, null)).toBe(line);
      expect(changeFact(t("zh-CN"), counts, null)).toBe(zhLine);
    },
  );
});
