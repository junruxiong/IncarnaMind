import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { InvalidInputError, NotFoundError } from "../../src/core/errors";
import { en } from "../../src/shared/i18n/en";
import { zhCN } from "../../src/shared/i18n/zh-CN";
import { createTempDataFolder, startCore } from "../helpers/core";

/*
 * The core's errors in the User's language (#627). An error the window shows carries a wording, a
 * key of the dictionary with its values, beside its English sentence, which the log and a notice's
 * Details keep. Every wording reads in Chinese, and a new error with only an English sentence
 * fails here, so none arrives untranslated.
 */

const CORE = join(import.meta.dirname, "../../src/core");

/**
 * The errors still thrown with only an English sentence, by file under src/core: mostly checks of
 * what the window sends, which a User never meets. A new error gets a wording instead
 * (src/core/errors.ts); when one gets a wording, lower its file's count.
 */
const SENTENCES_LEFT: Record<string, number> = {
  "agents/index.ts": 12,
  "agents/setup.ts": 7,
  "aiUse.ts": 7,
  "answers/index.ts": 6,
  "approvals.ts": 12,
  "citationStyles/request.ts": 6,
  "citedPages.ts": 1,
  "comparisons/index.ts": 12,
  "connectors/config.ts": 12,
  "connectors/index.ts": 4,
  "consent.ts": 7,
  "conversations/index.ts": 11,
  "core.ts": 17,
  "documents/evidence.ts": 2,
  "documents/index.ts": 20,
  "documents/library.ts": 7,
  "documents/retention.ts": 8,
  "documents/web/index.ts": 5,
  "edits/index.ts": 12,
  "embedding/active.ts": 3,
  "exports/index.ts": 3,
  "folders.ts": 1,
  "layout.ts": 24,
  "library/index.ts": 24,
  "meetings/index.ts": 20,
  "mindContent.ts": 2,
  "minds.ts": 6,
  "noteAdding.ts": 5,
  "outputFiles.ts": 6,
  "privacy.ts": 6,
  "providers/chat.ts": 13,
  "providers/embeddings.ts": 8,
  "providers/kinds.ts": 2,
  "providers/rerank.ts": 7,
  "recentlyDeleted.ts": 13,
  "referenceRecords/import/index.ts": 6,
  "referenceRecords/index.ts": 9,
  "runs/tasks.ts": 5,
  "scope.ts": 2,
  "settings.ts": 5,
  "skills/draft.ts": 3,
  "skills/index.ts": 10,
  "sources/index.ts": 11,
  "sources/tool.ts": 2,
  "suggestions/index.ts": 6,
  "tags/index.ts": 11,
  "tags/jev.ts": 6,
  "tags/tagger.ts": 1,
  "tools/web/tools.ts": 1,
  "usageData.ts": 4,
  "wordImports.ts": 3,
};

const SENTENCE_ONLY = /new (?:InvalidInputError|NotFoundError)\(\s*(?!\{)/g;
const WORDING_KEY = /new \w+Error\(\{\s*key: "([^"]+)"/g;

const coreSources = () =>
  readdirSync(CORE, { recursive: true, encoding: "utf8" })
    .filter((name) => name.endsWith(".ts"))
    .map((name) => ({
      file: name.split("\\").join("/"),
      text: readFileSync(join(CORE, name), "utf8"),
    }));

const placeholders = (text: string) => [...text.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();

describe("an error the window shows", () => {
  test("keeps its English sentence beside its wording", () => {
    const error = new InvalidInputError({
      key: "error.document.nameTooLong",
      values: { max: 500 },
    });
    expect(error.message).toBe("A Document's name can't be longer than 500 characters.");
    expect(error.wording).toEqual({ key: "error.document.nameTooLong", values: { max: 500 } });
    expect(new NotFoundError("It was deleted.").wording).toBeNull();
  });

  test("says its sentence in English whatever the interface language, and its wording for the window", async () => {
    const core = startCore(await createTempDataFolder());
    await core.updateSettings({ user: { language: "zh-CN" } });
    const renaming = core.renameDocument("no-such-id", "Tides");
    await expect(renaming).rejects.toThrow(NotFoundError);
    await expect(renaming).rejects.toMatchObject({
      message: "There is no such Document.",
      wording: { key: "error.document.notFound" },
    });
  });

  test("has Chinese words for every wording, with the values the English names", () => {
    const english = new Map(Object.entries(en));
    const untranslated = Object.entries(zhCN)
      .filter(([key]) => key.startsWith("error."))
      .filter(
        ([key, chinese]) =>
          !/\p{Script=Han}/u.test(chinese) ||
          placeholders(chinese).join() !== placeholders(english.get(key) ?? "").join(),
      )
      .map(([key]) => key);
    expect(untranslated).toEqual([]);
  });

  test("is worded from the error keys of the dictionary", () => {
    const strays = coreSources().flatMap(({ file, text }) =>
      [...text.matchAll(WORDING_KEY)]
        .map((m) => m[1] ?? "")
        .filter((key) => !key.startsWith("error.") || !Object.hasOwn(en, key))
        .map((key) => `${file}: ${key}`),
    );
    expect(strays).toEqual([]);
  });

  test("comes with a wording: no new error is thrown with only an English sentence", () => {
    const counted = Object.fromEntries(
      coreSources()
        .map(({ file, text }) => [file, text.match(SENTENCE_ONLY)?.length ?? 0] as const)
        .filter(([, count]) => count > 0)
        .sort(([a], [b]) => a.localeCompare(b)),
    );
    expect(
      counted,
      "Give the new error a wording, { key, values }, with its key in src/shared/i18n/en.ts and zh-CN.ts; or, when one got a wording, lower its file's count in SENTENCES_LEFT.",
    ).toEqual(SENTENCES_LEFT);
  });
});
