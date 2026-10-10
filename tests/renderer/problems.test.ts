import { describe, expect, test } from "vitest";
import type { ProviderErrorKind } from "../../src/core/api";
import { ANSWER_PROBLEMS, saveFailureReason } from "../../src/renderer/src/problems";
import { settingsPages } from "../../src/renderer/src/settingsPages";
import { type MessageKey, translate } from "../../src/shared/i18n";

const KINDS: ProviderErrorKind[] = [
  "auth",
  "model",
  "rate-limit",
  "network",
  "provider",
  "consent-declined",
  "not-signed-in",
  "plan-limit",
  "blocked",
  "too-long",
  "unknown",
];

describe("a failed Answer's problem line", () => {
  test("every kind of provider error has a line", () => {
    expect(Object.keys(ANSWER_PROBLEMS).sort()).toEqual([...KINDS].sort());
  });

  test("offers at most one action: a page of Settings, trying again, or nothing", () => {
    for (const kind of KINDS) {
      const { fix } = ANSWER_PROBLEMS[kind];
      if (fix.kind === "settings") expect(settingsPages).toContain(fix.page);
      expect(["settings", "retry", "none"]).toContain(fix.kind);
    }
  });

  test("what the User can fix in Settings opens the page that fixes it", () => {
    expect(ANSWER_PROBLEMS.auth.fix).toMatchObject({ kind: "settings", page: "models" });
    expect(ANSWER_PROBLEMS.model.fix).toMatchObject({ kind: "settings", page: "models" });
    // A declined data flow is allowed again on the Privacy page.
    expect(ANSWER_PROBLEMS["consent-declined"].fix).toMatchObject({
      kind: "settings",
      page: "privacy",
    });
  });

  test("a limit or a lost connection is tried again; a Question too long for the model is not", () => {
    for (const kind of ["rate-limit", "network", "provider", "unknown", "plan-limit"] as const) {
      expect(ANSWER_PROBLEMS[kind].fix.kind).toBe("retry");
    }
    expect(ANSWER_PROBLEMS["too-long"].fix.kind).toBe("none");
  });

  test("the line is one short sentence in both languages, naming the model where it can", () => {
    const labels = (language: "en" | "zh-CN") =>
      KINDS.map((kind) => {
        const { line, fix } = ANSWER_PROBLEMS[kind];
        const words = translate(language, line, { model: "claude-sonnet-5.5" });
        const label = fix.kind === "none" ? "" : translate(language, fix.label);
        return { kind, words, label };
      });
    for (const language of ["en", "zh-CN"] as const) {
      for (const { kind, words, label } of labels(language)) {
        expect(words, `${language} ${kind}`).not.toMatch(/\{/);
        // No full stop: the action follows on the same line.
        expect(words, `${language} ${kind}`).not.toMatch(/[.。]$/);
        if (ANSWER_PROBLEMS[kind].fix.kind !== "none") expect(label).not.toBe("");
      }
    }
    expect(translate("en", ANSWER_PROBLEMS.auth.line, { model: "claude-sonnet-5.5" })).toBe(
      "The key for claude-sonnet-5.5 was refused",
    );
    expect(translate("zh-CN", ANSWER_PROBLEMS.auth.line, { model: "claude-sonnet-5.5" })).toContain(
      "claude-sonnet-5.5",
    );
  });
});

describe("why a file couldn't be saved", () => {
  const key = (message: string): MessageKey | null => saveFailureReason(message);

  test("a folder that can't be written to is read-only", () => {
    expect(key("EACCES: permission denied, open '/Library/x.docx'")).toBe("export.failed.readOnly");
    expect(key("EROFS: read-only file system, open '/Volumes/x/a.docx'")).toBe(
      "export.failed.readOnly",
    );
    expect(key("EPERM: operation not permitted, open '/x/a.docx'")).toBe("export.failed.readOnly");
  });

  test("a full disk and a folder that's gone are told apart", () => {
    expect(key("ENOSPC: no space left on device, write")).toBe("export.failed.full");
    expect(key("ENOENT: no such file or directory, open '/gone/a.docx'")).toBe(
      "export.failed.gone",
    );
  });

  test("anything else is left to the system's own words", () => {
    expect(key("Something else happened.")).toBeNull();
  });

  test('the reasons read after "Couldn\'t save:" in both languages', () => {
    expect(
      translate("en", "export.failed.save", { reason: translate("en", "export.failed.readOnly") }),
    ).toBe("Couldn't save: the folder is read-only");
    expect(
      translate("zh-CN", "export.failed.save", {
        reason: translate("zh-CN", "export.failed.readOnly"),
      }),
    ).toBe("无法保存：这个文件夹是只读的");
  });
});
