import { describe, expect, test } from "vitest";
import type { ProviderErrorKind } from "../../src/core/api";
import { ANSWER_PROBLEMS, answerProblem, saveFailureReason } from "../../src/renderer/src/problems";
import { settingsPages } from "../../src/renderer/src/settingsPages";
import { type MessageKey, translate } from "../../src/shared/i18n";

const KINDS: ProviderErrorKind[] = [
  "auth",
  "model",
  "rate-limit",
  "network",
  "provider",
  "consent-declined",
  "local-mode",
  "not-signed-in",
  "plan-limit",
  "blocked",
  "too-long",
  "cost-cap",
  "unknown",
];

describe("a failed Answer's problem line", () => {
  test("every kind of provider error has a line", () => {
    expect(Object.keys(ANSWER_PROBLEMS).sort()).toEqual([...KINDS].sort());
  });

  test("offers at most one action: a page of Settings, trying again, a provider's page, signing in, or nothing", () => {
    for (const kind of KINDS) {
      const { fix } = ANSWER_PROBLEMS[kind];
      if (fix.kind === "settings") expect(settingsPages).toContain(fix.page);
      if (fix.kind === "link") expect(fix.url).toMatch(/^https:\/\//);
      expect(["settings", "retry", "link", "sign-in", "none"]).toContain(fix.kind);
    }
  });

  test("an ended ChatGPT sign-in offers to sign in again, right from the Answer", () => {
    expect(ANSWER_PROBLEMS["not-signed-in"].fix).toEqual({
      kind: "sign-in",
      label: "answer.error.signIn",
    });
  });

  test("a reached ChatGPT plan limit offers Manage usage, ChatGPT's own page for it", () => {
    expect(ANSWER_PROBLEMS["plan-limit"].fix).toEqual({
      kind: "link",
      url: "https://chatgpt.com/settings/usage",
      label: "answer.error.manageUsage",
    });
  });

  test("the monthly cap says so, and Change cap opens General, where the cap is", () => {
    expect(ANSWER_PROBLEMS["cost-cap"]).toEqual({
      line: "answer.error.cost-cap",
      fix: { kind: "settings", page: "general", label: "answer.error.changeCap" },
    });
    expect(translate("en", "answer.error.cost-cap")).toBe("Monthly cap reached");
    expect(translate("en", "answer.error.changeCap")).toBe("Change cap");
  });

  test("an Ollama Cloud allowance used up is managed on Ollama's own page", () => {
    expect(answerProblem("plan-limit", "ollama")).toEqual({
      line: "answer.error.plan-limit.ollama",
      fix: { kind: "link", url: "https://ollama.com/settings", label: "answer.error.manageUsage" },
    });
    expect(answerProblem("plan-limit", "chatgpt")).toBe(ANSWER_PROBLEMS["plan-limit"]);
    expect(answerProblem("plan-limit", undefined)).toBe(ANSWER_PROBLEMS["plan-limit"]);
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
    for (const kind of ["rate-limit", "network", "provider", "unknown"] as const) {
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
