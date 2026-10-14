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
  "ollama-not-running",
  "not-signed-in",
  "plan-limit",
  "blocked",
  "too-long",
  "cost-cap",
  "no-answer",
  "unknown",
];

describe("a failed Answer's problem line", () => {
  test("every kind of provider error has a line", () => {
    expect(Object.keys(ANSWER_PROBLEMS).sort()).toEqual([...KINDS].sort());
  });

  test("offers at most one action: a page of Settings, trying again, a provider's page, signing in, a model on this computer, Ollama, or nothing", () => {
    for (const kind of KINDS) {
      const { fix } = ANSWER_PROBLEMS[kind];
      if (fix.kind === "settings") expect(settingsPages).toContain(fix.page);
      if (fix.kind === "link") expect(fix.url).toMatch(/^https:\/\//);
      expect(["settings", "retry", "link", "sign-in", "local-model", "ollama", "none"]).toContain(
        fix.kind,
      );
    }
  });

  test("local mode stopping an Answer offers a local model in place, and never a company's settings", () => {
    expect(ANSWER_PROBLEMS["local-mode"].fix).toEqual({
      kind: "local-model",
      label: "answer.error.useLocal",
    });
  });

  test("Ollama not running says so in both languages, and its fix is Ollama's own (open it or get it)", () => {
    expect(ANSWER_PROBLEMS["ollama-not-running"]).toEqual({
      line: "answer.error.ollama-not-running",
      fix: { kind: "ollama", label: "localModel.open" },
    });
    expect(translate("en", "answer.error.ollama-not-running")).toBe(
      "Ollama isn't running on this computer",
    );
    expect(translate("zh-CN", "answer.error.ollama-not-running")).toBe(
      "Ollama 没有在这台电脑上运行",
    );
    expect(translate("en", "localModel.open")).toBe("Open Ollama");
    expect(translate("en", "providers.ollama.get")).toBe("Get Ollama");
    expect(translate("zh-CN", "localModel.open")).toBe("打开 Ollama");
    expect(translate("zh-CN", "providers.ollama.get")).toBe("下载 Ollama");
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

  test("a reply with no text says so, and offers Try again, in both languages", () => {
    expect(ANSWER_PROBLEMS["no-answer"]).toEqual({
      line: "answer.error.no-answer",
      fix: { kind: "retry", label: "answer.error.retry" },
    });
    expect(translate("en", "answer.error.no-answer")).toBe("The model sent no Answer");
    expect(translate("zh-CN", "answer.error.no-answer")).toBe("模型没有给出回答");
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
