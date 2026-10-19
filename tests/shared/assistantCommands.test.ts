import { describe, expect, test } from "vitest";
import { ASSISTANT_COMMANDS } from "../../src/shared/commands";

describe("The commands the assistant may run (#1110)", () => {
  // protects=the marked list and what each does, which decide whether a call asks: a command marked by mistake, or one whose effect changes, fails here first; fails_when=a command is added or changes kind without this test changing; no_claim=how each runs.
  test("are exactly these, each with what it does", () => {
    expect(ASSISTANT_COMMANDS).toEqual({
      "document.rename": { on: "document", does: "app", deletes: false },
      "document.moveTo": { on: "document", does: "app", deletes: false },
      "document.tags": { on: "document", does: "app", deletes: false },
      "document.delete": { on: "document", does: "app", deletes: true },
      "mind.rename": { on: "mind", does: "app", deletes: false },
      "mind.moveTo": { on: "mind", does: "app", deletes: false },
      "mind.delete": { on: "mind", does: "app", deletes: true },
      "mind.export": { on: "mind", does: "export", deletes: false },
      "documents.add": { on: "library", does: "add", deletes: false },
    });
  });

  // protects=nothing touching settings, consent, Privacy, approvals, keys, sign-ins, usage data or deleting for good is ever marked; fails_when=one is; no_claim=what the marked ones do.
  test("never touch settings, consent, Privacy, approvals, keys, sign-ins, usage data or deleting for good", () => {
    const denied =
      /^settings\.|consent|privacy|approval|secret|key|sign-?in|usage|permanent|deletePermanently|localMode/i;
    expect(Object.keys(ASSISTANT_COMMANDS).filter((id) => denied.test(id))).toEqual([]);
  });
});
