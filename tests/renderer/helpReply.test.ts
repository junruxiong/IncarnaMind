/**
 * A help button's Answer during a call, read as the floating panel lays it out: options
 * with the recommended first and what each leads to, readings with the moment they rest on,
 * and short lines, while it streams too.
 */
import { describe, expect, test } from "vitest";
import { quoted, readHelpReply } from "../../src/renderer/src/helpReply";

const textOf = (pieces: ReturnType<typeof readHelpReply>[number]["lead"]) =>
  pieces.map((piece) => (piece.kind === "text" ? piece.text : `[${piece.marker}]`)).join("");

describe("A help reply", () => {
  test("options: the recommended first, each with what it leads to, its Citation kept apart", () => {
    const items = readHelpReply(
      "★ We can't go below list, but we can extend the pilot to 60 days. [^1]\n→ keeps the deal moving\n\nLet me check with the team.\n→ buys time",
    );
    expect(items).toHaveLength(2);
    expect(items[0]?.recommended).toBe(true);
    expect(textOf(items[0]?.lead ?? [])).toBe(
      "We can't go below list, but we can extend the pilot to 60 days. [1]",
    );
    expect(textOf(items[0]?.effect ?? [])).toBe("keeps the deal moving");
    expect(items[1]?.recommended).toBe(false);
    expect(textOf(items[1]?.effect ?? [])).toBe("buys time");
  });

  test("an option shows in quotation marks, the model's own taken off, its Citation after them", () => {
    const [item] = readHelpReply('★ "Yes, Enterprise is $45 a seat" [^1]\n→ answers at once');
    expect(textOf(quoted(item?.lead ?? []))).toBe("“Yes, Enterprise is $45 a seat”[1]");
    const [chinese] = readHelpReply("★ 是的，企业版每席位每月 $45。[^2]\n→ 直接回答");
    expect(textOf(quoted(chinese?.lead ?? []))).toBe("“是的，企业版每席位每月 $45。”[2]");
  });

  test("readings: the moment of the transcript is a note between the reading and its move", () => {
    const items = readHelpReply(
      "★ They want the total for 40 seats under £10k\n12:04 “our budget stops at ten thousand” [^1]\n→ give the tiered price\n\nThey may be comparing you with another vendor\n12:06 “the other quote we have” [^2]\n→ ask what matters most",
    );
    expect(items).toHaveLength(2);
    expect(items[0]?.notes.map(textOf)).toEqual(["12:04 “our budget stops at ten thousand” [1]"]);
    expect(items[1]?.notes.map(textOf)).toEqual(["12:06 “the other quote we have” [2]"]);
  });

  test("lines: each bullet is one, with no lead-in or Citation definitions shown", () => {
    const items = readHelpReply(
      "**Summary**\n- The launch moves to 18 November [^1]\n- The preview stays on 28 October\n\n[^1]: Transcript",
    );
    expect(items.map((item) => textOf(item.lead))).toEqual([
      "Summary",
      "The launch moves to 18 November [1]",
      "The preview stays on 28 October",
    ]);
  });

  test("only the first star is a recommendation", () => {
    const items = readHelpReply("★ One\n→ a\n\n★ Two\n→ b");
    expect(items.map((item) => item.recommended)).toEqual([true, false]);
  });

  test("while it streams, an option before its arrow has no effect yet", () => {
    const items = readHelpReply("★ We can extend the pil");
    expect(items).toHaveLength(1);
    expect(items[0]?.effect).toBeNull();
  });
});
