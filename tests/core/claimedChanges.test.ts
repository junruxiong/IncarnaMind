import { describe, expect, test } from "vitest";
import type { AnswerToolCall, MindChange } from "../../src/core";
import { claimsIn, shownUpTo, unbackedClaims } from "../../src/core/answers/claimedChanges";

/**
 * The narrow net under the change log: a reply's plainest reports of a change
 * to a Mind, in English and Chinese, against what its Tool calls did, Mind
 * by Mind. It misses a vague sentence rather than flag an honest one.
 */

const MINDS = [
  { id: "q3", name: "Q3 review" },
  { id: "weekly", name: "周报" },
  { id: "tides", name: "Tide notes" },
];
const NAMES = MINDS.map((mind) => mind.name);
const kinds = (reply: string) => claimsIn(reply, NAMES).map((claim) => claim.kind);

const mindCall = (tool: string, change: Partial<MindChange> | null): AnswerToolCall => ({
  id: `${tool}-1`,
  tool,
  source: "minds",
  input: {},
  status: change ? "done" : "failed",
  resultCount: null,
  ...(change && {
    mindChange: { mindId: "q3", created: false, request: "r1", proposals: 2, ...change },
  }),
});

describe("What a reply claims it changed", () => {
  // protects=the plainest reports of a change read as claims; fails_when=one reads as none; no_claim=every wording a model may use.
  test.each([
    ["I made a new Mind, “Tide notes”, with a summary of both papers.", ["made"]],
    ["I created “Tide notes”.", ["made"]],
    ["Summary page created in “Tide notes”.", ["made"]],
    ["I added a longer summary section to “Q3 review”.", ["added"]],
    ["Added a risks section below the introduction in “Q3 review”.", ["added"]],
    ["- Added a risks section to “Q3 review”", ["added"]],
    ["Shortened the summary in “Q3 review” from 320 to 150 words.", ["changed"]],
    ["I made the summary longer.", ["changed"]],
    ["The summary is now twice as long.", ["changed"]],
    ["The changes are waiting for your review.", ["changed"]],
    ["The new page replaces the earlier version.", ["removed"]],
    ["I replaced the old summary with the cited one.", ["removed"]],
    ["I added the summary to “Q3 review” earlier.", ["added"]],
    ["我已在“周报”中添加了风险部分。", ["added"]],
    ["总结已扩写到原来的两倍。", ["changed"]],
    ["页面已创建。", ["made"]],
    ["我把风险部分删掉了。", ["removed"]],
  ])("%s", (reply, expected) => {
    expect(kinds(reply)).toEqual(expected);
  });

  // protects=answers about the Documents, refusals, Citations and words about the reply aren't claims; fails_when=one is flagged; no_claim=every Answer a model may write.
  test.each([
    "The quarter was good, and sales were strong.",
    "The authors added a new layer and proposed a method in “Attention Is All You Need”.",
    "Proposed in 2017, the Transformer replaced recurrence.",
    "I couldn't change “Q3 review”: 12 changes are waiting there.",
    "Nothing was changed in “Q3 review”: 12 changes are waiting in it.",
    "I made a mistake: my first proposal went into the wrong note.",
    "I've written a summary below.",
    "I made no changes.",
    "I've added 11 citations to the page.",
    "As I wrote before, the paper is a tutorial.",
    'The earlier reply was wrong to say I added a risks section to "Q3 review".',
    'The earlier claim that I added a risks section to "Q3 review" was wrong.',
    "之前说我已在“周报”中添加了风险部分，这是错的。",
    "Expanded two-paper summary (longer draft)\n1) Overview",
    "作者已经修改了模型。",
    "我没有修改任何内容。",
    "下面是我写好的总结。",
  ])("%s", (reply) => {
    expect(kinds(reply)).toEqual([]);
  });
});

describe("Claims against what the calls did", () => {
  // protects=a claimed change needs a call that made it; fails_when=a made-up change passes, or a real one is flagged; no_claim=whether the change was good.
  test("a claimed change needs a call that made it", () => {
    const reply = "I added a risks section to “Q3 review”.";
    expect(unbackedClaims(reply, [], MINDS)).toEqual(["added"]);
    expect(unbackedClaims(reply, [mindCall("add_to_mind", {})], MINDS)).toEqual([]);
    expect(unbackedClaims(reply, [mindCall("add_to_mind", null)], MINDS)).toEqual(["added"]);
  });

  // protects=a claim that names a Mind holds only by a change in that Mind; fails_when=a change elsewhere backs it; no_claim=none beyond protects.
  test("a claim naming a Mind needs a change in that Mind", () => {
    const elsewhere = [mindCall("add_to_mind", { mindId: "tides" })];
    expect(unbackedClaims("I added a risks section to “Q3 review”.", elsewhere, MINDS)).toEqual([
      "added",
    ]);
    expect(unbackedClaims("I added a risks section to “Tide notes”.", elsewhere, MINDS)).toEqual(
      [],
    );
  });

  // protects=a made Mind needs a call that made one; "replaced" needs text changed, not only added; a claim about an earlier Answer gets no pass; fails_when=any of these passes; no_claim=none beyond protects.
  test("made, replaced and earlier", () => {
    const added = [mindCall("add_to_mind", {})];
    expect(unbackedClaims("I made a new note, “Plan”.", added)).toEqual(["made"]);
    expect(
      unbackedClaims("I made a new note, “Plan”.", [mindCall("create_mind", { created: true })]),
    ).toEqual([]);
    expect(unbackedClaims("I replaced the old summary.", added)).toEqual(["removed"]);
    expect(
      unbackedClaims("I replaced the old summary.", [mindCall("edit_mind", { revised: true })]),
    ).toEqual([]);
    expect(unbackedClaims("I added the summary to “Q3 review” earlier.", [], MINDS)).toEqual([
      "added",
    ]);
  });
});

describe("A reply as it streams", () => {
  const shown = (text: string) => text.slice(0, shownUpTo(text, ["Q3 review"]));

  // protects=a reply streams as it comes but for the sentence that may claim a change, held until it is checked, and its last word, which may become one; fails_when=a claim shows before it is checked, or text that can't be one waits; no_claim=none beyond protects.
  test("shows all but what may claim a change", () => {
    expect(shown("A Mind is ")).toBe("A Mind is ");
    expect(shown("A Mind is a note")).toBe("A Mind is a ");
    expect(shown("The tides are late. I've added a risks section to “Q3 review”. More")).toBe(
      "The tides are late. ",
    );
    expect(shown("The tides are late. I've ad")).toBe("The tides are late. ");
    expect(shown("The tides are late. Short")).toBe("The tides are late. ");
    expect(shown("The summary is now twice")).toBe("");
    expect(shown("总结说潮汐很晚。总结已扩")).toBe("总结说潮汐很晚。");
    expect(shown("潮汐很晚")).toBe("潮汐很晚");
  });
});
