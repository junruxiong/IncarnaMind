import { describe, expect, test } from "vitest";
import type { AnswerToolCall, MindChange } from "../../src/core";
import { claimsIn, unbackedClaims } from "../../src/core/answers/claimedChanges";

/**
 * What a reply says it changed in a Mind, against what its Tool calls did
 * (#893): the words a reply reports a change with, in English and Chinese,
 * and the words that only look like one.
 */

const MINDS = ["Q3 review", "周报", "Reading group handout: paper summaries"];
const kinds = (reply: string) => claimsIn(reply, MINDS).map((claim) => claim.kind);

const mindCall = (tool: string, change: Partial<MindChange> | null): AnswerToolCall => ({
  id: `${tool}-1`,
  tool,
  source: "minds",
  input: {},
  status: change ? "done" : "failed",
  resultCount: null,
  ...(change && {
    mindChange: { mindId: "m1", created: false, request: "r1", proposals: 2, ...change },
  }),
});

describe("What a reply claims it changed", () => {
  // protects=the reports a reply makes of a change, as models write them, count as claims; fails_when=one of them reads as no claim; no_claim=every way a model may word a report.
  test.each([
    [
      'I made a new Mind, "Reading group handout: paper summaries", with a summary of both papers.',
      ["made"],
    ],
    ["I added a longer summary section to “Q3 review”.", ["added"]],
    ["Shortened the summary in “Q3 review” from 320 to 150 words.", ["changed"]],
    ["I've rewritten the risks section.", ["changed"]],
    ["I made the summary longer.", ["changed"]],
    ["The changes are waiting for your review.", ["changed"]],
    ["I replaced the old summary with the cited one.", ["removed"]],
    ["It has been added to your note.", ["added"]],
    ["我已在“周报”中添加了风险部分。", ["added"]],
    ["已新建“论文摘要”。", ["made"]],
    ["我把总结改写得更长了，修订等待你确认。", ["changed"]],
    ["我删除了重复的段落。", ["removed"]],
    ["我把风险部分删掉了。", ["removed"]],
  ])("%s", (reply, expected) => {
    expect(kinds(reply)).toEqual(expected);
  });

  // protects=answers about the Documents, refusals, and words about the reply itself aren't claims; fails_when=one of them reads as a claimed change, which would ask the model again for nothing; no_claim=every Answer a model may write.
  test.each([
    "The quarter was good, and sales were strong.",
    "The authors added a new layer and proposed a method in “Attention Is All You Need”.",
    "Proposed in 2017, the Transformer replaced recurrence.",
    "I couldn't change “Q3 review”: 12 changes are waiting there.",
    "Nothing was changed in “Q3 review”: 12 changes are waiting in it.",
    "I made a mistake: my first proposal went into the wrong note.",
    "I've written a summary below.",
    "I made no changes.",
    "I added citations to each claim.",
    "As I wrote before, the paper is a tutorial.",
    "作者已经修改了模型。",
    "我没有修改任何内容。",
    "下面是我写好的总结。",
  ])("%s", (reply) => {
    expect(kinds(reply)).toEqual([]);
  });

  // protects=a claim about an earlier Answer says so; fails_when=it reads as this Answer's; no_claim=none beyond protects.
  test("a claim about an earlier Answer says it is about one", () => {
    expect(claimsIn("I added the summary to “Q3 review” earlier.", MINDS)).toEqual([
      expect.objectContaining({ kind: "added", earlier: true }),
    ]);
  });
});

describe("Claims against what the calls did", () => {
  // protects=a claim no call backs is unbacked, and one a call backs isn't; fails_when=a made-up change passes, or a real one is flagged; no_claim=whether the change was good.
  test("a claimed change needs a call that made it", () => {
    const reply = "I added a risks section to “Q3 review”.";
    expect(unbackedClaims(reply, [], { minds: MINDS })).toEqual(["added"]);
    expect(unbackedClaims(reply, [mindCall("add_to_mind", {})], { minds: MINDS })).toEqual([]);
    expect(unbackedClaims(reply, [mindCall("add_to_mind", null)], { minds: MINDS })).toEqual([
      "added",
    ]);
  });

  // protects=a made Mind needs a call that made one; "replaced" needs the old text gone, not only new text added; fails_when=either passes on an addition alone; no_claim=none beyond protects.
  test("made needs a Mind made, and replaced needs text changed, not only added", () => {
    const added = [mindCall("add_to_mind", {})];
    expect(unbackedClaims("I made a new note, “Plan”.", added)).toEqual(["made"]);
    expect(
      unbackedClaims("I made a new note, “Plan”.", [mindCall("create_mind", { created: true })]),
    ).toEqual([]);
    expect(unbackedClaims("I replaced the old summary.", added)).toEqual(["removed"]);
    expect(
      unbackedClaims("I replaced the old summary.", [mindCall("edit_mind", { revised: true })]),
    ).toEqual([]);
  });

  // protects=a claim about an earlier Answer holds by that Answer's calls; fails_when=it is flagged though the earlier Answer made the change; no_claim=none beyond protects.
  test("a claim about an earlier Answer holds by its calls", () => {
    const reply = "I added the summary to “Q3 review” earlier.";
    expect(unbackedClaims(reply, [], { minds: MINDS })).toEqual(["added"]);
    expect(
      unbackedClaims(reply, [], { minds: MINDS, earlierCalls: [mindCall("add_to_mind", {})] }),
    ).toEqual([]);
  });
});
