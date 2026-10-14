import { describe, expect, test } from "vitest";
import { newNoteTitle } from "../../src/renderer/src/newNoteRequest";

/*
 * A request for an empty Mind by name (#202) is made at once, with no AI
 * step; anything that asks for more goes to the assistant.
 */

describe("A request for an empty Mind", () => {
  // protects=the ways people ask for an empty note by name, in English and Chinese, give its title; fails_when=one of them goes to the model, or the title keeps its quotes; no_claim=every phrasing there is.
  test.each([
    ["Create a new note called Fieldwork plan", "Fieldwork plan"],
    ["create a note named “Q4 budget”.", "Q4 budget"],
    ["Make a new Mind titled Reading list", "Reading list"],
    ["Start a new page called Ideas", "Ideas"],
    ['Create a new note called "Plan and budget"', "Plan and budget"],
    ["新建一篇叫“田野计划”的笔记", "田野计划"],
    ["新建一篇笔记，叫田野计划", "田野计划"],
    ["帮我新建一个名为“周报”的 Mind", "周报"],
    ["创建一个名为周报的笔记", "周报"],
  ])("%s", (request, title) => {
    expect(newNoteTitle(request)).toBe(title);
  });

  // protects=a request that asks for content too goes to the assistant; fails_when=it makes an empty Mind and drops what was asked; no_claim=none beyond protects.
  test.each([
    "Create a new note called Plan with three steps for visiting customers",
    "Make a new note called Summary about these papers",
    "Summarise this into a new Mind",
    "把这些文档总结成一篇新笔记",
    "新建一篇叫田野计划的笔记，包含三个步骤",
    "Create a new note called Fieldwork plan\nwith the steps below",
    "",
  ])("%s goes to the assistant", (request) => {
    expect(newNoteTitle(request)).toBeNull();
  });
});
