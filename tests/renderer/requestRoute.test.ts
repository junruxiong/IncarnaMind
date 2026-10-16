import { describe, expect, test } from "vitest";
import { routeRequest } from "../../src/renderer/src/requestRoute";

describe("What the composer's words ask for, with text selected (#880)", () => {
  test("a change goes to the text; anything else to the assistant, and a request that only sounds like one changes nothing at once", () => {
    const kinds = (words: string) => routeRequest(words).kind;
    expect(kinds("make the second paragraph more formal")).toBe("edit");
    expect(kinds("replace coast with shore")).toBe("edit");
    expect(kinds("把第二段改得更正式")).toBe("edit");
    expect(kinds("make sense of this for me")).toBe("answer");
    expect(kinds("What does this clause mean?")).toBe("answer");
    expect(kinds("make a list of the risks")).toBe("write");
    expect(routeRequest("turn this into a new note")).toEqual({ kind: "write", newMind: true });
    expect(kinds("can you make this shorter?")).toBe("unclear");
    expect(kinds("能把这段改短一点吗？")).toBe("unclear");
  });
});
