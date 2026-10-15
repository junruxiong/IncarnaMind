import { describe, expect, test } from "vitest";
import { partialFields } from "../../src/core/tools/partialInput";

/*
 * A Tool call's input read while the model writes it (#202): the text of each
 * string field so far, whatever the place the stream has reached.
 */

const INPUT = JSON.stringify({
  mind: "Project “plan”",
  after: "## Risks",
  count: { nested: [1, "two}"] },
  content: 'Line one.\n\nLine "two" \\ ✓ é \u0007',
});

describe("A Tool call's input, as the model writes it", () => {
  // protects=every prefix of the input reads as its fields so far, the last one cut where the stream is; fails_when=a field reads wrong text, an escape cut in two shows half of it, or a non-string value is taken for one; no_claim=inputs that aren't objects.
  test("every prefix gives the fields written so far, the last one unfinished", () => {
    const whole = JSON.parse(INPUT) as Record<string, unknown>;
    for (let end = 0; end <= INPUT.length; end++) {
      const fields = partialFields(INPUT.slice(0, end));
      for (const [key, field] of fields) {
        const full = whole[key];
        expect(typeof full).toBe("string");
        expect((full as string).startsWith(field.value)).toBe(true);
        if (field.done) expect(field.value).toBe(full);
      }
    }
    const done = partialFields(INPUT);
    expect(Object.fromEntries([...done].map(([key, field]) => [key, field.value]))).toEqual({
      mind: "Project “plan”",
      after: "## Risks",
      content: 'Line one.\n\nLine "two" \\ ✓ é \u0007',
    });
    expect([...done.values()].every((field) => field.done)).toBe(true);
  });

  test("text that isn't an object has no fields", () => {
    expect(partialFields("").size).toBe(0);
    expect(partialFields("[1, 2]").size).toBe(0);
    expect(partialFields('{"mind"').size).toBe(0);
  });
});
