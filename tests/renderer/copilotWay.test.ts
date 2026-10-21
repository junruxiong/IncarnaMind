import { describe, expect, test } from "vitest";
import { copilotWay } from "../../src/core/providers/copilot/service";

describe("The way the Copilot plan is offered", () => {
  test("is none in a build with no way to sign in, so first run and Settings show no row", () => {
    expect(copilotWay([])).toBeNull();
  });

  test("is the experimental way while it is the only one", () => {
    expect(copilotWay([{ id: "editor", experimental: true }])).toEqual({
      id: "editor",
      experimental: true,
    });
  });

  test("is IncarnaMind's own app once it has a client ID, with the experimental way behind it", () => {
    const own = { id: "own", experimental: false } as const;
    const editor = { id: "editor", experimental: true } as const;
    expect(copilotWay([own, editor])).toEqual(own);
    expect(copilotWay([editor, own])).toEqual(own);
  });
});
