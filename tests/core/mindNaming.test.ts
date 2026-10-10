import { describe, expect, test } from "vitest";
import { createTempDataFolder, startCore } from "../helpers/core";
import { appendParagraph, connectToMind, typeInto } from "../helpers/mindClient";

describe("a Mind without a title", () => {
  test("is named by its first line of text, which follows the edits", async () => {
    const core = startCore(await createTempDataFolder());
    const mind = await core.createMind();
    const client = await connectToMind(core, mind.id);

    expect((await core.listMinds())[0]?.autoTitle).toBe("");

    appendParagraph(client, "Tides and the moon");
    appendParagraph(client, "A second paragraph");
    await client.settled();
    expect((await core.listMinds())[0]?.autoTitle).toBe("Tides and the moon");

    typeInto(client, 0, "Tides".length, " in Bergen");
    await client.settled();
    expect((await core.listMinds())[0]?.autoTitle).toBe("Tides in Bergen and the moon");
  });

  test("is named from a long Chinese first line, cut at a character", async () => {
    const core = startCore(await createTempDataFolder());
    const mind = await core.createMind();
    const client = await connectToMind(core, mind.id);

    appendParagraph(client, "夜间降温".repeat(30));
    await client.settled();

    const title = (await core.listMinds())[0]?.autoTitle ?? "";
    expect(title).toBe("夜间降温".repeat(15));
  });

  test("is told to the app as it is typed", async () => {
    const core = startCore(await createTempDataFolder());
    const mind = await core.createMind();
    const client = await connectToMind(core, mind.id);
    const names: string[][] = [];
    core.on("minds.changed", (minds) => names.push(minds.map((each) => each.autoTitle)));

    appendParagraph(client, "Hello");
    await client.settled();

    expect(names.at(-1)).toEqual(["Hello"]);
  });

  test("a title the User typed is never replaced", async () => {
    const core = startCore(await createTempDataFolder());
    const mind = await core.createMind();
    const client = await connectToMind(core, mind.id);
    appendParagraph(client, "First line");
    await client.settled();

    await core.renameMind(mind.id, "My title");
    typeInto(client, 0, 0, "The ");
    await client.settled();

    const [listed] = await core.listMinds();
    expect(listed?.title).toBe("My title");
    expect(listed?.autoTitle).toBe("The First line");
  });

  test("an empty Mind has no name of its own", async () => {
    const core = startCore(await createTempDataFolder());
    const mind = await core.createMind();
    const client = await connectToMind(core, mind.id);
    appendParagraph(client, "   ");
    await client.settled();

    expect((await core.listMinds())[0]?.autoTitle).toBe("");
  });
});
