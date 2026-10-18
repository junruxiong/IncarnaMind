import { existsSync } from "node:fs";
import { mkdir, readdir, symlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { ApprovalRequest, ConversationAnswer, Core } from "../../src/core";
import { askAssistant } from "../helpers/assistant";
import { askAndFinish, setUpWithDocuments } from "../helpers/citations";
import { createTempDataFolder, nextEvent } from "../helpers/core";
import { type ModelCall, scriptedModel } from "../helpers/models";

/**
 * The assistant does what the app can (#1110), through the core's public interface as the
 * window asks: it finds what to act on with list_commands and runs commands on it with
 * run_command. A change inside IncarnaMind runs at once and one Undo takes back the Run; a file
 * saved outside the Output location and files added from a folder wait for the User. The model's
 * choices are scripted; real models' are tried by hand.
 */

const DOCUMENTS = [
  { name: "Northwind invoice March.md", contents: "# March\n\nNorthwind billed 1,200 euros.\n" },
  { name: "Northwind invoice April.md", contents: "# April\n\nNorthwind billed 1,400 euros.\n" },
  { name: "Lease Harbour Street.md", contents: "# Harbour Street\n\nThe rent is 900 euros.\n" },
  { name: "Draft plan.md", contents: "# Plan\n\nA first draft.\n" },
];

type Step = (results: ModelCall["results"]) => { tool: string; input: unknown };

/** A model that makes these calls, one a step, each from what the earlier ones returned, then replies. */
const calling = (...steps: Step[]) =>
  scriptedModel((call) => {
    const made = call.results.filter((each) => /^(?:list|run)_commands?$/.test(each.tool)).length;
    const step = steps[made];
    return step ? { calls: [step(call.results)] } : { text: "Done." };
  });

/** The ids list_commands last found. */
const idsFound = (results: ModelCall["results"]): string[] => {
  const listed = results.filter((each) => each.tool === "list_commands").at(-1)?.text ?? "";
  return [...listed.matchAll(/^- (\S+): /gm)].map((match) => match[1] ?? "");
};

const find =
  (name: string, kind = "document"): Step =>
  () => ({
    tool: "list_commands",
    input: { kind, name },
  });

const run =
  (command: string, input: Record<string, unknown> = {}): Step =>
  (results) => ({
    tool: "run_command",
    input: { command, ids: idsFound(results), ...input },
  });

const commandCalls = (answer: ConversationAnswer) =>
  answer.toolCalls.filter((call) => call.source === "commands");

/** The Run an Answer's commands were part of: what one Undo takes back. */
function runOf(answer: ConversationAnswer): string {
  const run = commandCalls(answer).find((call) => call.command)?.command?.run;
  if (!run) throw new Error("No command ran.");
  return run;
}

const names = async (core: Core) => (await core.listDocuments()).map((each) => each.name).sort();

describe("Commands inside IncarnaMind run at once, and one Undo takes back the Run (#1110)", {
  timeout: 60_000,
}, () => {
  // protects="rename the Northwind files by date" finds exactly those Documents and renames each, its date before its name, and one Undo puts both names back, Redo both new ones; fails_when=another Document is renamed, a rename waits, or Undo takes back only one; no_claim=that a real model dates them so.
  test("renamed by date, then back with one Undo, and again with Redo", async () => {
    const renameEach: Step = (results) => {
      const listed = results.find((each) => each.tool === "list_commands")?.text ?? "";
      const done = results.filter((each) => each.tool === "run_command").length;
      const [, id = "", name = "", added = ""] =
        [...listed.matchAll(/^- (\S+): “(.+?)” \(added (\S+?);/gm)][done] ?? [];
      return {
        tool: "run_command",
        input: { command: "document.rename", ids: [id], name: `${added} ${name}` },
      };
    };
    const model = calling(find("Northwind"), renameEach, renameEach);
    const { core, mind } = await setUpWithDocuments(model, DOCUMENTS);
    const before = await names(core);
    const added = (await core.listDocuments())[0]?.createdAt.slice(0, 10);

    const answer = await askAssistant(core, {
      text: "Rename the Northwind files by date",
      mindId: mind.id,
    });

    expect(await names(core)).toEqual(
      [
        "Draft plan",
        "Lease Harbour Street",
        `${added} Northwind invoice April`,
        `${added} Northwind invoice March`,
      ].sort(),
    );
    const [found, ...renames] = commandCalls(answer);
    expect(found?.found).toEqual({ kind: "document", count: 2, by: { name: "Northwind" } });
    expect(renames.map((call) => call.status)).toEqual(["done", "done"]);
    expect(renames.map((call) => call.approval)).toEqual([undefined, undefined]);

    await core.undoCommands(runOf(answer));
    expect(await names(core)).toEqual(before);
    await core.redoCommands(runOf(answer));
    expect((await names(core)).filter((name) => name.startsWith(`${added} `))).toHaveLength(2);
  });

  // protects=moving into a Folder that isn't there makes it and moves only what was found, and Undo moves them back and takes the Folder away; fails_when=the Folder stays after Undo, or a Document is left in it; no_claim=Folders the User had.
  test("moved into a Folder it made, and Undo takes both back", async () => {
    const model = calling(find("Lease"), run("document.moveTo", { folder: "Real estate" }));
    const { core, mind } = await setUpWithDocuments(model, DOCUMENTS);

    const answer = await askAssistant(core, {
      text: "Move the leases into Real estate",
      mindId: mind.id,
    });

    const library = await core.getLibrary();
    const folder = library.groups.find((each) => each.name === "Real estate");
    expect(folder).toBeDefined();
    const lease = (await core.listDocuments()).find((each) => each.name.startsWith("Lease"));
    expect(library.assignments.filter((each) => each.groupId === folder?.id)).toEqual([
      expect.objectContaining({ documentId: lease?.id }),
    ]);
    expect(commandCalls(answer).at(-1)?.command).toMatchObject({
      command: "document.moveTo",
      items: [{ kind: "document", id: lease?.id }],
      into: { kind: "folder", name: "Real estate", made: true },
    });

    await core.undoCommands(runOf(answer));
    const after = await core.getLibrary();
    expect(after.groups.map((each) => each.name)).not.toContain("Real estate");
    expect(after.assignments.filter((each) => each.documentId === lease?.id)).toEqual([]);
  });

  // protects=tagging with a Tag that isn't there makes it, tags only what was found, and Undo takes the Tag off and away; fails_when=the Tag stays, or another Document gets it; no_claim=automatic tagging.
  test("tagged with a new Tag, and Undo takes it away", async () => {
    const model = calling(find("invoice"), run("document.tags", { tag: "Finance" }));
    const { core, mind } = await setUpWithDocuments(model, DOCUMENTS);

    const answer = await askAssistant(core, { text: "Tag every invoice Finance", mindId: mind.id });

    const tag = (await core.listTags()).find((each) => each.name === "Finance");
    const tagged = (await core.listDocuments())
      .filter((each) => each.tags.some((one) => one.tagId === tag?.id))
      .map((each) => each.name)
      .sort();
    expect(tagged).toEqual(["Northwind invoice April", "Northwind invoice March"]);

    await core.undoCommands(runOf(answer));
    expect((await core.listTags()).map((each) => each.name)).not.toContain("Finance");
  });

  // protects=two commands called at once that would each make the same Tag both work, one Tag made; fails_when=the second fails because the first made the Tag meanwhile; no_claim=which call makes it.
  test("two calls at once that tag with the same new Tag both work", async () => {
    const model = scriptedModel((call) => {
      const listed = call.results.find((each) => each.tool === "list_commands");
      if (!listed) return { calls: [find("invoice")([])] };
      if (call.results.some((each) => each.tool === "run_command")) return { text: "Done." };
      return {
        calls: idsFound(call.results).map((id) => ({
          tool: "run_command",
          input: { command: "document.tags", ids: [id], tag: "Finance" },
        })),
      };
    });
    const { core, mind } = await setUpWithDocuments(model, DOCUMENTS);

    const answer = await askAssistant(core, { text: "Tag every invoice Finance", mindId: mind.id });

    expect(commandCalls(answer).map((call) => call.status)).toEqual(["done", "done", "done"]);
    const tags = (await core.listTags()).filter((each) => each.name === "Finance");
    expect(tags).toHaveLength(1);
    const tagged = (await core.listDocuments()).filter((each) =>
      each.tags.some((one) => one.tagId === tags[0]?.id),
    );
    expect(tagged).toHaveLength(2);
  });

  // protects=delete sends what was found to Recently deleted, as the User's own delete does, never deletes for good, and Undo brings it back; fails_when=it is deleted for good, or Undo can't restore it; no_claim=Recently deleted's own expiry.
  test("deleted to Recently deleted, and Undo brings it back", async () => {
    const model = calling(find("Draft"), run("document.delete"));
    const { core, mind } = await setUpWithDocuments(model, DOCUMENTS);

    const answer = await askAssistant(core, { text: "Delete the drafts", mindId: mind.id });

    expect(await names(core)).not.toContain("Draft plan");
    expect((await core.listDeleted()).map((each) => each.name)).toContain("Draft plan");
    await core.undoCommands(runOf(answer));
    expect(await names(core)).toContain("Draft plan");
  });

  // protects=a command that fails is reported as failed, from what it returned, and changes nothing, and one the assistant may not run (Privacy, keys, deleting for good) is refused; fails_when=the card says done, or something changed; no_claim=the reply's words.
  test("one that fails, or isn't the assistant's to run, is failed on its card and changes nothing", async () => {
    const model = calling(
      () => ({ tool: "run_command", input: { command: "document.delete", ids: ["no-such-id"] } }),
      () => ({ tool: "run_command", input: { command: "deleted.deletePermanently", ids: [] } }),
    );
    const { core, mind } = await setUpWithDocuments(model, DOCUMENTS);
    const before = await names(core);

    const answer = await askAssistant(core, { text: "Delete it", mindId: mind.id });

    expect(
      commandCalls(answer).map((call) => [call.status, call.commandProblem?.wording?.key]),
    ).toEqual([
      ["failed", "error.commands.noDocument"],
      // Not one of the Tool's commands: refused before anything runs.
      ["failed", undefined],
    ]);
    expect(await names(core)).toEqual(before);
    expect((await core.listDeleted()).length).toBe(0);
  });

  // protects=the commands' Tools are offered only to an Answer in the assistant; fails_when=a Question asked in a Mind can run commands; no_claim=outside agents, which are offered no write at all.
  test("are offered in the assistant, and not to a Question in a Mind", async () => {
    const tools: string[][] = [];
    const model = scriptedModel((call) => {
      tools.push(call.tools);
      return { text: "Done." };
    });
    const { core, mind, client } = await setUpWithDocuments(model, DOCUMENTS);

    await askAssistant(core, { text: "Rename the Northwind files by date", mindId: mind.id });
    expect(tools.at(-1)).toEqual(expect.arrayContaining(["list_commands", "run_command"]));
    await askAndFinish(core, client, mind.id, "Rename the Northwind files by date");
    expect(tools.at(-1)).not.toContain("run_command");
  });
});

describe("Commands that reach outside IncarnaMind wait for the User (#1110)", {
  timeout: 60_000,
}, () => {
  const exportMind = (folder?: string) =>
    calling(
      find("Tides", "mind"),
      run("mind.export", { format: "markdown", ...(folder && { folder }) }),
    );

  // protects=export into the Output location runs at once; fails_when=it asks, or nothing is saved there; no_claim=the file's contents, which the export tests check.
  test("export into the Output location runs at once", async () => {
    const { core, mind } = await setUpWithDocuments(exportMind(), DOCUMENTS);
    const output = await createTempDataFolder();
    await core.setOutputLocation(output);
    const asked: ApprovalRequest[] = [];
    core.on("approval.requested", (request) => asked.push(request));

    const answer = await askAssistant(core, {
      text: "Export this Mind as Markdown",
      mindId: mind.id,
    });

    expect(asked).toEqual([]);
    expect(await readdir(output)).toEqual(["Tides.md"]);
    expect(commandCalls(answer).at(-1)?.command?.file).toMatchObject({ name: "Tides.md" });
  });

  // protects=export to another folder waits for Allow, saying where and as what, with no "always"; Don't allow saves nothing and the model is told; fails_when=it saves before Allow, offers "always", or saves after Don't allow; no_claim=the card's look.
  test.each(["allow-once", "deny"] as const)(
    "export elsewhere waits, and %s decides",
    async (decision) => {
      const elsewhere = await createTempDataFolder();
      const { core, mind } = await setUpWithDocuments(exportMind(elsewhere), DOCUMENTS);
      const requested = nextEvent(core, "approval.requested");
      const asking = askAssistant(core, {
        text: "Export this Mind to that folder",
        mindId: mind.id,
      });

      const request = await requested;
      expect(request).toMatchObject({
        subject: { kind: "command", command: "mind.export" },
        tool: "run_command",
        ask: { kind: "export", mind: { id: mind.id }, format: "markdown", fileName: "Tides.md" },
      });
      expect(await readdir(elsewhere)).toEqual([]);
      await expect(core.respondToApproval(request.requestId, "always-allow")).rejects.toThrow();
      await core.respondToApproval(request.requestId, decision);
      const answer = await asking;

      expect(await readdir(elsewhere)).toEqual(decision === "deny" ? [] : ["Tides.md"]);
      expect(commandCalls(answer).at(-1)?.approval).toBe(
        decision === "deny" ? "denied" : "allowed",
      );
    },
  );

  // protects=adding files from a folder asks first, listing the files of the kinds asked for that are right in it and nothing a link leads to outside it, then adds them, and Undo takes them out; fails_when=a file outside the folder is listed or read, or files are added before Allow; no_claim=how each file is processed.
  test("adding a folder's files asks first, then adds only those", async () => {
    const inbox = await createTempDataFolder();
    const outside = await createTempDataFolder();
    await writeFile(join(inbox, "a.md"), "# A\n\nFirst.\n");
    await writeFile(join(inbox, "b.md"), "# B\n\nSecond.\n");
    await writeFile(join(inbox, "c.csv"), "a,b\n1,2\n");
    await writeFile(join(outside, "secret.md"), "# Secret\n");
    await symlink(join(outside, "secret.md"), join(inbox, "link.md"));
    await mkdir(join(inbox, "sub"));
    await writeFile(join(inbox, "sub", "d.md"), "# D\n");
    const model = calling(() => ({
      tool: "run_command",
      input: { command: "documents.add", folder: inbox, kinds: ["md"] },
    }));
    const { core, mind } = await setUpWithDocuments(model, []);
    const requested = nextEvent(core, "approval.requested");
    const asking = askAssistant(core, { text: "Add the notes in that folder", mindId: mind.id });

    const request = await requested;
    expect(request).toMatchObject({ ask: { kind: "add", files: ["a.md", "b.md"] } });
    expect(await core.listDocuments()).toEqual([]);
    await core.respondToApproval(request.requestId, "allow-once");
    const answer = await asking;

    expect(await names(core)).toEqual(["a", "b"]);
    await core.undoCommands(runOf(answer));
    expect(await core.listDocuments()).toEqual([]);
    expect(existsSync(join(inbox, "a.md"))).toBe(true);
  });
});
