import { existsSync } from "node:fs";
import { mkdir, readdir, symlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { ApprovalRequest, ConversationAnswer, Core } from "../../src/core";
import { openDatabase } from "../../src/core/storage";
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
    const made = call.results.filter((each) =>
      /^(?:list_commands|run_command|search_documents)$/.test(each.tool),
    ).length;
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
      input: { command: "documents.addFromFolder", folder: inbox, kinds: ["md"] },
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

describe("What the assistant's commands ask the User first (#1110)", {
  timeout: 90_000,
}, () => {
  const deny = (core: Core) => {
    const asked: ApprovalRequest[] = [];
    core.on("approval.requested", (request) => {
      asked.push(request);
      void core.respondToApproval(request.requestId, "deny");
    });
    return asked;
  };

  // protects=export into a folder inside the Output location, which a sync service may share, asks as anywhere else does; only the Output location itself runs at once; fails_when=a subfolder of it is written without asking; no_claim=the export itself.
  test("export into a folder inside the Output location asks", async () => {
    const output = await createTempDataFolder();
    const shared = join(output, "Shared with client");
    await mkdir(shared);
    const model = calling(
      find("Tides", "mind"),
      run("mind.export", { format: "markdown", folder: shared }),
    );
    const { core, mind } = await setUpWithDocuments(model, DOCUMENTS);
    await core.setOutputLocation(output);
    const asked = deny(core);

    await askAssistant(core, { text: "Export this Mind to the shared folder", mindId: mind.id });

    expect(asked.map((request) => "ask" in request && request.ask.kind)).toEqual(["export"]);
    expect(await readdir(shared)).toEqual([]);
  });

  // protects=once the Answer has read Passages, a delete asks, naming how many, and Don't allow deletes nothing (an injected instruction can't empty the Library); fails_when=it runs at once; no_claim=renames and moves of a few, which still run with Undo.
  test("a delete after reading Passages asks, and Don't allow deletes nothing", async () => {
    const model = calling(
      () => ({ tool: "search_documents", input: { query: "invoice" } }),
      find("Northwind"),
      run("document.delete"),
    );
    const { core, mind } = await setUpWithDocuments(model, DOCUMENTS);
    const asked = deny(core);
    const before = await names(core);

    const answer = await askAssistant(core, { text: "What did Northwind bill?", mindId: mind.id });

    expect(asked).toEqual([
      expect.objectContaining({
        ask: expect.objectContaining({
          kind: "change",
          command: "document.delete",
          reason: "afterReading",
          count: 2,
        }),
      }),
    ]);
    expect(commandCalls(answer).at(-1)?.approval).toBe("denied");
    expect(await names(core)).toEqual(before);
  });

  // protects=one call that would change more than 20 things asks first, whatever it is, with the count; fails_when=a mass rename, move or tag runs at once; no_claim=calls of 20 or fewer.
  test("a call that changes more than 20 things asks first", async () => {
    const many = Array.from({ length: 21 }, (_, index) => ({
      name: `Receipt ${index + 1}.md`,
      contents: `# Receipt ${index + 1}\n\nPaid.\n`,
    }));
    const model = calling(find("Receipt"), run("document.tags", { tag: "Paid" }));
    const { core, mind } = await setUpWithDocuments(model, many);
    const asked = deny(core);

    await askAssistant(core, { text: "Tag every receipt Paid", mindId: mind.id });

    expect(asked).toEqual([
      expect.objectContaining({
        ask: expect.objectContaining({ kind: "change", reason: "many", count: 21, into: "Paid" }),
      }),
    ]);
    expect((await core.listTags()).map((tag) => tag.name)).not.toContain("Paid");
  });

  // protects=deleting the Mind the User is in always asks; fails_when=it is sent to Recently deleted at once; no_claim=other Minds.
  test("deleting the Mind the User is in asks", async () => {
    const model = calling(find("Tides", "mind"), run("mind.delete"));
    const { core, mind } = await setUpWithDocuments(model, DOCUMENTS);
    const asked = deny(core);

    await askAssistant(core, { text: "Delete this Mind", mindId: mind.id });

    expect(asked).toEqual([
      expect.objectContaining({
        ask: expect.objectContaining({ reason: "focusedMind", count: 1 }),
      }),
    ]);
    expect((await core.listMinds()).map((each) => each.id)).toContain(mind.id);
  });

  // protects=a stored "always" for a command, synced from elsewhere, never lets one run without asking; fails_when=an export elsewhere runs at once; no_claim=Connectors' and Skills' policies.
  test("a stored always policy for a command is never applied", async () => {
    const elsewhere = await createTempDataFolder();
    const model = calling(
      find("Tides", "mind"),
      run("mind.export", { format: "markdown", folder: elsewhere }),
    );
    const { core, mind, dataDir } = await setUpWithDocuments(model, DOCUMENTS);
    const db = openDatabase(join(dataDir, "incarnamind.db"));
    const at = new Date().toISOString();
    db.run(
      `INSERT INTO approval_policies (id, subject_kind, subject_id, policy, created_at, updated_at)
       VALUES (?, 'command', 'mind.export', 'always', ?, ?)`,
      [crypto.randomUUID(), at, at],
    );
    db.close();
    const asked = deny(core);

    await askAssistant(core, { text: "Export this Mind there", mindId: mind.id });

    expect(asked).toHaveLength(1);
    expect(await readdir(elsewhere)).toEqual([]);
  });
});

describe("Undo of the assistant's commands (#1110)", { timeout: 90_000 }, () => {
  // protects=an Undo that can't take back one change takes back the rest, says which failed, and tried again once the cause is fixed takes back what is left; Redo then makes all of it again; fails_when=the Run is stuck half-undone, or Redo does nothing; no_claim=the window's line.
  test("a partial Undo is tried again, then Redo makes it all again", async () => {
    const model = calling(
      find("Lease"),
      (results) => ({
        tool: "run_command",
        input: { command: "document.rename", ids: idsFound(results), name: "Renamed lease" },
      }),
      find("invoice"),
      run("document.tags", { tag: "Finance" }),
    );
    const { core, mind } = await setUpWithDocuments(model, DOCUMENTS);
    const answer = await askAssistant(core, {
      text: "Rename the lease and tag the invoices",
      mindId: mind.id,
    });
    const lease = (await core.listDocuments()).find((each) => each.name === "Renamed lease");
    if (!lease) throw new Error("Not renamed.");
    await core.deleteDocument(lease.id);

    await expect(core.undoCommands(runOf(answer))).rejects.toMatchObject({
      wording: { key: "error.commands.undoPartly" },
    });
    expect((await core.listTags()).map((tag) => tag.name)).not.toContain("Finance");

    await core.restore([{ kind: "document", id: lease.id }]);
    await core.undoCommands(runOf(answer));
    expect(await names(core)).toContain("Lease Harbour Street");

    await core.redoCommands(runOf(answer));
    expect(await names(core)).toContain("Renamed lease");
    const finance = (await core.listTags()).find((tag) => tag.name === "Finance");
    expect(
      (await core.listDocuments()).filter((each) =>
        each.tags.some((one) => one.tagId === finance?.id),
      ),
    ).toHaveLength(2);
  });

  // protects=Undo leaves in place a Folder the Run made once the User filed their own things in it, and says so; fails_when=the User's Document is unfiled; no_claim=a Folder the User made.
  test("Undo keeps a Folder it made when the User filed something in it since", async () => {
    const model = calling(find("Lease"), run("document.moveTo", { folder: "Real estate" }));
    const { core, mind } = await setUpWithDocuments(model, DOCUMENTS);
    const answer = await askAssistant(core, {
      text: "Move the leases into Real estate",
      mindId: mind.id,
    });
    const folder = (await core.getLibrary()).groups.find((group) => group.name === "Real estate");
    const invoice = (await core.listDocuments()).find((each) => each.name.startsWith("Northwind"));
    if (!folder || !invoice) throw new Error("No Folder or invoice.");
    await core.moveToFolder({ documentIds: [invoice.id], folderId: folder.id });

    expect(await core.undoCommands(runOf(answer))).toEqual([
      { kind: "folder", name: "Real estate" },
    ]);

    const library = await core.getLibrary();
    expect(library.groups.map((group) => group.name)).toContain("Real estate");
    expect(library.assignments.find((each) => each.documentId === invoice.id)?.groupId).toBe(
      folder.id,
    );
    const lease = (await core.listDocuments()).find((each) => each.name.startsWith("Lease"));
    expect(
      library.assignments.find((each) => each.documentId === lease?.id)?.groupId ?? null,
    ).toBeNull();
  });

  // protects=Undo leaves in place a Tag the Run made once the User tagged other Documents with it, and takes it off only the Run's; fails_when=the User's own tagging is taken off; no_claim=automatic tagging.
  test("Undo keeps a Tag it made when the User tagged something else with it since", async () => {
    const model = calling(find("invoice"), run("document.tags", { tag: "Finance" }));
    const { core, mind } = await setUpWithDocuments(model, DOCUMENTS);
    const answer = await askAssistant(core, { text: "Tag every invoice Finance", mindId: mind.id });
    const finance = (await core.listTags()).find((tag) => tag.name === "Finance");
    const lease = (await core.listDocuments()).find((each) => each.name.startsWith("Lease"));
    if (!finance || !lease) throw new Error("No Tag or lease.");
    await core.addTagToDocuments([lease.id], finance.id);

    expect(await core.undoCommands(runOf(answer))).toEqual([{ kind: "tag", name: "Finance" }]);

    const tagged = (await core.listDocuments())
      .filter((each) => each.tags.some((one) => one.tagId === finance.id))
      .map((each) => each.name);
    expect(tagged).toEqual(["Lease Harbour Street"]);
  });

  // protects=a tag command that fails part way leaves no empty Tag behind; fails_when=the new Tag stays with nothing tagged; no_claim=a Tag that was there already.
  test("a tag command that fails leaves no new Tag behind", async () => {
    const model = scriptedModel((call) => {
      const listed = call.results.find((each) => each.tool === "list_commands");
      if (!listed) return { calls: [find("Draft")([])] };
      if (call.results.some((each) => each.tool === "run_command")) return { text: "Done." };
      const ids = idsFound(call.results);
      // Deleted first, then tagged: the Document is gone when the Tag is put on it.
      return {
        calls: [
          { tool: "run_command", input: { command: "document.delete", ids } },
          { tool: "run_command", input: { command: "document.tags", ids, tag: "Fresh" } },
        ],
      };
    });
    const { core, mind } = await setUpWithDocuments(model, DOCUMENTS);

    const answer = await askAssistant(core, { text: "Delete and tag the draft", mindId: mind.id });

    expect(commandCalls(answer).map((call) => call.status)).toEqual(["done", "done", "failed"]);
    expect((await core.listTags()).map((tag) => tag.name)).not.toContain("Fresh");
  });
});
