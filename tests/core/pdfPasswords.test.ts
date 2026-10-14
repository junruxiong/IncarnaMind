/**
 * A password-protected PDF (#763): it fails as "password-protected" until the
 * User unlocks it with its password, which is kept only in the keychain, by
 * Document, and opens it for reading, the image pass and the viewer; a wrong
 * one keeps nothing. Forgetting it, or deleting the Document, lets it go, and
 * the password never shows in a logged error.
 */
import { afterEach, describe, expect, test, vi } from "vitest";
import type { Core, Document } from "../../src/core";
import { createMemoryKeychain, createTempDataFolder, startCore } from "../helpers/core";
import {
  addAndProcess,
  createSourceFolder,
  waitForProcessing,
  writeSourceFile,
} from "../helpers/documents";
import { writeLockedPdf } from "../helpers/lockedPdf";

const PASSWORD = "tide-tables-7Q!";
const LINES = ["Harbour accounts 2025", "The harbour office opens on Mondays at 8am."];

/** A core with a locked PDF and a Markdown file added, the PDF failed as locked. */
async function withLockedPdf(overrides: Parameters<typeof startCore>[1] = {}) {
  const core = startCore(await createTempDataFolder(), overrides);
  const sources = await createSourceFolder();
  const [locked, notes] = await addAndProcess(core, [
    await writeLockedPdf(sources, "Harbour accounts.pdf", LINES, PASSWORD),
    await writeSourceFile(sources, "Notes.md", "# Notes\n\nNothing locked here.\n"),
  ]);
  return { core, locked: locked as Document, notes: notes as Document };
}

const text = async (core: Core, id: string) =>
  (await core.readDocumentText(id)).pages.map((page) => page.text).join("\n");

afterEach(() => vi.restoreAllMocks());

describe("A password-protected PDF", () => {
  test("fails as locked; a wrong password keeps nothing; the right one is kept and the PDF is read with it", async () => {
    const { core, locked, notes } = await withLockedPdf();
    expect(locked).toMatchObject({
      status: "failed",
      failure: { reason: "password-protected" },
    });
    const changes: string[][] = [];
    core.on("documentPasswords.changed", (ids) => changes.push(ids));

    expect(await core.unlockDocument({ documentId: locked.id, password: "not-it" })).toEqual({
      kind: "wrong-password",
    });
    expect(await core.listDocumentPasswords()).toEqual([]);
    expect(await core.documentPasswordFor(locked.id)).toBeNull();

    const unlocked = await core.unlockDocument({ documentId: locked.id, password: PASSWORD });
    expect(unlocked.kind).toBe("unlocked");
    const [read] = await waitForProcessing(core, [locked.id]);
    expect(read).toMatchObject({ status: "ready", failure: null, pageCount: 1 });
    expect(await text(core, locked.id)).toContain("The harbour office opens on Mondays at 8am.");
    expect(await core.listDocumentPasswords()).toEqual([locked.id]);
    expect(changes).toEqual([[locked.id]]);
    // The viewer's one way to it: a locked PDF's password, and nothing for any other Document.
    expect(await core.documentPasswordFor(locked.id)).toBe(PASSWORD);
    expect(await core.documentPasswordFor(notes.id)).toBeNull();
  });

  test("forgetting the password lets it go and keeps the text read; deleting the Document lets it go too", async () => {
    const { core, locked } = await withLockedPdf();
    await core.unlockDocument({ documentId: locked.id, password: PASSWORD });
    await waitForProcessing(core, [locked.id]);

    await core.forgetDocumentPassword(locked.id);
    expect(await core.listDocumentPasswords()).toEqual([]);
    expect(await core.documentPasswordFor(locked.id)).toBeNull();
    expect(await text(core, locked.id)).toContain("Harbour accounts 2025");

    await core.unlockDocument({ documentId: locked.id, password: PASSWORD });
    expect(await core.listDocumentPasswords()).toEqual([locked.id]);
    await core.deleteDocument(locked.id);
    await vi.waitFor(async () => expect(await core.listDocumentPasswords()).toEqual([]));
  });

  test("only a PDF is unlocked, and a password is needed", async () => {
    const { core, locked, notes } = await withLockedPdf();
    await expect(core.unlockDocument({ documentId: notes.id, password: PASSWORD })).rejects.toThrow(
      "Only a PDF is opened with a password.",
    );
    await expect(core.unlockDocument({ documentId: locked.id, password: "" })).rejects.toThrow(
      "Type the PDF's password to open it.",
    );
  });

  test("the password never shows in a logged error, however unlocking or reading goes", async () => {
    const logged: unknown[] = [];
    for (const method of ["error", "warn", "log", "info", "debug"] as const) {
      vi.spyOn(console, method).mockImplementation((...args: unknown[]) => {
        logged.push(...args);
      });
    }
    // A keychain that can't keep secrets: unlocking fails, and says why without the password.
    const refusing = await withLockedPdf({ keychain: createMemoryKeychain("plain-text") });
    const refused = await refusing.core
      .unlockDocument({ documentId: refusing.locked.id, password: PASSWORD })
      .catch((error: unknown) => error);
    expect(refused).toBeInstanceOf(Error);
    logged.push(refused);

    const { core, locked } = await withLockedPdf();
    await core.unlockDocument({ documentId: locked.id, password: `${PASSWORD}-wrong` });
    await core.unlockDocument({ documentId: locked.id, password: PASSWORD });
    await waitForProcessing(core, [locked.id]);
    await core
      .unlockDocument({ documentId: "no-such-document", password: PASSWORD })
      .catch((error: unknown) => logged.push(error));

    const written = logged
      .map((each) =>
        each instanceof Error ? `${each.message}\n${each.stack ?? ""}` : JSON.stringify(each),
      )
      .join("\n");
    expect(written).not.toContain(PASSWORD);
  });
});
