import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { Core } from "../../src/core";
import { wordFileOf, wordImportProgressOf } from "../../src/shared/wordFiles";
import { connectToMind, type MindClient } from "./mindClient";

/**
 * Imports a Word file's bytes as a Mind, saved as `name` in `folder` and made in the Folder
 * `folderId`, and waits until every Block is in it. Returns the Mind's id and a client of it.
 */
export async function importWord(
  core: Core,
  folder: string,
  bytes: Buffer,
  {
    name = "Lease.docx",
    origin = "someone-else",
    folderId = null,
  }: { name?: string; origin?: "own" | "someone-else"; folderId?: string | null } = {},
): Promise<{ mindId: string; client: MindClient }> {
  const path = join(folder, name);
  await writeFile(path, bytes);
  const result = await core.importWordFile({ path, origin, folderId });
  if (!result.ok) throw new Error(`Refused: ${result.refusal}`);
  const client = await connectToMind(core, result.mind.id);
  await new Promise<void>((resolve) => {
    const check = () => {
      if (wordFileOf(client.doc) && wordImportProgressOf(client.doc) === null) {
        client.doc.off("update", check);
        resolve();
      }
    };
    client.doc.on("update", check);
    check();
  });
  return { mindId: result.mind.id, client };
}
