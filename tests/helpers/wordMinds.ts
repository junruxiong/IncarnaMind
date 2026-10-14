import type { Core, ImportWordFileInput } from "../../src/core";
import { wordFileOf, wordImportProgressOf } from "../../src/shared/wordFiles";
import { connectToMind, type MindClient } from "./mindClient";

/** Imports a Word file as a Mind and waits until every Block is in it; returns a client of the Mind. */
export async function importWordMind(
  core: Core,
  input: ImportWordFileInput,
): Promise<{ mindId: string; client: MindClient }> {
  const result = await core.importWordFile(input);
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
