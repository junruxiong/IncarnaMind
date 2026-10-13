import { vi } from "vitest";
import type { CitationAttributes, Core, DrawnPage } from "../../src/core";

/** The smallest PNG there is: one pixel. */
export const PNG = Uint8Array.from(
  Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==",
    "base64",
  ),
);

/** Waits until the Citation's page is saved, and gives it. */
export async function savedPage(core: Core, citation: CitationAttributes): Promise<DrawnPage> {
  return vi.waitFor(
    async () => {
      const page = await core.citedPageImage(citation);
      if (!page) throw new Error("The cited page isn't saved yet.");
      return page;
    },
    { timeout: 20_000, interval: 50 },
  );
}
