import type { SharedDocument, SharePagePreview } from "../../core/api";

/** The size at which the sheet warns (#72): a file mail and chat apps may refuse. */
export const LARGE_BYTES = 20_000_000;

export const pictured = (document: SharedDocument, textOnly: ReadonlySet<string>) =>
  document.pictures > 0 && !textOnly.has(document.documentId);

export function sharedSize(preview: SharePagePreview, textOnly: ReadonlySet<string>) {
  let bytes = preview.textBytes;
  let pictures = 0;
  for (const document of preview.documents) {
    if (!pictured(document, textOnly)) continue;
    bytes += document.pictureBytes;
    pictures += document.pictures;
  }
  return { bytes, pictures };
}

/** Keeps as text the Documents whose pictures weigh most, until the file is under the warning. */
export function withFewerPictures(
  preview: SharePagePreview,
  textOnly: ReadonlySet<string>,
): Set<string> {
  const fewer = new Set(textOnly);
  const heaviest = preview.documents
    .filter((document) => pictured(document, textOnly))
    .sort((a, b) => b.pictureBytes - a.pictureBytes);
  for (const document of heaviest) {
    if (sharedSize(preview, fewer).bytes < LARGE_BYTES) break;
    fewer.add(document.documentId);
  }
  return fewer;
}
