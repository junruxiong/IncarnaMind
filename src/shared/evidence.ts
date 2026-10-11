/**
 * Whether checks leave a Document out of their evidence now (#80), as the
 * core's search decides it (src/core/documents/evidence.ts), for the window
 * to show: IncarnaMind wrote it, the User hasn't counted it, and neither
 * Settings nor its Folder counts such files. Copies of a document being
 * checked are left out too, but only for that check, which says so itself.
 */
import type { Document, LibrarySnapshot, Settings } from "../core/api";

export function leftOutOfEvidence(
  document: Pick<Document, "id" | "ownFile">,
  settings: Pick<Settings, "user"> | null,
  library: Pick<LibrarySnapshot, "groups" | "assignments"> | null,
): boolean {
  if (!document.ownFile || document.ownFile.counted) return false;
  if (settings && !settings.user.leaveOutOwnFiles) return false;
  const folderId = library?.assignments.find((each) => each.documentId === document.id)?.groupId;
  const folder = folderId ? library?.groups.find((group) => group.id === folderId) : undefined;
  return !folder?.countsOwnFiles;
}
