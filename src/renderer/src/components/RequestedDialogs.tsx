import { useRequests } from "../requests";
import { DeleteMindDialog } from "./DeleteMindDialog";
import { DeleteDocumentDialog } from "./DocumentsSection";
import { ExportDialog } from "./ExportDialog";
import { DeleteFolderDialog } from "./LibraryFolders";
import { UnlinkFolderDialog } from "./UnlinkFolderDialog";

/** The dialogs commands ask for (`requests.ts`), mounted once for every menu that can ask. */
export function RequestedDialogs() {
  const { deleteMind, deleteDocument, deleteFolder, unlink, exportMind, clear } = useRequests();
  return (
    <>
      <DeleteMindDialog mind={deleteMind} onClose={() => clear("deleteMind")} />
      <DeleteDocumentDialog target={deleteDocument} onClose={() => clear("deleteDocument")} />
      <DeleteFolderDialog folder={deleteFolder} onClose={() => clear("deleteFolder")} />
      <UnlinkFolderDialog target={unlink} onClose={() => clear("unlink")} />
      <ExportDialog mind={exportMind} onClose={() => clear("exportMind")} />
    </>
  );
}
