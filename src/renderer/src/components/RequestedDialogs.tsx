import { useRequests } from "../requests";
import { DeleteMindDialog } from "./DeleteMindDialog";
import { DeleteDocumentDialog } from "./DocumentsSection";
import { DeleteFolderDialog } from "./LibraryFolders";
import { UnlinkFolderDialog } from "./UnlinkFolderDialog";

export function RequestedDialogs() {
  const { deleteMind, deleteDocument, deleteFolder, unlink, clear } = useRequests();
  return (
    <>
      <DeleteMindDialog mind={deleteMind} onClose={() => clear("deleteMind")} />
      <DeleteDocumentDialog target={deleteDocument} onClose={() => clear("deleteDocument")} />
      <DeleteFolderDialog folder={deleteFolder} onClose={() => clear("deleteFolder")} />
      <UnlinkFolderDialog target={unlink} onClose={() => clear("unlink")} />
    </>
  );
}
