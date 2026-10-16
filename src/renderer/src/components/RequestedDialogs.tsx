import { useRequests } from "../requests";
import { DeletePermanentlyDialog } from "./DeletePermanentlyDialog";
import { LocalModelDialog } from "./providers/LocalModelDialog";
import { UnlinkFolderDialog } from "./UnlinkFolderDialog";

export function RequestedDialogs() {
  const { unlink, deletePermanently, chooseLocalModel, clear } = useRequests();
  return (
    <>
      <DeletePermanentlyDialog
        request={deletePermanently}
        onClose={() => clear("deletePermanently")}
      />
      <UnlinkFolderDialog target={unlink} onClose={() => clear("unlink")} />
      <LocalModelDialog
        open={chooseLocalModel !== null}
        onClose={() => clear("chooseLocalModel")}
      />
    </>
  );
}
