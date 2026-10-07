import type { ReactNode } from "react";
import { useT } from "../i18n";
import type { LoadedFile } from "./useDocumentFile";
import { ViewerHeader } from "./ViewerHeader";
import { FileGone, ViewerMessage } from "./ViewerMessage";

/**
 * A view while its file isn't ready to show: loading, gone (the quote is
 * still shown), or failed. `ready` draws it once it is.
 */
export function FileStates<T>({
  loaded,
  quote,
  ready,
}: {
  loaded: LoadedFile<T>;
  quote: string | undefined;
  ready(value: T): ReactNode;
}) {
  const t = useT();
  switch (loaded.kind) {
    case "loading":
      return (
        <>
          <ViewerHeader />
          <ViewerMessage>{t("viewer.loading")}</ViewerMessage>
        </>
      );
    case "missing":
      return (
        <>
          <ViewerHeader openable={false} />
          <FileGone quote={quote} />
        </>
      );
    case "failed":
      return (
        <>
          <ViewerHeader />
          <ViewerMessage tone="error">
            {loaded.message ? t("viewer.failed", { message: loaded.message }) : t("viewer.notText")}
          </ViewerMessage>
        </>
      );
    case "ready":
      return <>{ready(loaded.value)}</>;
  }
}
