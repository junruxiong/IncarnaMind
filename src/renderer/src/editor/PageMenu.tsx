import type { Editor } from "@tiptap/core";
import { useRef, useState } from "react";
import type * as Y from "yjs";
import { mindName } from "../../../shared/naming";
import type { Verification } from "../../../shared/verification";
import { MoveToPicker } from "../components/MoveToPicker";
import { ObjectMenu } from "../components/ObjectMenu";
import { useTabPlace } from "../components/tabs/tabItems";
import { menuItemClass, menuItemUnavailableClass } from "../components/ui";
import { files } from "../core";
import { showFileLabelKey } from "../fileInfo";
import { useFirstOpen } from "../firstOpen";
import { useLanguage, useT } from "../i18n";
import { fileStatusLabel } from "../linkedFolders";
import { useAppStore } from "../store";
import { openDocumentBeside } from "../verify";
import { MindCounts } from "./WordCount";
import { openOriginalAtCaret, useWordFile, wordFileTooltip } from "./WordOrigin";

/**
 * The ⋯ button at a Mind's page's top right (#908; the board
 * Page-Clean-Menu): the Mind's menu in the sidebar row's order; "From
 * {file}" with Show in Finder for a Mind imported from Word or made by
 * Verify, which opens the original beside it; and at its foot the counts and
 * the switch that shows the count while writing.
 */
export function PageMenu({
  editor,
  doc,
  mindId,
  verification,
  readOnly,
}: {
  editor: Editor;
  doc: Y.Doc;
  mindId: string;
  verification: Verification | null;
  readOnly: boolean;
}) {
  const t = useT();
  const language = useLanguage();
  const mind = useAppStore((state) => state.minds.find((each) => each.id === mindId));
  const name = (mind && mindName(mind)) || t("mind.untitled");
  const tabId = useTabPlace()?.tabId;
  // The quiet first open shows no tab strip, so there is no tab to rename in.
  const firstOpen = useFirstOpen();
  const holder = useRef<HTMLSpanElement>(null);
  const [moving, setMoving] = useState<HTMLElement | null>(null);
  const button = () => holder.current?.querySelector<HTMLElement>("[data-object-menu]") ?? null;
  const { file, report } = useWordFile(doc);
  const origin = verification
    ? { name: verification.documentName, documentId: verification.documentId, word: false }
    : file
      ? { name: file.name, documentId: file.documentId ?? null, word: true }
      : null;
  const original = useAppStore((state) =>
    origin?.documentId ? state.documents.find((each) => each.id === origin.documentId) : undefined,
  );
  const away = original ? fileStatusLabel(original.fileStatus) : null;
  return (
    <span ref={holder} className="contents">
      <ObjectMenu
        target={{ kind: "mind", id: mindId }}
        label={t("mind.more", { title: name })}
        testId="page"
        align="end"
        className="page-menu-button"
        here={{
          // Once the menu has closed and given the focus back, so the field or the picker keeps it.
          ...(tabId &&
            !firstOpen && {
              "mind.rename": () =>
                requestAnimationFrame(() => useAppStore.getState().renameTab(tabId)),
            }),
          "mind.moveTo": () => requestAnimationFrame(() => setMoving(button())),
        }}
        more={
          origin
            ? (close) => (
                <>
                  {/* Not in the Library, the original can't be opened: what it says of the file is still true. */}
                  <button
                    type="button"
                    role="menuitem"
                    data-testid="page-from"
                    aria-disabled={original ? undefined : true}
                    title={
                      origin.word
                        ? wordFileTooltip(report, t, language)
                        : t("verify.from.title", { name: origin.name })
                    }
                    onClick={() => {
                      if (!original) return;
                      close();
                      if (origin.word) openOriginalAtCaret(original.id, editor, doc);
                      else openDocumentBeside(original.id);
                    }}
                    className={`${menuItemClass} ${menuItemUnavailableClass} pl-2`}
                  >
                    <span className="min-w-0 flex-1 truncate">
                      {t(origin.word ? "wordImport.from" : "verify.from", { name: origin.name })}
                    </span>
                  </button>
                  {original && (
                    <button
                      type="button"
                      role="menuitem"
                      data-testid="page-show-original"
                      aria-disabled={away ? true : undefined}
                      title={away ? t(away.reason) : undefined}
                      onClick={() => {
                        if (away) return;
                        close();
                        files
                          .showDocumentInFolder(original.id)
                          .catch((error) => useAppStore.getState().reportError(error));
                      }}
                      className={`${menuItemClass} ${menuItemUnavailableClass} pl-2`}
                    >
                      {t(showFileLabelKey(navigator.platform))}
                    </button>
                  )}
                </>
              )
            : undefined
        }
        foot={<MindCounts editor={editor} doc={doc} switchable={!readOnly} />}
      />
      {/* Mounted only while open: closed, it would sit in the page beside the sidebar row's. */}
      {moving && (
        <MoveToPicker
          anchor={moving}
          name={name}
          current={mind?.folderId ?? null}
          onChoose={(folderId) => {
            if (folderId !== mind?.folderId) {
              void useAppStore.getState().moveToFolder({ mindIds: [mindId] }, folderId);
            }
          }}
          onClose={() => setMoving(null)}
        />
      )}
    </span>
  );
}
