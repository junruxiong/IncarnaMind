/**
 * What the viewer's selection menu does with the words selected (#101), each
 * time with a Citation of them checked where they were selected
 * (`citeSelection`): Ask about this puts them in the composer as a quote chip,
 * Add to note puts them in a Mind as a quote, and Copy with citation puts them
 * on the clipboard as copying that quote from a Mind would.
 */
import { getSchema } from "@tiptap/core";
import { DOMSerializer, Fragment } from "@tiptap/pm/model";
import {
  type AskedQuote,
  CITATION_NODE,
  type CitationAttributes,
  type CitationCheckReason,
  type CitedSelection,
  type Document,
  MOST_ASKED_QUOTES,
  type SelectedPart,
} from "../../../core/api";
import type { NodeJSON } from "../../../shared/blockJson";
import type { MessageKey } from "../../../shared/i18n";
import { mindName } from "../../../shared/naming";
import { type AddTarget, addInEditor } from "../addToNote";
import { focusComposer } from "../commands/assistant";
import { readMindHandle } from "../commands/handles";
import { NO_MIND, useComposer } from "../composer";
import { core } from "../core";
import { flashBlocks } from "../editor/blockOrigins";
import { sourceContext } from "../editor/copyCitations";
import { addFootnotes, citationsIn, copiedText, wordFootnoteHtml } from "../editor/copyNotes";
import { offerEditorUndo } from "../editor/editorUndo";
import { noteExtensions } from "../editor/noteSchema";
import { whenMindEditor } from "../editor/openEditors";
import { translateNow } from "../i18n";
import { useNotices } from "../notices";
import { useAppStore } from "../store";

/** Words selected in a Document's viewer, part by part (see `pdfSelection`). */
export interface ViewerSelection {
  document: Document;
  parts: SelectedPart[];
}

const MIND_OPEN_WAIT_MS = 5000;

const REFUSED: Partial<Record<CitationCheckReason, MessageKey>> = {
  "too-many-pages": "viewer.selection.refused.pages",
  "no-text": "viewer.selection.refused.noText",
};

async function citedSelection({ document, parts }: ViewerSelection): Promise<CitedSelection> {
  const cited = await core.citeSelection({ documentId: document.id, parts });
  // Only words found where they were selected are cited: a refused Citation's next check could
  // find the same words somewhere else and move it there.
  if (cited.check !== "found") {
    const reason = cited.checkReason && REFUSED[cited.checkReason];
    throw new Error(translateNow(reason ?? "viewer.selection.refused"));
  }
  return cited;
}

async function quoteOf(selection: ViewerSelection): Promise<NodeJSON> {
  const { document } = selection;
  const cited = await citedSelection(selection);
  const citation: CitationAttributes = {
    passageId: cited.passageId,
    documentId: document.id,
    documentName: document.name,
    contentHash: cited.contentHash,
    pageFrom: cited.pageFrom,
    pageTo: cited.pageTo,
    location: cited.location,
    quote: cited.quote,
    check: cited.check,
    checkReason: cited.checkReason,
  };
  return {
    type: "blockquote",
    content: [
      {
        type: "paragraph",
        content: [
          { type: "text", text: cited.quote },
          { type: CITATION_NODE, attrs: { ...citation } },
        ],
      },
    ],
  };
}

const nameOf = (mindId: string) => {
  const mind = useAppStore.getState().minds.find((each) => each.id === mindId);
  return mind ? mindName(mind) : "";
};

function putIn(
  editor: Parameters<typeof addInEditor>[0],
  mindId: string,
  quote: NodeJSON,
  atCursor: boolean,
  made = false,
) {
  const ids = addInEditor(editor, [quote], atCursor, () => undefined);
  if (!ids) return;
  flashBlocks(editor, ids);
  const name = nameOf(mindId);
  offerEditorUndo(editor, (t) =>
    made
      ? t("viewer.selection.addedNew")
      : t("viewer.selection.added", { mind: name || t("mind.untitled") }),
  );
}

/** Whether Add to note has a Mind to add to without asking which: the one the assistant reads. */
export const addsWithoutAsking = (): boolean => readMindHandle.current !== null;

/**
 * Adds the selection to `target` as a quote with its Citation: by default the
 * Mind the assistant reads, at its cursor (or end), as one step of its Undo;
 * a Mind chosen from the list at its end; a new Mind in the Document's Folder.
 * Those two are shown beside the Document.
 */
export async function addSelectionToNote(
  selection: ViewerSelection,
  target?: AddTarget,
): Promise<void> {
  try {
    const read = readMindHandle.current;
    if (!target && !read) return;
    const quote = await quoteOf(selection);
    if (!target || (target.kind === "mind" && read?.mindId === target.mindId)) {
      if (read && !read.editor.isDestroyed) {
        putIn(read.editor, read.mindId, quote, read.cursorInMind() || read.cursorPlaced());
      }
      return;
    }
    const made = target.kind === "new";
    const mindId = made
      ? (await core.createMind({ title: "", folderId: selection.document.folderId })).id
      : target.mindId;
    // Beside the Document, which stays in view.
    await useAppStore.getState().changeLayout({
      type: "open",
      item: { kind: "mind", mindId },
      where: { in: "side" },
    });
    const editor = await whenMindEditor(mindId, MIND_OPEN_WAIT_MS);
    if (editor) {
      putIn(editor, mindId, quote, false, made);
      return;
    }
    // A Mind a newer version wrote opens read-only, with no editor to take the quote.
    const name = nameOf(mindId);
    if (made) await core.deleteMind(mindId);
    useNotices.getState().showError(
      new Error(
        made
          ? translateNow("viewer.selection.notOpenedNew")
          : translateNow("viewer.selection.notOpened", {
              mind: name || translateNow("mind.untitled"),
            }),
      ),
      {
        label: (t) => t("viewer.selection.copy"),
        run: () => copySelectionWithCitation(selection),
      },
    );
  } catch (error) {
    useAppStore.getState().reportError(error);
  }
}

const sameQuote = (one: AskedQuote, other: AskedQuote) =>
  one.documentId === other.documentId &&
  one.quote === other.quote &&
  JSON.stringify(one.location) === JSON.stringify(other.location);

/**
 * Puts the selection in the composer as a quote chip, after any already there, and the cursor
 * in the composer, unfolding the assistant: the Question asked next is about it.
 */
export async function askAboutSelection(selection: ViewerSelection): Promise<void> {
  try {
    const { quote, pageFrom, pageTo, location } = await citedSelection(selection);
    const { document } = selection;
    const asked: AskedQuote = {
      documentId: document.id,
      documentName: document.name,
      quote,
      pageFrom,
      pageTo,
      location,
    };
    const composer = useComposer.getState();
    const key = composer.seat?.mindId ?? NO_MIND;
    const quotes = composer.draft(key).quotes ?? [];
    if (!quotes.some((each) => sameQuote(each, asked))) {
      if (quotes.length >= MOST_ASKED_QUOTES) {
        throw new Error(
          translateNow("error.question.quotes.tooMany", { count: MOST_ASKED_QUOTES }),
        );
      }
      composer.update(key, { quotes: [...quotes, asked] });
    }
    focusComposer();
  } catch (error) {
    useAppStore.getState().reportError(error);
  }
}

let schema: ReturnType<typeof getSchema> | null = null;

/**
 * Copies the selection as copying its quote from a Mind does (#102): HTML
 * with the Citation as a footnote, which pastes back into a Mind as a
 * Citation, and text with "[1]" and its Source.
 */
export async function copySelectionWithCitation(selection: ViewerSelection): Promise<void> {
  try {
    schema ??= getSchema(noteExtensions());
    const fragment = Fragment.from(schema.nodeFromJSON(await quoteOf(selection)));
    const context = sourceContext();
    const holder = document.createElement("div");
    holder.append(DOMSerializer.fromSchema(schema).serializeFragment(fragment));
    addFootnotes(holder, citationsIn(fragment), context);
    await navigator.clipboard.write([
      new ClipboardItem({
        "text/html": new Blob([wordFootnoteHtml(holder.innerHTML)], { type: "text/html" }),
        "text/plain": new Blob([copiedText(fragment, schema, context)], { type: "text/plain" }),
      }),
    ]);
    useNotices.getState().confirm("viewer.selection.copied");
  } catch (error) {
    useAppStore.getState().reportError(error);
  }
}
