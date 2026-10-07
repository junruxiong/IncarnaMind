import { NodeViewWrapper, type ReactNodeViewProps } from "@tiptap/react";
import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ANSWER_BLOCK, BLOCK_ID_ATTRIBUTE, type CitationAttributes } from "../../../core/api";
import {
  badgeMessage,
  type CitationState,
  citationState,
  citedPages,
} from "../../../shared/citations";
import { useAnswers } from "../answers";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { citationNumberOf } from "./citationNumbers";
import { CheckMarkIcon, CheckStateIcon } from "./icons";
import { useMindId } from "./mindContext";
import { Popover } from "./Popover";

/** The Answer a Citation sits in, if it does (it may have been copied into a Note). */
function answerAround(
  editor: ReactNodeViewProps["editor"],
  pos: number | undefined,
): { answerId: string; questionId: string } | null {
  if (pos === undefined) return null;
  const resolved = editor.state.doc.resolve(pos);
  for (let depth = resolved.depth; depth > 0; depth--) {
    const node = resolved.node(depth);
    if (node.type.name !== ANSWER_BLOCK) continue;
    const answerId = node.attrs[BLOCK_ID_ATTRIBUTE];
    const questionId = node.attrs.questionId;
    return typeof answerId === "string" && typeof questionId === "string"
      ? { answerId, questionId }
      : null;
  }
  return null;
}

/**
 * A Citation in the text: an 18px marker with its number, the Citation's
 * order within its Answer (see `CitationNumbers`). It stays neutral unless the
 * quote wasn't found; the check itself shows in the margin (`MarginChecks`),
 * or, in a narrow Mind, as an icon inside the marker. Clicking it opens the
 * cited page and shows its card: the check in words, the quote, and, when the
 * quote wasn't found, ways on: open the page anyway, remove the Citation, or
 * regenerate the Answer.
 */
export function CitationView({
  node,
  editor,
  getPos,
  deleteNode,
  decorations,
}: ReactNodeViewProps) {
  const t = useT();
  const mindId = useMindId();
  const attributes = node.attrs as CitationAttributes;
  const documents = useAppStore((state) =>
    state.status.kind === "ready" ? state.documents : null,
  );
  const openDocument = useAppStore((state) => state.openDocument);
  const state = citationState(attributes, documents);
  const pages = citedPages(attributes);
  const badge = badgeMessage(state, attributes);
  const badgeText = t(badge.key, badge.params);
  const documentName = attributes.documentName ?? "";
  const number = citationNumberOf(decorations);
  const [open, setOpen] = useState(false);
  const marker = useRef<HTMLButtonElement>(null);

  /**
   * Opens the cited page (or the Document), with the quote to highlight if
   * asked, and the Citation's check and number: the viewer colours the quote
   * by the check (amber when opened anyway after "not found") and shows the
   * check mark beside it.
   */
  const openCited = (withQuote: boolean) => {
    const documentId = state.documentId ?? attributes.documentId;
    if (!documentId) return;
    openDocument({
      documentId,
      // A Document without pages whose quote wasn't found opens at the top.
      pageFrom: attributes.pageFrom ?? undefined,
      pageTo: attributes.pageTo ?? undefined,
      quote: withQuote && attributes.quote ? attributes.quote : undefined,
      citation: { check: state.check, ...(number > 0 ? { number } : {}) },
    });
  };

  const onClick = () => {
    setOpen(true);
    if (state.check === "found") openCited(true);
    // A deleted Document: the viewer shows the stored quote with "Document removed".
    else if (state.reason === "document-removed") openCited(true);
    else if (state.check !== "not-found") openCited(false);
  };

  // The number, the Document, the page and the check: the marker only shows the number.
  const label = t("citation.label", {
    number,
    document: pages ? t("citation.where.page", { document: documentName, pages }) : documentName,
    badge: badgeText,
  });

  return (
    <NodeViewWrapper
      as="span"
      className="citation"
      data-testid="citation"
      data-check={state.check}
      data-reason={state.reason ?? undefined}
      data-number={number || undefined}
      contentEditable={false}
    >
      <button
        ref={marker}
        type="button"
        data-testid="citation-chip"
        data-check={state.check}
        data-number={number || undefined}
        className="citation-marker"
        aria-label={label}
        title={label}
        aria-haspopup="dialog"
        aria-expanded={open}
        // Keep the editor's selection and focus where they are.
        onMouseDown={(event) => event.preventDefault()}
        onClick={onClick}
      >
        <CheckMarkIcon check={state.check} className="citation-marker-icon" />
        <span>{number || "·"}</span>
      </button>
      {open &&
        marker.current &&
        // In the body, not the editor: ProseMirror must not see the card's events.
        createPortal(
          <Popover
            anchor={marker.current}
            onDismiss={() => setOpen(false)}
            aria-label={label}
            data-testid="citation-card"
          >
            <CitationCard
              state={state}
              attributes={attributes}
              badgeText={badgeText}
              onOpen={(withQuote) => {
                openCited(withQuote);
                setOpen(false);
              }}
              onRemove={() => {
                setOpen(false);
                deleteNode();
              }}
              onRegenerate={() => {
                const answer = answerAround(editor, getPos());
                setOpen(false);
                if (answer) {
                  void useAnswers.getState().regenerate(mindId, answer.answerId, answer.questionId);
                }
              }}
              inAnswer={answerAround(editor, getPos()) !== null}
            />
          </Popover>,
          document.body,
        )}
    </NodeViewWrapper>
  );
}

/**
 * A Citation's card: the state line with its icon, the Document, the quote
 * washed in the state's colour, what the state means (or why the quote wasn't
 * found), and what to do next.
 */
function CitationCard({
  state,
  attributes,
  badgeText,
  inAnswer,
  onOpen,
  onRemove,
  onRegenerate,
}: {
  state: CitationState;
  attributes: CitationAttributes;
  badgeText: string;
  inAnswer: boolean;
  onOpen(withQuote: boolean): void;
  onRemove(): void;
  onRegenerate(): void;
}) {
  const t = useT();
  const pages = citedPages(attributes);
  const name = attributes.documentName ?? "";
  const checked = state.check === "found" || state.check === "not-found";
  // Once checked, the state line names the page, or the Document if it has no pages.
  const source = checked
    ? pages
      ? name
      : null
    : pages
      ? t("citation.where.page", { document: name, pages })
      : name;
  const reason =
    state.check === "checking"
      ? t("citation.reason.checking")
      : state.reason
        ? t(`citation.reason.${state.reason}`)
        : null;
  const removed = state.reason === "document-removed";
  return (
    <div className={`citation-card citation-card--${state.check}`} contentEditable={false}>
      <div className="citation-card-head">
        <p data-testid="citation-badge" data-check={state.check} className="citation-badge">
          <CheckStateIcon check={state.check} className="citation-badge-icon" />
          <span>{badgeText}</span>
        </p>
        {source && <p className="citation-card-source">{source}</p>}
      </div>
      {attributes.quote && (
        <figure className="citation-card-quote">
          <figcaption className="sr-only">{t("citation.quote")}</figcaption>
          <blockquote>
            <span className="citation-card-wash">{attributes.quote}</span>
          </blockquote>
        </figure>
      )}
      {reason && (
        <p data-testid="citation-reason" className="citation-card-note">
          {reason}
        </p>
      )}
      {state.check === "found" && <p className="citation-card-note">{t("citation.meaning")}</p>}
      <div className="citation-card-actions">
        {state.check === "not-found" ? (
          <>
            <button
              type="button"
              data-testid="citation-open-anyway"
              className="citation-card-action citation-card-action--primary"
              // With the quote: the viewer washes it amber wherever it does find it.
              onClick={() => onOpen(true)}
            >
              {t(pages ? "citation.openAnyway.page" : "citation.openAnyway.document")}
            </button>
            <button
              type="button"
              data-testid="citation-remove"
              className="citation-card-action"
              onClick={onRemove}
            >
              {t("citation.remove")}
            </button>
            {inAnswer && (
              <button
                type="button"
                data-testid="citation-regenerate"
                className="citation-card-action"
                onClick={onRegenerate}
              >
                {t("citation.regenerate")}
              </button>
            )}
          </>
        ) : (
          !removed && (
            <button
              type="button"
              data-testid="citation-open"
              className="citation-card-action citation-card-action--primary"
              onClick={() => onOpen(state.check === "found")}
            >
              {t(pages ? "citation.open.page" : "citation.open.document")}
            </button>
          )
        )}
      </div>
    </div>
  );
}
