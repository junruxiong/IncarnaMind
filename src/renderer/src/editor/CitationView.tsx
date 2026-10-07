import { NodeViewWrapper, type ReactNodeViewProps } from "@tiptap/react";
import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  ANSWER_BLOCK,
  BLOCK_ID_ATTRIBUTE,
  type CitationAttributes,
  type CitationCheck,
} from "../../../core/api";
import {
  badgeMessage,
  type CitationState,
  citationState,
  citedPages,
} from "../../../shared/citations";
import { useAnswers } from "../answers";
import {
  CantCheckIcon,
  CheckingIcon,
  QuoteFoundIcon,
  QuoteNotFoundIcon,
} from "../components/icons";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { useMindId } from "./mindContext";
import { Popover } from "./Popover";

const ICONS: Record<CitationCheck, typeof QuoteFoundIcon> = {
  checking: CheckingIcon,
  found: QuoteFoundIcon,
  "not-found": QuoteNotFoundIcon,
  "cant-check": CantCheckIcon,
};

export function BadgeIcon({ check, className }: { check: CitationCheck; className?: string }) {
  const Icon = ICONS[check];
  return <Icon className={className} />;
}

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
 * A Citation in the text: a small badge with the state of its check (an icon,
 * a colour, the page) that opens the cited page. Clicking it also shows its
 * card: the badge in words, the quote, and, when the quote wasn't found, ways
 * on: open the page anyway, remove the Citation, or regenerate the Answer.
 */
export function CitationView({ node, editor, getPos, deleteNode }: ReactNodeViewProps) {
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
  const [open, setOpen] = useState(false);
  const chip = useRef<HTMLButtonElement>(null);

  /** Opens the cited page (or the Document), highlighting the quote only if it was found. */
  const openCited = (withQuote: boolean) => {
    const documentId = state.documentId ?? attributes.documentId;
    if (!documentId) return;
    openDocument({
      documentId,
      // A Document without pages whose quote wasn't found opens at the top.
      pageFrom: attributes.pageFrom ?? undefined,
      pageTo: attributes.pageTo ?? undefined,
      quote: withQuote && attributes.quote ? attributes.quote : undefined,
    });
  };

  const onClick = () => {
    setOpen(true);
    if (state.check === "found") openCited(true);
    // A deleted Document: the viewer shows the stored quote with "Document removed".
    else if (state.reason === "document-removed") openCited(true);
    else if (state.check !== "not-found") openCited(false);
  };

  const label = t("citation.label", {
    document: pages ? t("citation.where.page", { document: documentName, pages }) : documentName,
    badge: badgeText,
  });
  const short =
    state.check === "checking"
      ? badgeText
      : pages
        ? t("citation.chip.page", { pages })
        : documentName;

  return (
    <NodeViewWrapper
      as="span"
      className="citation"
      data-testid="citation"
      data-check={state.check}
      data-reason={state.reason ?? undefined}
      contentEditable={false}
    >
      <button
        ref={chip}
        type="button"
        data-testid="citation-chip"
        className={`citation-chip citation-chip--${state.check}`}
        aria-label={label}
        title={label}
        aria-haspopup="dialog"
        aria-expanded={open}
        // Keep the editor's selection and focus where they are.
        onMouseDown={(event) => event.preventDefault()}
        onClick={onClick}
      >
        <BadgeIcon check={state.check} className="citation-chip-icon" />
        <span className="citation-chip-text">{short}</span>
      </button>
      {open &&
        chip.current &&
        // In the body, not the editor: ProseMirror must not see the card's events.
        createPortal(
          <Popover
            anchor={chip.current}
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
  const reason =
    state.check === "checking"
      ? t("citation.reason.checking")
      : state.reason
        ? t(`citation.reason.${state.reason}`)
        : null;
  const removed = state.reason === "document-removed";
  return (
    <div className="citation-card" contentEditable={false}>
      <p
        data-testid="citation-badge"
        data-check={state.check}
        className={`citation-badge citation-badge--${state.check}`}
      >
        <BadgeIcon check={state.check} className="size-4 shrink-0" />
        <span>{badgeText}</span>
      </p>
      {reason && (
        <p data-testid="citation-reason" className="citation-card-reason">
          {reason}
        </p>
      )}
      <p className="citation-card-source">
        {pages ? t("citation.where.page", { document: name, pages }) : name}
      </p>
      {attributes.quote && (
        <figure className="citation-card-quote">
          <figcaption>{t("citation.quote")}</figcaption>
          <blockquote>{attributes.quote}</blockquote>
        </figure>
      )}
      {state.check === "found" && <p className="citation-card-meaning">{t("citation.meaning")}</p>}
      <div className="citation-card-actions">
        {state.check === "not-found" ? (
          <>
            <button
              type="button"
              data-testid="citation-open-anyway"
              className="citation-card-action"
              onClick={() => onOpen(false)}
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
              className="citation-card-action"
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
