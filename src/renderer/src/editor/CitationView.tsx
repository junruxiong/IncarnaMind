import { NodeViewWrapper, type ReactNodeViewProps } from "@tiptap/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  ANSWER_BLOCK,
  BLOCK_ID_ATTRIBUTE,
  type CitationAttributes,
  type CitationCheck,
} from "../../../core/api";
import { badgeMessage, type CitationState, citationState } from "../../../shared/citations";
import { citationLocation, formatLocation, openKind } from "../../../shared/locations";
import { useAnswers } from "../answers";
import { citationHandle } from "../commands/handles";
import { useTabPlace } from "../components/tabs/tabItems";
import { useT } from "../i18n";
import { groupsMerged } from "../layoutSlice";
import { recordUsage, useAppStore } from "../store";
import { citationNumberOf } from "./citationNumbers";
import { CheckMarkIcon, CheckStateIcon } from "./icons";
import { useMindId } from "./mindContext";
import { Popover } from "./Popover";
import { onPeek } from "./peek";

/** How long a pointer rests on a Citation before its card shows, and how long it may leave before it goes. */
const SHOW_MS = 120;
const HIDE_MS = 200;

/** Whether opening the cited place washes the quote: it is there, unless the check could not find it. */
const quoteSaysSo = (check: CitationCheck) =>
  check === "found" || check === "unsupported" || check === "contradicted";

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
 * or, in a narrow Mind, as an icon inside the marker. Hovering or focusing it
 * (or its mark in the margin) shows its card: the check in words, the quote,
 * and Open; Esc closes it. Clicking it opens the cited page and keeps the
 * card; when the quote wasn't found, the card offers ways on: open the page
 * anyway, remove the Citation, or regenerate the Answer.
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
  const kept = useAppStore((state) => state.keptCitationTexts);
  const openDocument = useAppStore((state) => state.openDocument);
  /** The Mind's group: the Document opens in the group to its right. */
  const place = useTabPlace();
  const state = citationState(attributes, documents, kept);
  const location = citationLocation(attributes);
  const where = location ? formatLocation(location, t) : null;
  const badge = badgeMessage(state, attributes, t);
  const badgeText = t(badge.key, badge.params);
  const documentName = attributes.documentName ?? "";
  const number = citationNumberOf(decorations);
  /** Clicked: the card stays until it is dismissed. */
  const [pinned, setPinned] = useState(false);
  /** Hovered or focused, with the card, the marker or its mark. */
  const [peeking, setPeeking] = useState(false);
  const open = pinned || peeking;
  const marker = useRef<HTMLButtonElement>(null);
  const askers = useRef(new Set<string>());
  const timer = useRef<number | undefined>(undefined);

  /** Shows the card while anyone asks for it; hides it shortly after the last stops. */
  const hold = useCallback((source: string, on: boolean) => {
    if (on) askers.current.add(source);
    else askers.current.delete(source);
    window.clearTimeout(timer.current);
    const wanted = askers.current.size > 0;
    timer.current = window.setTimeout(() => setPeeking(wanted), wanted ? SHOW_MS : HIDE_MS);
  }, []);
  const dismiss = useCallback(() => {
    askers.current.clear();
    window.clearTimeout(timer.current);
    setPinned(false);
    setPeeking(false);
  }, []);
  useEffect(() => {
    const element = marker.current;
    const stop = element && onPeek(element, ({ source, on }) => hold(source, on));
    return () => {
      stop?.();
      window.clearTimeout(timer.current);
    };
  }, [hold]);

  /**
   * Opens the cited place (a page, slide, section, rows or lines), or the
   * Document, with the quote to highlight if asked, and the Citation's
   * check, number and label: the viewer colours the quote by the check
   * (amber when opened anyway after "not found") and shows the check mark,
   * with where it points, beside it.
   */
  const openCited = (withQuote: boolean) => {
    const documentId = state.documentId ?? attributes.documentId;
    if (!documentId) return;
    openDocument(
      {
        documentId,
        // A whole TXT or Markdown file cited before Locations, whose quote wasn't found, opens at the top.
        pageFrom: attributes.pageFrom ?? undefined,
        pageTo: attributes.pageTo ?? undefined,
        quote: withQuote && attributes.quote ? attributes.quote : undefined,
        citation: {
          check: state.check,
          ...(number > 0 ? { number } : {}),
          ...(where ? { label: where } : {}),
          ...(attributes.contentHash ? { contentHash: attributes.contentHash } : {}),
        },
      },
      { as: "citation", ...(place && { from: place.groupId }) },
    );
  };

  const onClick = () => {
    setPinned(true);
    // "Check a Citation in the example", in Get started.
    useAppStore.getState().updateGettingStarted({ citationChecked: true });
    recordUsage({ event: "citation_opened", fields: { check: state.check } });
    if (quoteSaysSo(state.check)) openCited(true);
    // A deleted Document: the viewer shows the stored quote with "Document removed".
    else if (state.reason === "document-removed") openCited(true);
    else if (state.check !== "not-found") openCited(false);
  };

  /** What the card's Open does: the cited place, the quote washed where the check found it. */
  const openFromCard = () => {
    openCited(quoteSaysSo(state.check));
    dismiss();
    // The focus stays in the Mind, unless the window shows one group and the Document came to the front.
    if (!groupsMerged()) editor.view.focus();
  };
  const openFromCardRef = useRef(openFromCard);
  openFromCardRef.current = openFromCard;
  // "Open cited page" (a command) opens the Citation whose card shows.
  useEffect(() => {
    if (!open) return;
    const handle = { open: () => openFromCardRef.current() };
    citationHandle.current = handle;
    return () => {
      if (citationHandle.current === handle) citationHandle.current = null;
    };
  }, [open]);

  // The number, the Document, where it points and the check: the marker only shows the number.
  const label = t("citation.label", {
    number,
    document: where
      ? t("citation.where", { document: documentName, location: where })
      : documentName,
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
        onPointerEnter={() => hold("pointer", true)}
        onPointerLeave={() => hold("pointer", false)}
        onFocus={() => hold("focus", true)}
        onBlur={() => hold("focus", false)}
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
            placement="top"
            onDismiss={dismiss}
            aria-label={label}
            data-testid="citation-card"
            data-check={state.check}
            onPointerEnter={() => hold("card", true)}
            onPointerLeave={() => hold("card", false)}
            onFocus={() => hold("card-focus", true)}
            onBlur={() => hold("card-focus", false)}
          >
            <CitationCard
              state={state}
              attributes={attributes}
              badgeText={badgeText}
              onOpen={(withQuote) => {
                openCited(withQuote);
                dismiss();
                if (!groupsMerged()) editor.view.focus();
              }}
              onRemove={() => {
                dismiss();
                deleteNode();
              }}
              onRegenerate={() => {
                const answer = answerAround(editor, getPos());
                dismiss();
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
  const location = citationLocation(attributes);
  const where = location ? formatLocation(location, t) : null;
  const opens = openKind(location);
  const name = attributes.documentName ?? "";
  const checked = state.check === "found" || state.check === "not-found";
  // Once checked, the state line names the place, or the Document if it names none.
  const source = checked
    ? where
      ? name
      : null
    : where
      ? t("citation.where", { document: name, location: where })
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
      {attributes.quote && state.check !== "not-found" && (
        <figure className="citation-card-quote">
          <figcaption className="sr-only">{t("citation.quote")}</figcaption>
          <blockquote>
            <span className="citation-card-wash">{attributes.quote}</span>
          </blockquote>
        </figure>
      )}
      {attributes.verdict && (
        <p data-testid="citation-verdict" className="citation-card-verdict">
          {attributes.verdict}
        </p>
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
              {t(`citation.openAnyway.${opens}`)}
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
              onClick={() => onOpen(quoteSaysSo(state.check))}
            >
              {t(`citation.open.${opens}`)}
            </button>
          )
        )}
      </div>
    </div>
  );
}
