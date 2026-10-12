import { NodeViewWrapper, type ReactNodeViewProps } from "@tiptap/react";
import { useCallback, useContext, useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  ANSWER_BLOCK,
  BLOCK_ID_ATTRIBUTE,
  type CitationAttributes,
  type CitationCheck,
  type FormattedCluster,
  type StyledRun,
} from "../../../core/api";
import {
  badgeMessage,
  type CitationState,
  citationSource,
  citationState,
} from "../../../shared/citations";
import {
  citationLocation,
  formatLocation,
  isTranscriptLocation,
  openKind,
} from "../../../shared/locations";
import { useAnswers } from "../answers";
import { type MindFormatting, useMindFormatting } from "../citationFormatting";
import { citationHandle } from "../commands/handles";
import { type CommandContext, commands } from "../commands/registry";
import { ObjectMenu, useObjectMenu } from "../components/ObjectMenu";
import { useTabPlace } from "../components/tabs/tabItems";
import { useGap, useLanguage, useT } from "../i18n";
import { groupsMerged } from "../layoutSlice";
import type { ObjectRef } from "../objects";
import { recordUsage, useAppStore } from "../store";
import { pageDate } from "../webPages";
import {
  type ClusterPlace,
  citationClusters,
  clusterPlaceOf,
  memberMarker,
} from "./citationClusters";
import { citationNumberOf } from "./citationNumbers";
import { QuoteWithUnsureWords, ScanLine, useScannedQuote } from "./citationScan";
import { CheckMarkIcon, CheckStateIcon } from "./icons";
import { InFootnoteContext, MindIdContext, RegenerateAnswerContext } from "./mindContext";
import { Popover } from "./Popover";
import { onPeek, peekCitation } from "./peek";
import { StyledRuns, StyledRunView } from "./StyledRuns";
import { runsText } from "./styledText";

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
 * A Citation in the text: in the Mind's citation style (#289), a chip in the
 * style's words, "(Bowler et al., 2010, p. 4)" or "[1]", which the first of
 * Citations side by side draws for all of them (see `CitationClusters`); in a
 * note style, or in Numbered, an 18px marker with its number, its place among
 * the Mind's footnotes (see `CitationNumbers`). It stays neutral unless the
 * quote wasn't found; the check itself shows in the margin (`MarginChecks`),
 * or, in a narrow Mind, as an icon inside the marker. In a footnote's text
 * it is part of the footnote: the marker shows where it points and its
 * check's icon instead of a number. Hovering or focusing it (or its mark in
 * the margin) shows its card: the check in words, the quote, and Open; Esc
 * closes it. Clicking it opens the cited page and keeps the card; when the
 * quote wasn't found, the card offers ways on: open the page anyway, remove
 * the Citation, or regenerate the Answer.
 */
export function CitationView({
  node,
  editor,
  getPos,
  deleteNode,
  decorations,
}: ReactNodeViewProps) {
  const t = useT();
  // None in the assistant, whose Answers are read only and say how they are written again.
  const mindId = useContext(MindIdContext);
  const regenerateHere = useContext(RegenerateAnswerContext);
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
  const number = citationNumberOf(decorations);
  // In a footnote's text it is part of the footnote: it shows where it points, not a number.
  const inFootnote = useContext(InFootnoteContext);
  const pointsTo = citationSource(attributes, t).text;
  /** Clicked: the card stays until it is dismissed. */
  const [pinned, setPinned] = useState(false);
  /** Hovered or focused, with the card, the marker or its mark. */
  const [peeking, setPeeking] = useState(false);
  const open = pinned || peeking;
  const marker = useRef<HTMLButtonElement>(null);
  /** The marker whose card the pointer over this one asks for: its own, or a neighbour's. */
  const pointed = useRef<HTMLElement | null>(null);
  const formatting = useMindFormatting(mindId);
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

  const menuId = useId();
  const objectMenu = useObjectMenu({ kind: "citation", id: menuId });
  const here = {
    ...(state.reason !== "document-removed" && {
      "citation.open": () => openFromCardRef.current(),
    }),
    // Not where the text can't be changed, such as an Answer in the assistant.
    ...(editor.isEditable && {
      "citation.remove": () => {
        dismiss();
        deleteNode();
      },
    }),
  };

  // The number, the Document, where it points and the check: the marker only shows the number.
  const label = inFootnote
    ? t("citation.inFootnote.label", { document: pointsTo, badge: badgeText })
    : t("citation.label", { number, document: pointsTo, badge: badgeText });

  // In the Mind's citation style (#289): the first Citation of a cluster draws it whole.
  const inCluster = clusterPlaceOf(decorations);
  const styled = styledMarker(formatting, inCluster);
  const members = inCluster
    ? (citationClusters(editor.state)[inCluster.cluster]?.members ?? [])
    : [];
  /** The marker of the cluster's Citation that wrote `work`: this one's, or a neighbour's. */
  const markerOfWork = (work: number | null) =>
    inCluster === null || work === null || work === inCluster.member
      ? marker.current
      : memberMarker(editor.view.dom, inCluster.cluster, work);
  /** Shows the card of the Citation the pointer is over, and hides the one it left. */
  const pointAt = (target: HTMLElement | null) => {
    const before = pointed.current;
    if (before === target) return;
    pointed.current = target;
    if (before === marker.current) hold("pointer", false);
    else if (before) peekCitation(before, "cluster", false);
    if (target === marker.current) hold("pointer", true);
    else if (target) peekCitation(target, "cluster", true);
  };

  return (
    <NodeViewWrapper
      as="span"
      className="citation"
      data-testid="citation"
      data-check={state.check}
      data-reason={state.reason ?? undefined}
      data-number={number || undefined}
      data-styled={styled.kind === "numbered" ? undefined : styled.kind}
      contentEditable={false}
      {...objectMenu.rowProps}
    >
      <button
        ref={marker}
        type="button"
        data-testid="citation-chip"
        data-check={state.check}
        data-number={number || undefined}
        className={`citation-marker ${
          inFootnote
            ? "citation-marker--source"
            : styled.kind === "joined"
              ? "citation-marker--joined"
              : styled.kind === "runs"
                ? `citation-marker--styled ${styled.raised ? "citation-marker--raised" : ""}`
                : ""
        }`}
        aria-label={label}
        title={styled.kind === "runs" ? undefined : label}
        aria-haspopup="dialog"
        aria-expanded={open}
        // Keep the editor's selection and focus where they are.
        onMouseDown={(event) => event.preventDefault()}
        // In a cluster, the pointer shows the card of the Citation whose words it is over.
        onPointerMove={(event) => pointAt(markerOfWork(workAt(event.target, inCluster)))}
        onPointerLeave={() => pointAt(null)}
        onFocus={() => hold("focus", true)}
        onBlur={() => hold("focus", false)}
        onClick={(event) => {
          const target = markerOfWork(workAt(event.target, inCluster));
          if (target && target !== marker.current) target.click();
          else onClick();
        }}
      >
        <CheckMarkIcon check={state.check} className="citation-marker-icon" />
        {inFootnote ? (
          <span>{pointsTo}</span>
        ) : styled.kind === "runs" ? (
          <ClusterRuns
            runs={styled.runs}
            checks={members.map((member) =>
              member.attributes === attributes
                ? state.check
                : citationState(member.attributes, documents, kept).check,
            )}
          />
        ) : styled.kind === "joined" ? null : (
          // A Transcript's Citation shows the time it points at, "14:26", in Numbered (#136).
          <span data-testid="citation-chip-text">
            {styled.kind === "numbered" && isTranscriptLocation(location) && where
              ? where
              : number || "·"}
          </span>
        )}
      </button>
      <ObjectMenu
        target={{ kind: "citation", id: menuId }}
        label={t("citation.more")}
        here={here}
        className="citation-menu-anchor"
        tabIndex={-1}
      />
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
              onRemove={
                editor.isEditable
                  ? () => {
                      dismiss();
                      deleteNode();
                    }
                  : null
              }
              onRegenerate={() => {
                dismiss();
                if (regenerateHere) {
                  regenerateHere();
                  return;
                }
                const answer = answerAround(editor, getPos());
                if (answer && mindId) {
                  void useAnswers.getState().regenerate(mindId, answer.answerId, answer.questionId);
                }
              }}
              inAnswer={
                regenerateHere !== null ||
                (mindId !== null && answerAround(editor, getPos()) !== null)
              }
              menu={{ target: { kind: "citation", id: menuId }, here }}
              note={noteOf(formatting, inCluster)}
            />
          </Popover>,
          document.body,
        )}
    </NodeViewWrapper>
  );
}

/** How a Citation's marker reads in the Mind's citation style. */
type StyledMarker =
  /** IncarnaMind's numbered chip: Numbered, or the style's form until it is formatted. */
  | { kind: "numbered" }
  /**
   * The style's in-text form, drawn by the cluster's first Citation; `raised` when the style
   * raises all of it (GB/T 7714's "[1]"), which the chip then does instead of its text.
   */
  | { kind: "runs"; runs: FormattedCluster["runs"]; raised: boolean }
  /** A note style: the note's number among the Mind's footnotes (`CitationNumbers`). */
  | { kind: "note" }
  /** A later Citation of a cluster its first one draws: nothing of its own to show. */
  | { kind: "joined" };

/**
 * What a Citation's marker shows: the style's form once its cluster is formatted as it is now
 * (the answer may be for clusters that have changed since), IncarnaMind's number until then.
 */
function styledMarker(
  formatting: MindFormatting | undefined,
  place: ClusterPlace | null,
): StyledMarker {
  const formatted = formatting?.formatted;
  if (!formatting || !formatted || !place || formatted.form === "numbered") {
    return { kind: "numbered" };
  }
  if (formatted.style !== formatting.style) return { kind: "numbered" };
  const cluster = formatted.clusters[place.cluster];
  if (!cluster || formatting.signatures[place.cluster] !== place.signature) {
    return { kind: "numbered" };
  }
  if (place.member > 0) return { kind: "joined" };
  if (formatted.form === "note") return { kind: "note" };
  const raised =
    cluster.runs.length > 0 &&
    cluster.runs.every((run) => run.script === "sup" || run.text.trim() === "");
  return {
    kind: "runs",
    runs: raised ? cluster.runs.map(({ script: _, ...run }) => run) : cluster.runs,
    raised,
  };
}

/** In a note style, the note a Citation's cluster is written as, once it is formatted. */
function noteOf(
  formatting: MindFormatting | undefined,
  place: ClusterPlace | null,
): StyledRun[] | null {
  const formatted = formatting?.formatted;
  if (!formatted || !place || formatted.form !== "note") return null;
  if (formatting.signatures[place.cluster] !== place.signature) return null;
  return (formatted.clusters[place.cluster]?.runs ?? []).map(({ work: _, ...run }) => run);
}

/** Which of a cluster's works the element is part of; its own place for its affixes. */
function workAt(target: EventTarget | null, place: ClusterPlace | null): number | null {
  if (!place || !(target instanceof Element)) return null;
  const work = target.closest<HTMLElement>("[data-work]")?.dataset.work;
  return work === undefined ? place.member : Number(work);
}

/**
 * A cluster in the style's words, each work's words in its own Citation's colours (a quote
 * not found turns amber), and no links: a click here opens the Citation, not an address.
 */
function ClusterRuns({
  runs,
  checks,
}: {
  runs: FormattedCluster["runs"];
  /** Each member's check, by its place in the cluster. */
  checks: readonly CitationCheck[];
}) {
  return runs.map(({ work, link: _, ...run }, index) => (
    <span
      // biome-ignore lint/suspicious/noArrayIndexKey: runs have no identity but their place.
      key={index}
      data-work={work}
      data-check={work === undefined ? undefined : checks[work]}
      className="citation-run"
    >
      <StyledRunView run={run} />
    </span>
  ));
}

/**
 * A Citation's card: the state line with its icon, the Document (for a web
 * page, its site and when it was fetched too), the quote washed in the
 * state's colour, what the state means (or why the quote wasn't found), and
 * what to do next (for a web page, also "Open the live page").
 */
function CitationCard({
  state,
  attributes,
  badgeText,
  inAnswer,
  note,
  onOpen,
  onRemove,
  onRegenerate,
  menu,
}: {
  /** The ⋯ at the card's end: the same menu as the right-click on the marker. */
  menu: { target: ObjectRef; here: CommandContext["here"] };
  state: CitationState;
  attributes: CitationAttributes;
  badgeText: string;
  inAnswer: boolean;
  /** In a note style, the note the Citation is written in, as an export will write it. */
  note: readonly StyledRun[] | null;
  onOpen(withQuote: boolean): void;
  /** Null where the text can't be changed, such as an Answer in the assistant. */
  onRemove: (() => void) | null;
  onRegenerate(): void;
}) {
  const t = useT();
  const language = useLanguage();
  const gap = useGap();
  const web = useAppStore(
    (store) => store.documents.find((each) => each.id === state.documentId)?.web ?? null,
  );
  const location = citationLocation(attributes);
  const where = location ? formatLocation(location, t) : null;
  const opens = web ? "web" : openKind(location);
  const name = attributes.documentName ?? "";
  // When the version it quotes was fetched: an older one's, if the page changed since.
  const fetchedAt =
    web?.versions.find((each) => each.contentHash === attributes.contentHash)?.fetchedAt ??
    web?.fetchedAt;
  const checked = state.check === "found" || state.check === "not-found";
  // Once checked, the state line names the place, or the Document if it names none.
  const source = checked
    ? where
      ? name
      : null
    : where
      ? citationSource(attributes, t).text
      : name;
  const reason =
    state.check === "checking"
      ? t("citation.reason.checking")
      : state.reason
        ? t(`citation.reason.${state.reason}`)
        : null;
  const removed = state.reason === "document-removed";
  const scanned = useScannedQuote(attributes);
  return (
    <div className={`citation-card citation-card--${state.check}`} contentEditable={false}>
      <div className="citation-card-head">
        <p data-testid="citation-badge" data-check={state.check} className="citation-badge">
          <CheckStateIcon check={state.check} className="citation-badge-icon" />
          <span>{badgeText}</span>
        </p>
        {source && <p className="citation-card-source">{source}</p>}
        {scanned && <ScanLine scanned={scanned} />}
        {web && fetchedAt && (
          <p data-testid="citation-web-source" className="citation-card-source">
            {web.siteName} · {t("webPages.fetched", { date: pageDate(fetchedAt, language) })}
          </p>
        )}
      </div>
      {attributes.quote && state.check !== "not-found" && (
        <figure className="citation-card-quote">
          <figcaption className="sr-only">{t("citation.quote")}</figcaption>
          <blockquote>
            <span className="citation-card-wash">
              <QuoteWithUnsureWords quote={attributes.quote} unsure={scanned?.unsure ?? []} />
            </span>
          </blockquote>
        </figure>
      )}
      {attributes.verdict && (
        <p data-testid="citation-verdict" className="citation-card-verdict">
          {attributes.verdict}
        </p>
      )}
      {note && (
        <p data-testid="citation-styled-note" className="citation-card-styled">
          <span className="citation-card-styled-label">{t("citation.note")}</span>
          {gap(t("citation.note"), runsText(note))}
          <StyledRuns runs={note} />
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
            {onRemove && (
              <button
                type="button"
                data-testid="citation-remove"
                className="citation-card-action"
                onClick={onRemove}
              >
                {t("citation.remove")}
              </button>
            )}
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
        <ObjectMenu
          target={menu.target}
          label={t("citation.more")}
          here={menu.here}
          className="citation-card-action ml-auto"
        />
        {web && state.documentId && (
          <button
            type="button"
            data-testid="citation-open-live"
            title={web.address}
            className="citation-card-action"
            onClick={() => {
              if (!state.documentId) return;
              const focus = { kind: "document" as const, id: state.documentId };
              commands
                .run("document.openLivePage", { selection: [], focus, surface: "mind" })
                .catch(useAppStore.getState().reportError);
            }}
          >
            {t("webPages.openLive")}
          </button>
        )}
      </div>
    </div>
  );
}
