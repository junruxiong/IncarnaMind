import type { Editor } from "@tiptap/core";
import { type CSSProperties, useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type * as Y from "yjs";
import { CITATION_NODE, type CitationAttributes, type ClaimCalculation } from "../../../core/api";
import { shownDocumentName } from "../../../shared/citations";
import { formulaText } from "../../../shared/formula";
import { citationLocation, formatLocation, openKind } from "../../../shared/locations";
import { savedPageSource } from "../../../shared/webSource";
import { addAsCitation, checkAgain, removeClaimMark } from "../checks";
import { CountItLine } from "../components/CountItLine";
import { askInAssistant } from "../conversations";
import { useFindSources } from "../findSources";
import { translateNow, useLanguage, useT } from "../i18n";
import { groupsMerged } from "../layoutSlice";
import { openCited } from "../openCited";
import { useAppStore } from "../store";
import { useReadableQuote } from "./citationScan";
import type { ShownClaim } from "./claimChecks";
import { ClaimMarkIcon, type ClaimMarkState } from "./icons";
import { Popover } from "./Popover";

/** How long a pointer rests on a mark before its card shows, and how long it may leave before it goes. */
const SHOW_MS = 120;
const HIDE_MS = 200;

/** The card's colours, as a Citation card's in the same state. */
const TONE: Record<ClaimMarkState, string> = {
  checking: "checking",
  supported: "found",
  unsupported: "unsupported",
  contradicted: "contradicted",
  "not-checked": "cant-check",
  "needs-recheck": "checking",
};

/** Whether the sentence already cites the Passage its check found. */
function citesEvidence(editor: Editor, claim: ShownClaim): boolean {
  const evidence = claim.check.evidence;
  if (!evidence) return false;
  let cited = false;
  editor.state.doc.nodesBetween(claim.sentence.from, claim.sentence.to, (node) => {
    if (node.type.name !== CITATION_NODE) return true;
    const attributes = node.attrs as CitationAttributes;
    if (attributes.documentId === evidence.documentId && attributes.quote === evidence.quote) {
      cited = true;
    }
    return false;
  });
  return cited;
}

/** The words of the sentence a check is about. */
const sentenceOf = (editor: Editor, claim: ShownClaim) =>
  editor.state.doc.textBetween(claim.sentence.from, claim.sentence.to, " ", " ").trim();

/**
 * Asks the assistant to look for the sentence on the web (#128), which a model
 * that can search does. The card says so first when web search is off (`ClaimCard`).
 */
function searchTheWeb(mindId: string, sentence: string): void {
  void askInAssistant({
    text: translateNow("claim.webQuestion", { sentence }),
    mindId,
  });
}

/**
 * A sentence check's mark in the margin (#96), level with the sentence's
 * first line: a quiet tick with "+" when the Passage found can become its
 * Citation, an amber "!", an outlined red "≠", a grey ring when it wasn't
 * checked, ↻ when it needs a re-check. Hovering or focusing it shows its
 * card, and clicking keeps the card until it is dismissed; Esc closes it.
 */
export function ClaimMark({
  claim,
  editor,
  mindId,
  doc,
  label,
  style,
  reveal,
  onRevealed,
}: {
  claim: ShownClaim;
  editor: Editor;
  mindId: string;
  doc: Y.Doc;
  label: string;
  style: CSSProperties;
  reveal: boolean;
  onRevealed(): void;
}) {
  const state = claim.shown.state;
  const button = useRef<HTMLButtonElement>(null);
  const [pinned, setPinned] = useState(false);
  const [peeking, setPeeking] = useState(false);
  const askers = useRef(new Set<string>());
  const timer = useRef<number | undefined>(undefined);
  const open = pinned || peeking;
  // A figure worked out has inputs, not one Passage to cite.
  const plus = state === "supported" && !claim.check.calculation && !citesEvidence(editor, claim);

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
  useEffect(() => () => window.clearTimeout(timer.current), []);

  /** After an action the card goes, and the focus stays in the Mind. */
  const done = () => {
    dismiss();
    if (!editor.isDestroyed && !groupsMerged()) editor.view.focus();
  };

  return (
    <>
      <button
        ref={button}
        type="button"
        data-testid="claim-mark"
        data-check={state}
        data-claim={claim.check.id}
        data-open={open || undefined}
        data-addable={plus || undefined}
        className={`margin-check claim-mark ${reveal ? "margin-check--reveal" : ""}`}
        style={style}
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
        onClick={() => {
          setPinned(true);
          // "Open a check in the example", in Get started.
          const { examples, updateGettingStarted } = useAppStore.getState();
          if (examples?.mindId === mindId) updateGettingStarted({ citationChecked: true });
        }}
        onAnimationEnd={(event) => {
          if (event.target === event.currentTarget && reveal) onRevealed();
        }}
      >
        <ClaimMarkIcon state={state} className="margin-check-icon" />
        {plus && <span aria-hidden="true">+</span>}
      </button>
      {open &&
        button.current &&
        createPortal(
          <Popover
            anchor={button.current}
            placement="bottom-end"
            onDismiss={dismiss}
            aria-label={label}
            data-testid="claim-card"
            data-check={state}
            onPointerEnter={() => hold("card", true)}
            onPointerLeave={() => hold("card", false)}
            onFocus={() => hold("card-focus", true)}
            onBlur={() => hold("card-focus", false)}
          >
            <ClaimCard
              claim={claim}
              addable={plus}
              onAdd={() => {
                addAsCitation(editor, claim);
                done();
              }}
              onCheckAgain={(allDocuments) => {
                void checkAgain(editor, mindId, doc, claim, allDocuments);
                done();
              }}
              onOpen={done}
              onFix={() => {
                editor
                  .chain()
                  .focus()
                  .setTextSelection({ from: claim.sentence.from, to: claim.sentence.to })
                  .run();
                done();
              }}
              onFindSources={() => {
                useFindSources.getState().show({ query: sentenceOf(editor, claim) });
                done();
              }}
              onSearchWeb={() => {
                searchTheWeb(mindId, sentenceOf(editor, claim));
                done();
              }}
              onTurnOnWebSearch={() => {
                useAppStore.getState().openSettings("search");
                done();
              }}
              onRemove={() => {
                removeClaimMark(editor, claim);
                done();
              }}
            />
          </Popover>,
          document.body,
        )}
    </>
  );
}

/**
 * A figure the check worked out (#707): how ("£1,234m ÷ £942m = 1.31x"),
 * what the sentence says when they differ, and each input's Citation, which
 * opens its Document at the quote. "Calculated", never "verified" (ADR-0021).
 */
function Working({ calculation, onOpen }: { calculation: ClaimCalculation; onOpen(): void }) {
  const t = useT();
  const documents = useAppStore((store) => store.documents);
  const { figure } = calculation;
  const byName = new Map(figure.inputs.map((input) => [input.name.toLowerCase(), input]));
  const working = formulaText(
    figure.formula,
    (name) => byName.get(name.toLowerCase())?.number.chars ?? null,
  );
  return (
    <div data-testid="claim-working" className="claim-card-working">
      <p className="citation-card-note">{t("claim.calculated.title")}</p>
      <p data-testid="claim-figure" className="claim-card-figure">
        {working} = {figure.shown}
      </p>
      {!calculation.agrees && (
        <p data-testid="claim-stated" className="claim-card-says">
          {t("claim.calculated.stated", { stated: calculation.stated })}
        </p>
      )}
      <ul aria-label={t("claim.calculated.inputs")} className="claim-card-inputs">
        {figure.inputs.map((input) => (
          <li key={input.name}>
            <WorkingInput input={input} documents={documents} onOpen={onOpen} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function WorkingInput({
  input,
  documents,
  onOpen,
}: {
  input: ClaimCalculation["figure"]["inputs"][number];
  documents: ReturnType<typeof useAppStore.getState>["documents"];
  onOpen(): void;
}) {
  const t = useT();
  const quote = useReadableQuote(input.citation);
  const location = citationLocation(input.citation);
  const where = location ? formatLocation(location, t) : null;
  const source = [shownDocumentName(input.citation, documents), where].filter(Boolean).join(" · ");
  return (
    <button
      type="button"
      data-testid="claim-input"
      className="claim-card-input"
      title={quote}
      onClick={() => {
        openCited(input.citation, { check: "found", withQuote: true, label: where });
        onOpen();
      }}
    >
      <span className="claim-card-input-figure">
        {input.label ? `${input.label}: ` : ""}
        {input.number.chars}
      </span>
      <span className="claim-card-input-source">{source}</span>
    </button>
  );
}

/**
 * A sentence check's card: its state; for a Passage found, the Document and
 * Location with the quote washed in the state's colour, and Add as Citation
 * when it isn't cited yet; for a contradiction, what the Document says; for a
 * figure worked out, its working and inputs; for nothing found, where it
 * looked, with Search all Documents, or, for a court decision no Document
 * holds, that it can't be checked here, with Remove the mark only; otherwise
 * why, with Check again. A Passage from a saved web page names the page.
 * Documents IncarnaMind wrote that matched are named, with Count it.
 */
function ClaimCard({
  claim,
  addable,
  onAdd,
  onCheckAgain,
  onOpen,
  onFix,
  onFindSources,
  onSearchWeb,
  onTurnOnWebSearch,
  onRemove,
}: {
  claim: ShownClaim;
  addable: boolean;
  onAdd(): void;
  onCheckAgain(allDocuments?: boolean): void;
  onOpen(): void;
  onFix(): void;
  onFindSources(): void;
  onSearchWeb(): void;
  onTurnOnWebSearch(): void;
  onRemove(): void;
}) {
  const t = useT();
  const language = useLanguage();
  const webSearchOn = useAppStore((store) => store.settings?.user.webSearch === true);
  // Search the web with web search off says why, and where to turn it on, before anything else happens.
  const [webOffShown, setWebOffShown] = useState(false);
  const { shown, check } = claim;
  const state = shown.state;
  const decided = state === "supported" || state === "contradicted";
  const calculation = decided ? check.calculation : null;
  // A figure that agrees can sit beside a Passage that contradicts the rest of the sentence.
  const fromPassage = !calculation || (state === "contradicted" && calculation.agrees);
  // An unsupported sentence keeps the nearest line when it says something different (#770).
  const differs = state === "unsupported" && check.evidence !== null;
  const evidence = (decided && fromPassage) || differs ? check.evidence : null;
  const location = evidence ? citationLocation(evidence) : null;
  const quote = useReadableQuote(evidence);
  const where = location ? formatLocation(location, t) : null;
  const decisionMissing = state === "unsupported" && check.citedCaseUnchecked;
  const decisionUnchecked = decided && check.citedCaseUnchecked;
  const decisionHeld =
    check.citedCaseFound &&
    (state === "supported" || state === "contradicted" || state === "unsupported") &&
    evidence !== null &&
    where !== null;
  const folderName = useAppStore((store) =>
    check.scope.folderId === null
      ? null
      : (store.library?.groups.find((group) => group.id === check.scope.folderId)?.name ?? null),
  );
  const documents = useAppStore((store) => store.documents);
  const savedPage = evidence
    ? documents.find((each) => each.id === evidence.documentId)?.web
    : undefined;
  const savedFrom = savedPage && {
    address: savedPage.address,
    savedAt:
      savedPage.versions.find((each) => each.contentHash === evidence?.contentHash)?.fetchedAt ??
      savedPage.fetchedAt,
  };
  const searchedFolder = !check.scope.widened && folderName !== null;
  const ownWorkings =
    state === "unsupported" && check.excluded.some((left) => left.reason === "own-workings");
  const reason =
    shown.state === "not-checked"
      ? t(`claim.reason.${shown.reason}`, { detail: check.detail ?? "" })
      : shown.state === "needs-recheck"
        ? t(`claim.recheck.${shown.why}`)
        : null;

  return (
    <div className={`citation-card citation-card--${TONE[state]}`} contentEditable={false}>
      <div className="citation-card-head">
        <p data-testid="claim-badge" data-check={state} className="citation-badge">
          <ClaimMarkIcon state={state} className="citation-badge-icon" />
          <span>
            {decisionMissing ? t("claim.courtDecision.title") : t(`claim.state.${state}`)}
          </span>
        </p>
        {state === "contradicted" && check.says && (
          <p data-testid="claim-says" className="claim-card-says">
            {check.says}
          </p>
        )}
        {differs && (
          <p data-testid="claim-differs" className="claim-card-says">
            {t("claim.unsupported.differs")}
          </p>
        )}
        {evidence && (
          <p data-testid="claim-source" className="citation-card-source">
            {[shownDocumentName(evidence, documents), where].filter(Boolean).join(" · ")}
          </p>
        )}
        {savedFrom && (
          <p
            data-testid="claim-web-source"
            className="citation-card-source whitespace-normal"
            title={savedPageSource(savedFrom, language, t, { compact: false })}
          >
            {savedPageSource(savedFrom, language, t, { compact: true })}
          </p>
        )}
      </div>
      {evidence?.quote && (
        <figure className="citation-card-quote">
          <figcaption className="sr-only">{t("citation.quote")}</figcaption>
          <blockquote>
            <span data-testid="claim-quote" className="citation-card-wash">
              {quote}
            </span>
          </blockquote>
        </figure>
      )}
      {calculation && <Working calculation={calculation} onOpen={onOpen} />}
      {decisionMissing && (
        <p data-testid="claim-court-decision" className="citation-card-note">
          {t("claim.courtDecision.body")}
        </p>
      )}
      {decisionUnchecked && (
        <p data-testid="claim-cited-case-unchecked" className="citation-card-note">
          {t("claim.citedCaseUnchecked")}
        </p>
      )}
      {decisionHeld && evidence && (
        <p data-testid="claim-court-decision-held" className="citation-card-note">
          {t("claim.courtDecision.held", {
            document: shownDocumentName(evidence, documents) ?? "",
            location: where ?? "",
          })}
        </p>
      )}
      {state === "unsupported" && !differs && !decisionMissing && (
        <p
          data-testid={ownWorkings ? "claim-own-workings" : "claim-nothing"}
          className="citation-card-note"
        >
          {ownWorkings
            ? t("claim.unsupported.ownWorkings")
            : searchedFolder
              ? t("claim.unsupported.inFolder", { folder: folderName })
              : t("claim.unsupported.anywhere")}
        </p>
      )}
      {webOffShown && !webSearchOn && (
        <p data-testid="claim-web-off" className="citation-card-note">
          {t("claim.webSearchOff")}{" "}
          <button
            type="button"
            data-testid="claim-web-off-settings"
            className="font-semibold text-accent underline-offset-2 hover:underline"
            onClick={onTurnOnWebSearch}
          >
            {t("claim.webSearchOff.action")}
          </button>
        </p>
      )}
      {reason && (
        <p data-testid="claim-reason" className="citation-card-note">
          {reason}
        </p>
      )}
      {check.excluded.map((left) => {
        const document =
          left.reason === "own-file" && documents.find((each) => each.id === left.documentId);
        return (
          document && (
            <CountItLine key={left.documentId} document={document} testId="claim-left-out">
              {t("evidence.excluded.ownFile", { name: left.documentName || document.name })}
            </CountItLine>
          )
        );
      })}
      <div className="citation-card-actions">
        {addable && (
          <button
            type="button"
            data-testid="claim-add-citation"
            className="citation-card-action citation-card-action--primary"
            onClick={onAdd}
          >
            {t("claim.addAsCitation")}
          </button>
        )}
        {evidence?.documentId && (
          <button
            type="button"
            data-testid="claim-open"
            className={`citation-card-action ${addable ? "" : "citation-card-action--primary"}`}
            onClick={() => {
              if (!evidence.documentId) return;
              openCited(
                { ...evidence, documentId: evidence.documentId },
                { check: "found", withQuote: true, label: where },
              );
              onOpen();
            }}
          >
            {t(`citation.open.${openKind(location)}`)}
          </button>
        )}
        {differs && (
          <button
            type="button"
            data-testid="claim-fix"
            className="citation-card-action"
            onClick={onFix}
          >
            {t("claim.fixSentence")}
          </button>
        )}
        {state === "unsupported" && searchedFolder && !decisionMissing && (
          <button
            type="button"
            data-testid="claim-search-all"
            className="citation-card-action citation-card-action--primary"
            onClick={() => onCheckAgain(true)}
          >
            {t("claim.searchAll")}
          </button>
        )}
        {/* Where the Documents say something different the card has Open the page and Fix the sentence already, so it offers no more ways to look. */}
        {state === "unsupported" && !differs && !decisionMissing && (
          <>
            <button
              type="button"
              data-testid="claim-find-sources"
              className="citation-card-action"
              onClick={onFindSources}
            >
              {t("claim.findSources")}
            </button>
            <button
              type="button"
              data-testid="claim-search-web"
              className="citation-card-action"
              onClick={() => (webSearchOn ? onSearchWeb() : setWebOffShown(true))}
            >
              {t("claim.searchWeb")}
            </button>
          </>
        )}
        {state === "unsupported" && (
          <button
            type="button"
            data-testid="claim-remove"
            className="citation-card-action"
            onClick={onRemove}
          >
            {t("claim.removeMark")}
          </button>
        )}
        {(state === "not-checked" || state === "needs-recheck") && (
          <button
            type="button"
            data-testid="claim-check-again"
            className="citation-card-action citation-card-action--primary"
            onClick={() => onCheckAgain()}
          >
            {t("claim.checkAgain")}
          </button>
        )}
      </div>
    </div>
  );
}
