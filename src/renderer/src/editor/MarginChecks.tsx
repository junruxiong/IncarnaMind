import type { Transaction } from "@tiptap/pm/state";
import type { Editor } from "@tiptap/react";
import { type CSSProperties, useEffect, useRef, useState } from "react";
import type * as Y from "yjs";
import type { CitationAttributes, CitationCheck } from "../../../core/api";
import { badgeMessage, citationState } from "../../../shared/citations";
import { useT } from "../i18n";
import { motionMode, subscribeMotion } from "../motion";
import { useAppStore } from "../store";
import { ClaimMark } from "./ClaimCheckCard";
import { type NumberedCitation, numberedCitations, textBlockEdited } from "./citationNumbers";
import { type ShownClaim, shownClaims } from "./claimChecks";
import { CheckMarkIcon, type ClaimMarkState } from "./icons";
import { peekCitation } from "./peek";

const MARK_HEIGHT = 18;
/** Between two marks stacked because their markers share a line (or nearly). */
const MARK_GAP = 4;
/** When an Answer finishes, its checks appear one after another, this far apart. */
const REVEAL_STEP_MS = 40;

interface Placed {
  /** A Citation's key (`numberCitations`), its Block and number; or "claim:" and a sentence check's id. */
  key: string;
  number: number;
  label: string;
  /** Whether the marker's card is open. */
  open: boolean;
  /** The vertical middle of the marker, from the top of the column. */
  middle: number;
  top: number;
  /** How long its entrance waits, if it is appearing because its Answer just finished. */
  delay: number | null;
}

type Mark =
  | (Placed & {
      kind: "citation";
      check: CitationCheck;
      /** The marker in the text: the mark is aligned to it, and clicking the mark clicks it. */
      marker: HTMLElement;
    })
  | (Placed & {
      kind: "claim";
      check: ClaimMarkState;
      /** A sentence's check (#96): aligned to the sentence's first line, with a card of its own. */
      claim: ShownClaim;
    });

/** What a mark is for: its Citation's marker, or its sentence's check as it is now. */
const subjectOf = (mark: Mark) => (mark.kind === "citation" ? mark.marker : mark.claim.check);

/** A Citation's check and its mark's label, for its number. */
interface Description {
  number: number;
  check: CitationCheck;
  label: string;
}

/**
 * Whether a change to the editor's DOM may move marks: not one in the text
 * block being typed in (`measure` checks that block), to a node removed since
 * (its removal is a change of its own), or an attribute set to the value it
 * had (the editor sets its own at every change).
 */
function mayMoveMarks(record: MutationRecord, typedIn: Node | null): boolean {
  const { target } = record;
  if (!target.isConnected || typedIn?.contains(target)) return false;
  return (
    record.type !== "attributes" ||
    record.oldValue !== (target as Element).getAttribute(record.attributeName ?? "")
  );
}

const sameMarks = (a: readonly Mark[], b: readonly Mark[]) =>
  a.length === b.length &&
  a.every((mark, index) => {
    const other = b[index];
    return (
      other !== undefined &&
      mark.key === other.key &&
      mark.check === other.check &&
      mark.top === other.top &&
      mark.label === other.label &&
      mark.open === other.open &&
      mark.delay === other.delay &&
      subjectOf(mark) === subjectOf(other)
    );
  });

/**
 * The column of Citation checks in the right margin, the Mind's signature:
 * for each Citation an 18px mark with its state's icon and colour and its
 * number, level with the line its marker is on, so the checks can be scanned
 * down the page. Marks whose markers share a line stack under each other.
 * Clicking one does what clicking its marker does: it opens the Citation's card.
 * A sentence the User had checked (#96) has its own mark, level with its
 * first line (see `ClaimMark`).
 *
 * It is an overlay beside the editor that measures the markers: again after
 * every change to the Mind, every change to the editor's DOM (Answers stream
 * in, formulas render), every resize and every font load, at most once a
 * frame. Typing in a text block with no Citation in it, most of what happens,
 * is the exception: it moves no mark unless the block grows or shrinks, so
 * only the block is measured then. When an Answer
 * finishes, its checks appear top to bottom, 40ms apart (not with reduced
 * motion: see styles.css). Under 900px the column folds away and each marker
 * shows its check instead.
 */
export function MarginChecks({
  editor,
  mind = null,
}: {
  editor: Editor;
  /** The Mind and its document, whose sentence checks have marks too: a card checks again there. */
  mind?: { id: string; doc: Y.Doc } | null;
}) {
  const t = useT();
  const documents = useAppStore((state) =>
    state.status.kind === "ready" ? state.documents : null,
  );
  const kept = useAppStore((state) => state.keptCitationTexts);
  const layer = useRef<HTMLDivElement>(null);
  const [marks, setMarks] = useState<readonly Mark[]>([]);
  const inputs = useRef({ t, documents, kept });
  /**
   * Each Citation's check and label, by its attributes (an edit elsewhere keeps
   * them), until the words, the Documents or the kept text change: worked out
   * once, not at every measure.
   */
  const described = useRef(new WeakMap<CitationAttributes, Description>());
  /** Each mark's state as last drawn: a change from "checking" is its Answer, or its check, finishing. */
  const seen = useRef(new Map<string, CitationCheck | ClaimMarkState>());
  /** The entrance delays of the marks that appeared that way, by key and check. */
  const reveals = useRef(new Map<string, number>());
  const remeasure = useRef<() => void>(() => undefined);
  const withClaims = mind !== null;

  useEffect(() => {
    let frame = 0;
    /** Whether the next frame measures every mark, rather than those `typedIn` may have moved. */
    let all = true;
    /** The text block typed in since the last frame, when nothing else changed (`textBlockEdited`). */
    let typedIn: HTMLElement | null = null;
    /** The text block typed in when the marks were last measured, and its height then. */
    let typed: { block: HTMLElement; height: number } | null = null;
    /** Each Citation's marker and where it was when last measured, by key (`only` keeps them). */
    let placed = new Map<string, { marker: HTMLElement; open: boolean; middle: number }>();
    const describe = ({ attributes, number }: NumberedCitation): Description => {
      const known = described.current.get(attributes);
      if (known?.number === number) return known;
      const { t: translate, documents: live, kept: keptTexts } = inputs.current;
      const state = citationState(attributes, live, keptTexts);
      const badge = badgeMessage(state, attributes, translate);
      const description = {
        number,
        check: state.check,
        label: translate("citation.mark.label", {
          number,
          badge: translate(badge.key, badge.params),
        }),
      };
      described.current.set(attributes, description);
      return description;
    };
    const measure = () => {
      frame = 0;
      const column = layer.current;
      if (!column || editor.isDestroyed) return;
      const block = typedIn;
      typedIn = null;
      const height = block?.getBoundingClientRect().height ?? 0;
      // Typing that keeps its block's height moves only the marks in that block, if any.
      const only =
        !all && block && typed?.block === block && typed.height === height ? block : null;
      if (only && !only.querySelector(".citation-marker, [data-claim-check]")) return;
      all = false;
      typed = block ? { block, height } : null;
      const origin = column.getBoundingClientRect().top;
      const before = placed;
      placed = new Map();
      const next: Mark[] = [];
      const revealed: Mark[] = [];
      for (const citation of numberedCitations(editor.state)) {
        let place = only ? before.get(citation.key) : undefined;
        if (!place || only?.contains(place.marker)) {
          const dom = editor.view.nodeDOM(citation.pos);
          const marker =
            dom instanceof HTMLElement ? dom.querySelector<HTMLElement>(".citation-marker") : null;
          if (!marker) continue;
          const rect = marker.getBoundingClientRect();
          if (rect.height === 0) continue;
          place = {
            marker,
            open: marker.getAttribute("aria-expanded") === "true",
            middle: rect.top + rect.height / 2 - origin,
          };
        }
        placed.set(citation.key, place);
        const { check, label } = describe(citation);
        const mark: Mark = {
          kind: "citation",
          key: citation.key,
          number: citation.number,
          check,
          label,
          ...place,
          top: 0,
          delay: null,
        };
        if (seen.current.get(mark.key) === "checking" && mark.check !== "checking") {
          revealed.push(mark);
        }
        seen.current.set(mark.key, mark.check);
        next.push(mark);
      }
      const translate = inputs.current.t;
      for (const claim of withClaims ? shownClaims(editor.state) : []) {
        let coords: { top: number; bottom: number };
        try {
          coords = editor.view.coordsAtPos(claim.from, 1);
        } catch {
          continue;
        }
        const state = claim.shown.state;
        const mark: Mark = {
          kind: "claim",
          key: `claim:${claim.check.id}`,
          number: 0,
          check: state,
          label: translate("claim.mark.label", { state: translate(`claim.state.${state}`) }),
          claim,
          open: false,
          middle: (coords.top + coords.bottom) / 2 - origin,
          top: 0,
          delay: null,
        };
        if (seen.current.get(mark.key) === "checking" && state !== "checking") revealed.push(mark);
        seen.current.set(mark.key, state);
        next.push(mark);
      }
      // Top to bottom; a mark that would overlap the one above goes under it.
      next.sort((a, b) => a.middle - b.middle);
      let bottom = Number.NEGATIVE_INFINITY;
      for (const mark of next) {
        mark.top = Math.max(Math.round(mark.middle - MARK_HEIGHT / 2), bottom + MARK_GAP);
        bottom = mark.top + MARK_HEIGHT;
      }
      // Only with motion on: otherwise the marks are there at once, with no cascade to end.
      if (motionMode() === "on") {
        revealed
          .sort((a, b) => a.top - b.top)
          .forEach((mark, index) => {
            reveals.current.set(`${mark.key}:${mark.check}`, index * REVEAL_STEP_MS);
          });
      }
      for (const mark of next)
        mark.delay = reveals.current.get(`${mark.key}:${mark.check}`) ?? null;
      setMarks((drawn) => (sameMarks(drawn, next) ? drawn : next));
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };
    const request = () => {
      all = true;
      schedule();
    };
    remeasure.current = request;
    request();

    const onTransaction = ({ transaction }: { transaction: Transaction }) => {
      const at = textBlockEdited(transaction);
      const block = at === null ? null : editor.view.nodeDOM(at);
      if (!(block instanceof HTMLElement) || (typedIn && typedIn !== block)) {
        request();
        return;
      }
      typedIn = block;
      schedule();
    };
    editor.on("transaction", onTransaction);
    const resizes = new ResizeObserver(request);
    resizes.observe(editor.view.dom);
    // React draws node views after the transaction; Answers stream in; KaTeX renders.
    const mutations = new MutationObserver((records) => {
      if (records.some((record) => mayMoveMarks(record, typedIn))) request();
    });
    mutations.observe(editor.view.dom, {
      subtree: true,
      childList: true,
      characterData: true,
      attributes: true,
      attributeOldValue: true,
      attributeFilter: ["aria-expanded", "data-check", "data-number", "class", "style"],
    });
    document.fonts.addEventListener("loadingdone", request);
    void document.fonts.ready.then(request);
    return () => {
      cancelAnimationFrame(frame);
      remeasure.current = () => undefined;
      editor.off("transaction", onTransaction);
      resizes.disconnect();
      mutations.disconnect();
      document.fonts.removeEventListener("loadingdone", request);
    };
  }, [editor, withClaims]);

  // The interface language, the Documents (one deleted can't be checked) and the text kept of
  // unlinked ones change marks too.
  useEffect(() => {
    inputs.current = { t, documents, kept };
    described.current = new WeakMap();
    remeasure.current();
  }, [t, documents, kept]);

  // Motion turned to Reduced or Off stops the cascade (styles.css), so it never ends on its own:
  // its marks are simply there, and can't play it later if motion comes back on.
  useEffect(
    () =>
      subscribeMotion(() => {
        if (motionMode() === "on") return;
        reveals.current.clear();
        setMarks((drawn) =>
          drawn.some((mark) => mark.delay !== null)
            ? drawn.map((mark) => (mark.delay === null ? mark : { ...mark, delay: null }))
            : drawn,
        );
      }),
    [],
  );

  return (
    <div ref={layer} className="margin-checks" data-testid="margin-checks">
      {marks.map((mark) =>
        mark.kind === "claim" ? (
          mind && (
            <ClaimMark
              key={`${mark.key}:${mark.check}`}
              claim={mark.claim}
              editor={editor}
              mindId={mind.id}
              doc={mind.doc}
              label={mark.label}
              reveal={mark.delay !== null}
              style={{ top: mark.top, "--reveal-delay": `${mark.delay ?? 0}ms` } as CSSProperties}
              onRevealed={() => {
                reveals.current.delete(`${mark.key}:${mark.check}`);
                setMarks((drawn) =>
                  drawn.map((each) => (each === mark ? { ...each, delay: null } : each)),
                );
              }}
            />
          )
        ) : (
          <button
            // Keyed by the check too, so a mark whose check just came in is new, and enters.
            key={`${mark.key}:${mark.check}`}
            type="button"
            data-testid="margin-check"
            data-check={mark.check}
            data-number={mark.number}
            data-open={mark.open || undefined}
            className={`margin-check ${mark.delay !== null ? "margin-check--reveal" : ""}`}
            style={{ top: mark.top, "--reveal-delay": `${mark.delay ?? 0}ms` } as CSSProperties}
            aria-label={mark.label}
            title={mark.label}
            // Keep the editor's selection and focus where they are.
            onMouseDown={(event) => event.preventDefault()}
            // Its Citation's card opens on hover and focus, as the marker's own does.
            onPointerEnter={() => peekCitation(mark.marker, "mark-pointer", true)}
            onPointerLeave={() => peekCitation(mark.marker, "mark-pointer", false)}
            onFocus={() => peekCitation(mark.marker, "mark-focus", true)}
            onBlur={() => peekCitation(mark.marker, "mark-focus", false)}
            onClick={() => {
              if (mark.marker.isConnected) mark.marker.click();
            }}
            // Entered once: it mustn't enter again when the column shows again after folding away.
            onAnimationEnd={(event) => {
              if (event.target !== event.currentTarget || mark.delay === null) return;
              reveals.current.delete(`${mark.key}:${mark.check}`);
              setMarks((drawn) =>
                drawn.map((each) => (each === mark ? { ...each, delay: null } : each)),
              );
            }}
          >
            <CheckMarkIcon check={mark.check} className="margin-check-icon" />
            {mark.number}
          </button>
        ),
      )}
    </div>
  );
}
