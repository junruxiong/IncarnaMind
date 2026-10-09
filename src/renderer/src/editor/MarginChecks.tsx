import type { Transaction } from "@tiptap/pm/state";
import type { Editor } from "@tiptap/react";
import { type CSSProperties, useEffect, useRef, useState } from "react";
import type { CitationAttributes, CitationCheck } from "../../../core/api";
import { badgeMessage, citationState } from "../../../shared/citations";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { type NumberedCitation, numberedCitations, textBlockEdited } from "./citationNumbers";
import { CheckMarkIcon } from "./icons";

const MARK_HEIGHT = 18;
/** Between two marks stacked because their markers share a line (or nearly). */
const MARK_GAP = 4;
/** When an Answer finishes, its checks appear one after another, this far apart. */
const REVEAL_STEP_MS = 40;

interface Mark {
  /** The Citation's key (`numberCitations`): its Block and number. */
  key: string;
  number: number;
  check: CitationCheck;
  label: string;
  /** The marker in the text: the mark is aligned to it, and clicking the mark clicks it. */
  marker: HTMLElement;
  /** Whether the marker's card is open. */
  open: boolean;
  /** The vertical middle of the marker, from the top of the column. */
  middle: number;
  top: number;
  /** How long its entrance waits, if it is appearing because its Answer just finished. */
  delay: number | null;
}

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
      mark.marker === other.marker
    );
  });

/**
 * The column of Citation checks in the right margin, the Mind's signature:
 * for each Citation an 18px mark with its state's icon and colour and its
 * number, level with the line its marker is on, so the checks can be scanned
 * down the page. Marks whose markers share a line stack under each other.
 * Clicking one does what clicking its marker does: it opens the Citation's card.
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
export function MarginChecks({ editor }: { editor: Editor }) {
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
  /** Each Citation's check as last drawn: a change from "checking" is its Answer finishing. */
  const seen = useRef(new Map<string, CitationCheck>());
  /** The entrance delays of the marks that appeared that way, by key and check. */
  const reveals = useRef(new Map<string, number>());
  const remeasure = useRef<() => void>(() => undefined);

  useEffect(() => {
    let frame = 0;
    /** Whether the next frame measures every mark, rather than those `typedIn` may have moved. */
    let all = true;
    /** The text block typed in since the last frame, when nothing else changed (`textBlockEdited`). */
    let typedIn: HTMLElement | null = null;
    /** The text block typed in when the marks were last measured, and its height then. */
    let typed: { block: HTMLElement; height: number } | null = null;
    /** Each Citation's marker and where it was when last measured, by key (`only` keeps them). */
    let placed = new Map<string, Pick<Mark, "marker" | "open" | "middle">>();
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
      if (only && !only.querySelector(".citation-marker")) return;
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
      // Top to bottom; a mark that would overlap the one above goes under it.
      next.sort((a, b) => a.middle - b.middle);
      let bottom = Number.NEGATIVE_INFINITY;
      for (const mark of next) {
        mark.top = Math.max(Math.round(mark.middle - MARK_HEIGHT / 2), bottom + MARK_GAP);
        bottom = mark.top + MARK_HEIGHT;
      }
      revealed
        .sort((a, b) => a.top - b.top)
        .forEach((mark, index) => {
          reveals.current.set(`${mark.key}:${mark.check}`, index * REVEAL_STEP_MS);
        });
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
  }, [editor]);

  // The interface language, the Documents (one deleted can't be checked) and the text kept of
  // unlinked ones change marks too.
  useEffect(() => {
    inputs.current = { t, documents, kept };
    described.current = new WeakMap();
    remeasure.current();
  }, [t, documents, kept]);

  return (
    <div ref={layer} className="margin-checks" data-testid="margin-checks">
      {marks.map((mark) => (
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
      ))}
    </div>
  );
}
