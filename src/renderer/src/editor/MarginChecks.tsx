import type { Editor } from "@tiptap/react";
import { type CSSProperties, useEffect, useRef, useState } from "react";
import type { CitationCheck } from "../../../core/api";
import { badgeMessage, citationState } from "../../../shared/citations";
import { useT } from "../i18n";
import { useAppStore } from "../store";
import { numberCitations } from "./citationNumbers";
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
 * in, formulas render), every resize and every font load. When an Answer
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
  /** Each Citation's check as last drawn: a change from "checking" is its Answer finishing. */
  const seen = useRef(new Map<string, CitationCheck>());
  /** The entrance delays of the marks that appeared that way, by key and check. */
  const reveals = useRef(new Map<string, number>());
  const remeasure = useRef<() => void>(() => undefined);

  useEffect(() => {
    let frame = 0;
    const measure = () => {
      frame = 0;
      const column = layer.current;
      if (!column || editor.isDestroyed) return;
      const origin = column.getBoundingClientRect().top;
      const { t: translate, documents: live, kept: keptTexts } = inputs.current;
      const next: Mark[] = [];
      const revealed: Mark[] = [];
      for (const citation of numberCitations(editor.state.doc)) {
        const dom = editor.view.nodeDOM(citation.pos);
        const marker =
          dom instanceof HTMLElement ? dom.querySelector<HTMLElement>(".citation-marker") : null;
        if (!marker) continue;
        const rect = marker.getBoundingClientRect();
        if (rect.height === 0) continue;
        const state = citationState(citation.attributes, live, keptTexts);
        const badge = badgeMessage(state, citation.attributes, translate);
        const mark: Mark = {
          key: citation.key,
          number: citation.number,
          check: state.check,
          label: translate("citation.mark.label", {
            number: citation.number,
            badge: translate(badge.key, badge.params),
          }),
          marker,
          open: marker.getAttribute("aria-expanded") === "true",
          middle: rect.top + rect.height / 2 - origin,
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
    const request = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };
    remeasure.current = request;
    request();

    editor.on("transaction", request);
    const resizes = new ResizeObserver(request);
    resizes.observe(editor.view.dom);
    // React draws node views after the transaction; Answers stream in; KaTeX renders.
    const mutations = new MutationObserver(request);
    mutations.observe(editor.view.dom, {
      subtree: true,
      childList: true,
      characterData: true,
      attributes: true,
      attributeFilter: ["aria-expanded", "data-check", "data-number", "class", "style"],
    });
    document.fonts.addEventListener("loadingdone", request);
    void document.fonts.ready.then(request);
    return () => {
      cancelAnimationFrame(frame);
      remeasure.current = () => undefined;
      editor.off("transaction", request);
      resizes.disconnect();
      mutations.disconnect();
      document.fonts.removeEventListener("loadingdone", request);
    };
  }, [editor]);

  // The interface language, the Documents (one deleted can't be checked) and the text kept of
  // unlinked ones change marks too.
  useEffect(() => {
    inputs.current = { t, documents, kept };
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
