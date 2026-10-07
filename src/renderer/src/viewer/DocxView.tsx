import { renderAsync } from "docx-preview";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Document } from "../../../core/api";
import { type DocxResult, extractDocx } from "../../../core/documents/formats/docx";
import type { ViewerTarget } from "../store";
import { markQuote, textNodes, unmarkQuote } from "./domQuote";
import { FileStates } from "./FileStates";
import { CitationMark, quoteMarkOf, quoteTone } from "./quoteMark";
import { OutlineButton, type OutlineEntry, UnitOutline } from "./UnitOutline";
import { useDocumentFile } from "./useDocumentFile";
import { ViewerHeader } from "./ViewerHeader";

/** docx-preview's class prefix: a paragraph in style "Heading2" is `p.docx_heading2`. */
const CLASS_NAME = "docx";

/** A style id as docx-preview turns it into a class name. */
const styleClass = (id: string) =>
  `${CLASS_NAME}_${id.replace(/[ .]+/g, "-").replace(/[&]+/g, "and").toLowerCase()}`;

/** Space around the pages, and between a quote's line and its mark, in CSS pixels. */
const PADDING = 24;
const MARK_GAP = 12;
/** Room the mark, without its label, keeps from the page's right edge. */
const MARK_ROOM = 40;

interface Loaded {
  bytes: Uint8Array;
  docx: DocxResult;
}

/**
 * The zoom that fits the drawn pages to the width there is beside the
 * outline, if it is open, never larger than they are.
 */
function fitZoom(outer: HTMLElement | null, body: HTMLElement): number {
  const page = body.querySelector<HTMLElement>(`section.${CLASS_NAME}`);
  if (!outer || !page || page.offsetWidth === 0) return 1;
  return Math.min(1, Math.max(0.05, (outer.clientWidth - 2 * PADDING) / page.offsetWidth));
}

const read = async (bytes: Uint8Array): Promise<Loaded> => ({
  bytes,
  docx: await extractDocx(bytes),
});

/**
 * A Word Document (ADR-0011): drawn page-like by docx-preview (Apache-2.0),
 * its pages split only where the file breaks them, so they are not Word's
 * pages: a Citation points at the section a quote sits under, which the
 * outline lists. Opened at a Citation, it goes to the cited section and
 * highlights the quote in it (anywhere in the Document if it isn't there),
 * with the Citation's mark in the page's margin beside it. Images come as
 * `data:` URLs, so the app's Content-Security-Policy stays as it is.
 */
export function DocxView({ document, target }: { document: Document; target: ViewerTarget }) {
  const loaded = useDocumentFile(document.id, read);
  return (
    <FileStates
      loaded={loaded}
      quote={target.quote}
      ready={(value) => <DocxPages loaded={value} target={target} />}
    />
  );
}

interface MarkPlace {
  top: number;
  left: number;
  /** The page's margin past the text, which the mark's label must fit in. */
  room: number;
}

function DocxPages({ loaded, target }: { loaded: Loaded; target: ViewerTarget }) {
  const scroller = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const host = useRef<HTMLDivElement>(null);
  const styles = useRef<HTMLDivElement>(null);
  const [rendered, setRendered] = useState<"no" | "yes" | "failed">("no");
  const [failure, setFailure] = useState("");
  const [zoom, setZoom] = useState(1);
  const [outlineOpen, setOutlineOpen] = useState(false);
  const [current, setCurrent] = useState<number | null>(null);
  const [marks, setMarks] = useState<HTMLElement[]>([]);
  const [place, setPlace] = useState<MarkPlace | null>(null);
  const mark = useMemo(() => quoteMarkOf(target.citation), [target.citation]);
  const { docx } = loaded;

  // Draw the pages once.
  useEffect(() => {
    const body = host.current;
    const style = styles.current;
    if (!body || !style) return;
    let cancelled = false;
    renderAsync(loaded.bytes, body, style, {
      className: CLASS_NAME,
      inWrapper: true,
      breakPages: true,
      ignoreLastRenderedPageBreak: false,
      experimental: true,
      // Images and fonts as data: URLs: the app's policy allows those, not blob: ones.
      useBase64URL: true,
      renderChanges: false,
      renderComments: false,
      renderAltChunks: false,
      trimXmlDeclaration: true,
    }).then(
      () => {
        if (cancelled) return;
        // Links open in the User's browser (the main process sends new windows there).
        for (const link of body.querySelectorAll<HTMLAnchorElement>("a[href]")) {
          if (/^(?:https?:|mailto:)/i.test(link.href)) {
            link.target = "_blank";
            link.rel = "noreferrer";
          } else link.removeAttribute("href");
        }
        // Fitted before anything is placed or scrolled to.
        setZoom(fitZoom(scroller.current, body));
        setRendered("yes");
      },
      (error: unknown) => {
        if (cancelled) return;
        setFailure(error instanceof Error ? error.message : String(error));
        setRendered("failed");
      },
    );
    return () => {
      cancelled = true;
      body.replaceChildren();
      style.replaceChildren();
    };
  }, [loaded.bytes]);

  /** The heading elements docx-preview drew, in order: those with text, as the outline lists them. */
  const headingElements = useCallback((): HTMLElement[] => {
    const body = host.current;
    if (!body || docx.headingStyles.length === 0) return [];
    const selector = docx.headingStyles.map((id) => `p.${CSS.escape(styleClass(id))}`).join(",");
    return [...body.querySelectorAll<HTMLElement>(selector)].filter(
      (element) => (element.textContent ?? "").trim() !== "",
    );
  }, [docx]);

  // The pages keep fitting the viewer's width as it is resized.
  useLayoutEffect(() => {
    const outer = scroller.current;
    const body = host.current;
    if (!outer || !body || rendered !== "yes") return;
    const observer = new ResizeObserver(() => setZoom(fitZoom(outer, body)));
    observer.observe(outer);
    return () => observer.disconnect();
  }, [rendered]);

  /**
   * The cited section, as a range of the drawn document, from its heading to
   * the next one, with its heading (null for the text before any heading).
   */
  const citedSection = useCallback((): { range: Range; heading: HTMLElement | null } | null => {
    const body = host.current;
    const unit = target.pageFrom;
    if (!body || unit === undefined) return null;
    const headings = headingElements();
    if (headings.length !== docx.headings.length) return null; // drawn otherwise than read
    const cited = docx.units.find((each) => each.page === unit);
    if (!cited || cited.label?.notes) return null;
    // The heading the cited Unit is under, and the one after the Units cited.
    const last = target.pageTo ?? unit;
    let start = -1;
    let end = -1;
    docx.headings.forEach((heading, index) => {
      if (heading.unit <= unit) start = index;
      if (end < 0 && heading.unit > last) end = index;
    });
    const range = body.ownerDocument.createRange();
    const heading = headings[start] ?? null;
    if (heading) range.setStartBefore(heading);
    else range.setStart(body, 0);
    const next = headings[end];
    if (next) range.setEndBefore(next);
    else range.setEnd(body, body.childNodes.length);
    return { range, heading };
  }, [docx, headingElements, target.pageFrom, target.pageTo]);

  // Each open request highlights its quote: in the cited section first, then anywhere.
  // biome-ignore lint/correctness/useExhaustiveDependencies: each open request goes there again
  useEffect(() => {
    const body = host.current;
    if (!body || rendered !== "yes") return;
    unmarkQuote(body);
    const quote = target.quote;
    const section = citedSection();
    let found: HTMLElement[] | null = null;
    if (quote) {
      const find = () =>
        (section && markQuote(textNodes(body, section.range), quote)) ??
        markQuote(textNodes(body), quote);
      found = find();
      if (!found) {
        // A footnote's number in the text isn't in the text a quote is checked
        // against: read on across numbers set as superscripts, and try again.
        const numbers = [...body.querySelectorAll<HTMLElement>("sup")].filter((sup) =>
          /^\s*\d{1,3}\s*$/.test(sup.textContent ?? ""),
        );
        for (const sup of numbers) sup.dataset.quoteSkip = "";
        found = find();
        for (const sup of numbers) delete sup.dataset.quoteSkip;
      }
    }
    setMarks(found ?? []);
    if (section?.heading) setCurrent(headingElements().indexOf(section.heading));
    // Go to the quote, or else to the cited section, or the top.
    const container = scroller.current;
    if (!container) return;
    const anchor = found?.[0] ?? section?.heading ?? null;
    if (anchor) {
      const top =
        anchor.getBoundingClientRect().top -
        container.getBoundingClientRect().top +
        container.scrollTop;
      container.scrollTop = Math.max(0, top - (found ? container.clientHeight / 3 : PADDING));
    } else container.scrollTop = 0;
  }, [rendered, citedSection, target.request, target.quote]);

  // The Citation's mark: in the page's right margin, level with the quote's first line.
  // biome-ignore lint/correctness/useExhaustiveDependencies: placed again when the pages zoom
  useLayoutEffect(() => {
    const first = marks[0];
    const box = content.current;
    if (!mark || !first || !box) {
      setPlace(null);
      return;
    }
    const placeMark = () => {
      const origin = box.getBoundingClientRect();
      const line = first.getClientRects()[0] ?? first.getBoundingClientRect();
      const page = first.closest(`section.${CLASS_NAME}`)?.getBoundingClientRect();
      const paragraph = first.closest("p, td, li")?.getBoundingClientRect() ?? line;
      const pageRight = page?.right ?? paragraph.right + MARK_GAP + MARK_ROOM;
      setPlace({
        top: line.top - origin.top + line.height / 2 - 9,
        left: Math.min(paragraph.right + MARK_GAP, pageRight - MARK_ROOM) - origin.left,
        room: pageRight - paragraph.right - MARK_GAP - 6,
      });
    };
    placeMark();
    const observer = new ResizeObserver(placeMark);
    observer.observe(box);
    return () => observer.disconnect();
  }, [marks, mark, zoom]);

  // The outline's entry for the section in view.
  const onScroll = () => {
    const container = scroller.current;
    if (!container || !outlineOpen) return;
    const top = container.getBoundingClientRect().top + 48;
    let index: number | null = null;
    headingElements().forEach((element, at) => {
      if (element.getBoundingClientRect().top <= top) index = at;
    });
    setCurrent(index);
  };

  const outline: OutlineEntry[] = useMemo(
    () =>
      docx.headings.map((heading, index) => ({
        key: index,
        title: heading.text,
        depth: Math.min(heading.level - 1, 3),
      })),
    [docx],
  );
  const goTo = (index: number) => {
    const container = scroller.current;
    const element = headingElements()[index];
    if (!container || !element) return;
    container.scrollTop =
      element.getBoundingClientRect().top -
      container.getBoundingClientRect().top +
      container.scrollTop -
      PADDING;
    setCurrent(index);
  };

  return (
    <>
      <ViewerHeader
        leading={
          outline.length > 0 && (
            <OutlineButton open={outlineOpen} onToggle={() => setOutlineOpen((open) => !open)} />
          )
        }
      />
      <div className="flex min-h-0 flex-1">
        {outlineOpen && <UnitOutline entries={outline} current={current} onGo={goTo} />}
        <div
          ref={scroller}
          data-testid="viewer-docx"
          data-rendered={rendered}
          data-quote-tone={quoteTone(target.citation)}
          onScroll={onScroll}
          className="viewer-backdrop relative min-w-0 flex-1 overflow-auto"
        >
          {rendered === "failed" && (
            <p role="alert" className="px-6 py-8 text-center text-ui text-danger">
              {failure}
            </p>
          )}
          <div ref={content} className="relative">
            <div ref={styles} />
            <div ref={host} className="docx-view" style={{ zoom }} />
            {mark && place && (
              <CitationMark
                mark={mark}
                room={place.room}
                style={{ top: place.top, left: place.left }}
              />
            )}
          </div>
        </div>
      </div>
    </>
  );
}
