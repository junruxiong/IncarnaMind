import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Document } from "../../../core/api";
import {
  extractPptx,
  imageDataUrl,
  type PptxResult,
  type SlideOutline,
} from "../../../core/documents/formats/pptx";
import { findQuoteInPieces, type TextRange } from "../../../shared/quoteMatch";
import { useT } from "../i18n";
import type { ViewerTarget } from "../store";
import { FileStates } from "./FileStates";
import { HighlightedText } from "./highlightedText";
import { CitationMark, marginRoom, markTopFor, quoteMarkOf, quoteTone } from "./quoteMark";
import { OutlineButton, type OutlineEntry, UnitOutline } from "./UnitOutline";
import { useDocumentFile } from "./useDocumentFile";
import { ViewerHeader } from "./ViewerHeader";

interface Loaded {
  bytes: Uint8Array;
  deck: PptxResult;
}

const read = async (bytes: Uint8Array): Promise<Loaded> => ({
  bytes,
  deck: await extractPptx(bytes),
});

/**
 * A PowerPoint deck (ADR-0011), as a slide-by-slide outline read with the
 * code that indexed it: each slide's number and title, its text, tables and
 * charts' figures, its images (as `data:` URLs, which the app's
 * Content-Security-Policy allows), then its speaker notes. Opened at a
 * Citation, it goes to the cited slide and highlights the quote there, in the
 * slide's text or its notes, with the Citation's mark beside it.
 */
export function SlidesView({ document, target }: { document: Document; target: ViewerTarget }) {
  const loaded = useDocumentFile(document.id, read);
  return (
    <FileStates
      loaded={loaded}
      quote={target.quote}
      ready={(value) => <Slides loaded={value} target={target} />}
    />
  );
}

/** One string a slide shows, with an id the highlight finds it by. */
interface Shown {
  id: string;
  text: string;
}

/** What a slide shows, in order: the strings a quote is looked for in. */
function shownStrings(slide: SlideOutline): Shown[] {
  const shown: Shown[] = [];
  if (slide.title) shown.push({ id: "title", text: slide.title });
  for (const [block, lines] of slide.blocks.entries()) {
    for (const [line, text] of lines.entries()) shown.push({ id: `block${block}.${line}`, text });
  }
  for (const [table, rows] of slide.tables.entries()) {
    for (const [row, cells] of rows.entries()) {
      for (const [cell, text] of cells.entries()) {
        shown.push({ id: `table${table}.${row}.${cell}`, text });
      }
    }
  }
  for (const [index, chart] of slide.charts.entries()) {
    for (const [line, text] of chart.split("\n").entries()) {
      shown.push({ id: `chart${index}.${line}`, text });
    }
  }
  for (const [line, text] of slide.notes.entries()) shown.push({ id: `notes${line}`, text });
  return shown;
}

/** Where a quote is in a slide: its highlighted parts, by the id of the string they are in. */
function findInSlide(slide: SlideOutline, quote: string): Map<string, TextRange[]> | null {
  const shown = shownStrings(slide).filter((each) => each.text);
  const found = findQuoteInPieces(
    shown.map((each) => ({ text: each.text, breakAfter: true })),
    quote,
  );
  if (!found) return null;
  const byId = new Map<string, TextRange[]>();
  for (const part of found) {
    const id = (shown[part.piece] as Shown).id;
    byId.set(id, [...(byId.get(id) ?? []), { start: part.start, end: part.end }]);
  }
  return byId;
}

/** The title an outline shows for a slide: its title, or else its first line of text. */
const outlineTitle = (slide: SlideOutline) =>
  slide.title ?? slide.blocks[0]?.[0] ?? slide.tables[0]?.[0]?.[0] ?? "";

function Slides({ loaded, target }: { loaded: Loaded; target: ViewerTarget }) {
  const t = useT();
  const { slides } = loaded.deck;
  const scroller = useRef<HTMLDivElement>(null);
  const slideElements = useRef(new Map<number, HTMLElement>());
  const [outlineOpen, setOutlineOpen] = useState(false);
  const [current, setCurrent] = useState<number | null>(null);
  const [images, setImages] = useState<ReadonlyMap<string, string | null>>(new Map());
  const appliedRequest = useRef<number | null>(null);

  // The quote, in the cited slide (or the next, for two), else in any slide.
  const highlight = useMemo(() => {
    const quote = target.quote;
    if (!quote) return null;
    const from = target.pageFrom ?? 1;
    const to = target.pageTo ?? from;
    const cited = slides.filter((slide) => slide.number >= from && slide.number <= to);
    const ordered = [...cited, ...slides.filter((slide) => !cited.includes(slide))];
    for (const slide of ordered) {
      const ranges = findInSlide(slide, quote);
      if (ranges) return { slide: slide.number, ranges };
    }
    return null;
  }, [slides, target.quote, target.pageFrom, target.pageTo]);
  const mark = useMemo(() => quoteMarkOf(target.citation), [target.citation]);

  // Images, read from the package as data: URLs once the slides are shown.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const urls = new Map<string, string | null>();
      for (const slide of slides) {
        for (const image of slide.images) {
          if (urls.has(image.part)) continue;
          urls.set(image.part, await imageDataUrl(loaded.bytes, image.part));
          if (cancelled) return;
        }
      }
      setImages(urls);
    })();
    return () => {
      cancelled = true;
    };
  }, [loaded.bytes, slides]);

  // Each open request goes to its quote, else to its slide, else the top.
  useLayoutEffect(() => {
    const container = scroller.current;
    if (!container || appliedRequest.current === target.request) return;
    appliedRequest.current = target.request;
    const first = container.querySelector<HTMLElement>("[data-quote-highlight]");
    const slide = slideElements.current.get(highlight?.slide ?? target.pageFrom ?? 0);
    const anchor = first ?? slide;
    if (!anchor) {
      container.scrollTop = 0;
      return;
    }
    const top =
      anchor.getBoundingClientRect().top -
      container.getBoundingClientRect().top +
      container.scrollTop;
    container.scrollTop = Math.max(0, top - (first ? container.clientHeight / 3 : 24));
    setCurrent(highlight?.slide ?? target.pageFrom ?? null);
  }, [highlight, target.request, target.pageFrom]);

  const onScroll = () => {
    const container = scroller.current;
    if (!container || !outlineOpen) return;
    const top = container.getBoundingClientRect().top + 48;
    let number: number | null = null;
    for (const [slide, element] of slideElements.current) {
      if (element.getBoundingClientRect().top <= top && (number === null || slide > number)) {
        number = slide;
      }
    }
    setCurrent(number ?? slides[0]?.number ?? null);
  };

  const outline: OutlineEntry[] = slides.map((slide) => ({
    key: slide.number,
    title: outlineTitle(slide) || t("viewer.slide.untitled"),
    depth: 0,
    prefix: String(slide.number),
  }));
  const goTo = (number: number) => {
    const container = scroller.current;
    const element = slideElements.current.get(number);
    if (!container || !element) return;
    container.scrollTop =
      element.getBoundingClientRect().top -
      container.getBoundingClientRect().top +
      container.scrollTop -
      24;
    setCurrent(number);
  };

  return (
    <>
      <ViewerHeader
        leading={
          slides.length > 0 && (
            <OutlineButton open={outlineOpen} onToggle={() => setOutlineOpen((open) => !open)} />
          )
        }
      />
      <div className="flex min-h-0 flex-1">
        {outlineOpen && <UnitOutline entries={outline} current={current} onGo={goTo} />}
        <div
          ref={scroller}
          data-testid="viewer-slides"
          data-quote-tone={quoteTone(target.citation)}
          onScroll={onScroll}
          className="viewer-backdrop viewer-slides-backdrop min-w-0 flex-1 overflow-auto select-text"
        >
          {slides.map((slide) => (
            <SlideSheet
              key={slide.number}
              slide={slide}
              images={images}
              ranges={highlight?.slide === slide.number ? highlight.ranges : null}
              mark={highlight?.slide === slide.number ? mark : null}
              setElement={(element) => {
                if (element) slideElements.current.set(slide.number, element);
                else slideElements.current.delete(slide.number);
              }}
            />
          ))}
        </div>
      </div>
    </>
  );
}

function SlideSheet({
  slide,
  images,
  ranges,
  mark,
  setElement,
}: {
  slide: SlideOutline;
  images: ReadonlyMap<string, string | null>;
  ranges: Map<string, TextRange[]> | null;
  mark: ReturnType<typeof quoteMarkOf>;
  setElement(element: HTMLElement | null): void;
}) {
  const t = useT();
  const column = useRef<HTMLDivElement>(null);
  const [markTop, setMarkTop] = useState<number | null>(null);
  const [markRoom, setMarkRoom] = useState(0);
  const text = (id: string, value: string) => (
    <HighlightedText text={value} ranges={ranges?.get(id)} />
  );

  // The mark sits beside the quote's first line, wherever the slide wraps it.
  useLayoutEffect(() => {
    const element = column.current;
    if (!element || !mark || !ranges) {
      setMarkTop(null);
      return;
    }
    const place = () => {
      const first = element.querySelector("[data-quote-highlight]");
      setMarkTop(first ? markTopFor(first, element) : null);
      setMarkRoom(marginRoom(element));
    };
    place();
    const observer = new ResizeObserver(place);
    observer.observe(element);
    return () => observer.disconnect();
  }, [mark, ranges]);

  return (
    <article
      ref={setElement}
      data-testid="viewer-slide"
      data-slide={slide.number}
      aria-label={t("viewer.slide.label", { number: slide.number })}
      className={`viewer-page viewer-slide ${slide.hidden ? "viewer-slide--hidden" : ""}`}
    >
      <div ref={column} className="relative">
        <p className="viewer-slide-number">
          {t("viewer.slide.label", { number: slide.number })}
          {slide.hidden && <span> · {t("viewer.slide.hidden")}</span>}
        </p>
        {slide.title && <h2 className="viewer-slide-title">{text("title", slide.title)}</h2>}
        <div className="viewer-slide-body">
          {slide.blocks.map((lines, block) =>
            lines.length > 1 ? (
              // biome-ignore lint/suspicious/noArrayIndexKey: a slide's blocks never reorder
              <ul key={block}>
                {lines.map((line, index) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: neither do their lines
                  <li key={index}>{text(`block${block}.${index}`, line)}</li>
                ))}
              </ul>
            ) : (
              // biome-ignore lint/suspicious/noArrayIndexKey: a slide's blocks never reorder
              <p key={block}>{text(`block${block}.0`, lines[0] ?? "")}</p>
            ),
          )}
          {slide.tables.map((rows, table) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: a slide's tables never reorder
            <table key={table} className="viewer-slide-table">
              <tbody>
                {rows.map((cells, row) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: nor their rows
                  <tr key={row} className={row === 0 ? "viewer-slide-row--head" : undefined}>
                    {cells.map((cell, index) => (
                      // biome-ignore lint/suspicious/noArrayIndexKey: nor their cells
                      <td key={index} className="viewer-slide-cell">
                        {text(`table${table}.${row}.${index}`, cell)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          ))}
          {slide.charts.map((chart, index) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: a slide's charts never reorder
            <figure key={index} className="viewer-slide-chart">
              <figcaption>{t("viewer.slide.chart")}</figcaption>
              {chart.split("\n").map((line, at) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: nor their lines
                <p key={at}>{text(`chart${index}.${at}`, line)}</p>
              ))}
            </figure>
          ))}
          {slide.images.map((image, index) => {
            const url = images.get(image.part);
            return url ? (
              // biome-ignore lint/suspicious/noArrayIndexKey: a slide's images never reorder
              <img key={index} src={url} alt={image.alt} className="viewer-slide-image" />
            ) : null;
          })}
        </div>
        {slide.notes.length > 0 && (
          <section data-testid="viewer-slide-notes" className="viewer-slide-notes">
            <h3>{t("viewer.slide.notes")}</h3>
            {slide.notes.map((line, index) => (
              // biome-ignore lint/suspicious/noArrayIndexKey: notes never reorder
              <p key={index}>{text(`notes${index}`, line)}</p>
            ))}
          </section>
        )}
        {mark && markTop !== null && (
          <CitationMark
            mark={mark}
            room={markRoom}
            style={{ top: markTop, left: "calc(100% + 12px)" }}
          />
        )}
      </div>
    </article>
  );
}
