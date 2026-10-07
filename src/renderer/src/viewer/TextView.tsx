import { type ReactNode, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Document } from "../../../core/api";
import { decodeText } from "../../../core/documents/decode";
import { lineUnits, markdownUnits, type SourceUnit } from "../../../core/documents/formats/text";
import { findQuote, type TextRange } from "../../../shared/quoteMatch";
import { useT } from "../i18n";
import type { ViewerTarget } from "../store";
import { loadDocumentBytes, MissingDocumentError } from "./documentBytes";
import { type Block, type Inline, parseMarkdown } from "./markdown";
import { CitationMark, marginRoom, markTopFor, quoteMarkOf, quoteTone } from "./quoteMark";
import { OutlineButton, type OutlineEntry, UnitOutline } from "./UnitOutline";
import { ViewerHeader } from "./ViewerHeader";
import { FileGone, ViewerMessage } from "./ViewerMessage";

type Loaded =
  | { kind: "loading" }
  | { kind: "ready"; text: string }
  | { kind: "missing" }
  | { kind: "failed"; message: string };

/** Nothing to highlight. */
const NO_HIGHLIGHT: readonly TextRange[] = [];

/**
 * A TXT or Markdown Document, readable in the viewer: set in the Mind's serif
 * on a page, like a PDF's. Opened at a Citation, it highlights the quote in
 * the cited section (Markdown) or lines (TXT), each part of a quote with an
 * ellipsis, and scrolls to it; a quote not found there is looked for in the
 * whole file. Without a quote, it opens at the cited Unit, or the top. A
 * Markdown file's headings form an outline.
 */
export function TextView({ document, target }: { document: Document; target: ViewerTarget }) {
  const t = useT();
  const [loaded, setLoaded] = useState<Loaded>({ kind: "loading" });
  const [outlineOpen, setOutlineOpen] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    loadDocumentBytes(document.id, controller.signal).then(
      (bytes) => {
        // Decoded the way processing decodes it, so quotes from its Passages match.
        const text = decodeText(bytes);
        setLoaded(text === null ? { kind: "failed", message: "" } : { kind: "ready", text });
      },
      (error: unknown) => {
        if (controller.signal.aborted) return;
        if (error instanceof MissingDocumentError) setLoaded({ kind: "missing" });
        else setLoaded({ kind: "failed", message: error instanceof Error ? error.message : "" });
      },
    );
    return () => controller.abort();
  }, [document.id]);

  const markdown = document.kind === "markdown";
  const units = useMemo(
    () =>
      loaded.kind !== "ready" ? [] : markdown ? markdownUnits(loaded.text) : lineUnits(loaded.text),
    [loaded, markdown],
  );
  const outline = useMemo(() => (markdown ? headingsOf(units) : []), [markdown, units]);

  return (
    <>
      <ViewerHeader
        openable={loaded.kind !== "missing"}
        leading={
          outline.length > 0 && (
            <OutlineButton open={outlineOpen} onToggle={() => setOutlineOpen((open) => !open)} />
          )
        }
      />
      {loaded.kind === "loading" ? (
        <ViewerMessage>{t("viewer.loading")}</ViewerMessage>
      ) : loaded.kind === "missing" ? (
        <FileGone quote={target.quote} />
      ) : loaded.kind === "failed" ? (
        <ViewerMessage tone="error">
          {loaded.message ? t("viewer.failed", { message: loaded.message }) : t("viewer.notText")}
        </ViewerMessage>
      ) : (
        <TextContent
          text={loaded.text}
          markdown={markdown}
          units={units}
          outline={outlineOpen ? outline : null}
          target={target}
        />
      )}
    </>
  );
}

/** A Markdown file's headings, for the outline: each section's, keyed by the heading's index. */
function headingsOf(units: readonly SourceUnit[]): OutlineEntry[] {
  const entries: OutlineEntry[] = [];
  for (const [unit, heading] of headingIndexes(units)) {
    if (heading < 0) continue;
    const path = unit.label?.path ?? [];
    entries.push({ key: heading, title: path.at(-1) ?? "", depth: Math.max(0, path.length - 1) });
  }
  return entries;
}

/**
 * The heading each section starts at, by its index among the file's headings
 * (-1 for the text before the first), for the Units that start a section.
 */
function headingIndexes(units: readonly SourceUnit[]): Map<SourceUnit, number> {
  const indexes = new Map<SourceUnit, number>();
  let heading = -1;
  for (const unit of units) {
    if (unit.label?.part) continue;
    if ((unit.label?.path ?? []).length > 0) heading++;
    indexes.set(unit, heading);
  }
  return indexes;
}

function TextContent({
  text,
  markdown,
  units,
  outline,
  target,
}: {
  text: string;
  markdown: boolean;
  units: readonly SourceUnit[];
  outline: readonly OutlineEntry[] | null;
  target: ViewerTarget;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const column = useRef<HTMLDivElement>(null);
  const appliedRequest = useRef<number | null>(null);
  const [current, setCurrent] = useState<number | null>(null);

  // The cited Units' span of the file, when it has those Units.
  const cited = useMemo(() => {
    if (target.pageFrom === undefined) return null;
    const to = target.pageTo ?? target.pageFrom;
    const span = units.filter((unit) => unit.page >= (target.pageFrom ?? 0) && unit.page <= to);
    const first = span[0];
    const last = span.at(-1);
    return first && last ? { start: first.start, end: last.end, first } : null;
  }, [units, target.pageFrom, target.pageTo]);

  // The quote: in the cited Units first, then anywhere in the file.
  const highlight = useMemo(() => {
    const quote = target.quote;
    if (!quote) return NO_HIGHLIGHT;
    if (cited) {
      const inside = findQuote(text.slice(cited.start, cited.end), quote);
      if (inside) {
        return inside.map((range) => ({
          start: range.start + cited.start,
          end: range.end + cited.start,
        }));
      }
    }
    return findQuote(text, quote) ?? NO_HIGHLIGHT;
  }, [text, target.quote, cited]);
  const blocks = useMemo(() => (markdown ? parseMarkdown(text) : null), [text, markdown]);
  const citedHeading = useMemo(() => {
    if (!cited) return null;
    let heading: number | null = null;
    for (const [unit, index] of headingIndexes(units)) {
      if (unit.page <= cited.first.page) heading = index;
    }
    return heading;
  }, [cited, units]);
  const mark = quoteMarkOf(target.citation);
  const [markTop, setMarkTop] = useState<number | null>(null);
  const [markRoom, setMarkRoom] = useState(0);

  // The Citation's mark sits beside the quote's first line, wherever the text wraps it.
  const showMark = mark !== null && highlight.length > 0;
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-placed for each new highlight
  useLayoutEffect(() => {
    const element = column.current;
    if (!element || !showMark) {
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
  }, [showMark, highlight]);

  // Each open request goes to its quote, else to its cited Unit, else the top.
  // Opening the Document again without either leaves the reader where they were.
  useLayoutEffect(() => {
    const container = scroller.current;
    if (!container || appliedRequest.current === target.request) return;
    const first = appliedRequest.current === null;
    appliedRequest.current = target.request;
    const quoted = container.querySelector<HTMLElement>("[data-quote-highlight]");
    const unit =
      citedHeading !== null && citedHeading >= 0
        ? container.querySelector(`[data-heading="${citedHeading}"]`)
        : cited
          ? container.querySelector(`[data-unit="${cited.first.page}"]`)
          : null;
    if (quoted) scrollToElement(container, quoted, container.clientHeight / 3);
    else if (unit) scrollToElement(container, unit, 24);
    else if (target.quote || target.pageFrom !== undefined || first) container.scrollTop = 0;
  }, [target.request, target.quote, target.pageFrom, cited, citedHeading]);

  const onScroll = () => {
    const container = scroller.current;
    if (!container || !outline) return;
    const top = container.getBoundingClientRect().top + 48;
    let heading: number | null = null;
    for (const element of container.querySelectorAll<HTMLElement>("[data-heading]")) {
      if (element.getBoundingClientRect().top <= top) heading = Number(element.dataset.heading);
    }
    setCurrent(heading);
  };

  let headingCount = 0;
  return (
    <div className="flex min-h-0 flex-1">
      {outline && (
        <UnitOutline
          entries={outline}
          current={current}
          onGo={(heading) => {
            const container = scroller.current;
            const element = container?.querySelector(`[data-heading="${heading}"]`);
            if (container && element) scrollToElement(container, element, 24);
            setCurrent(heading);
          }}
        />
      )}
      <div
        ref={scroller}
        data-testid="viewer-text"
        data-quote-tone={quoteTone(target.citation)}
        onScroll={onScroll}
        className="viewer-backdrop viewer-text-backdrop min-h-0 min-w-0 flex-1 overflow-auto select-text"
      >
        <article className="viewer-page viewer-text-page">
          <div ref={column} className="relative">
            {blocks ? (
              <div className="document-markdown">
                {blocks.map((block, index) => (
                  <MarkdownBlock
                    // biome-ignore lint/suspicious/noArrayIndexKey: the blocks never reorder
                    key={index}
                    block={block}
                    source={text}
                    highlight={highlight}
                    heading={block.kind === "heading" ? headingCount++ : null}
                  />
                ))}
              </div>
            ) : (
              <pre className="document-plain">
                <PlainText source={text} units={units} highlight={highlight} />
              </pre>
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
      </div>
    </div>
  );
}

/** Scrolls `container` so `element` is `offset` pixels below its top. */
function scrollToElement(container: HTMLElement, element: Element, offset: number): void {
  const top =
    element.getBoundingClientRect().top -
    container.getBoundingClientRect().top +
    container.scrollTop;
  container.scrollTop = Math.max(0, top - offset);
}

/** A plain-text file, each block of lines in a span the viewer can scroll to. */
function PlainText({
  source,
  units,
  highlight,
}: {
  source: string;
  units: readonly SourceUnit[];
  highlight: readonly TextRange[];
}) {
  const parts: ReactNode[] = [];
  let at = 0;
  for (const unit of units) {
    if (unit.start > at) {
      parts.push(
        <Slice
          key={`gap${at}`}
          source={source}
          start={at}
          end={unit.start}
          highlight={highlight}
        />,
      );
    }
    parts.push(
      <span key={unit.page} data-unit={unit.page}>
        <Slice source={source} start={unit.start} end={unit.end} highlight={highlight} />
      </span>,
    );
    at = unit.end;
  }
  parts.push(
    <Slice key={`gap${at}`} source={source} start={at} end={source.length} highlight={highlight} />,
  );
  return <>{parts}</>;
}

/** A span of the source, with the parts inside `highlight` (ranges in order) marked. */
function Slice({
  source,
  start,
  end,
  highlight,
}: {
  source: string;
  start: number;
  end: number;
  highlight: readonly TextRange[];
}) {
  const inside = highlight.filter((range) => range.end > start && range.start < end);
  if (inside.length === 0) return source.slice(start, end);
  const parts: ReactNode[] = [];
  let at = start;
  for (const range of inside) {
    const from = Math.max(at, range.start);
    const to = Math.min(end, range.end);
    if (from >= to) continue;
    parts.push(source.slice(at, from));
    // A part split across inlines (e.g. at a line break) joins up square at the seam.
    const joins = `${from > range.start ? " quote-highlight--joins-before" : ""}${
      to < range.end ? " quote-highlight--joins-after" : ""
    }`;
    parts.push(
      <mark key={from} data-quote-highlight="" className={`quote-highlight${joins}`}>
        {source.slice(from, to)}
      </mark>,
    );
    at = to;
  }
  parts.push(source.slice(at, end));
  return <>{parts}</>;
}

function renderInlines(
  inlines: readonly Inline[],
  source: string,
  highlight: readonly TextRange[],
): ReactNode[] {
  return inlines.map((inline, index) => renderInline(inline, index, source, highlight));
}

function renderInline(
  inline: Inline,
  key: number,
  source: string,
  highlight: readonly TextRange[],
): ReactNode {
  switch (inline.kind) {
    case "text":
      return (
        <Slice
          key={key}
          source={source}
          start={inline.start}
          end={inline.end}
          highlight={highlight}
        />
      );
    case "code":
      return (
        <code key={key}>
          <Slice source={source} start={inline.start} end={inline.end} highlight={highlight} />
        </code>
      );
    case "strong":
      return <strong key={key}>{renderInlines(inline.children, source, highlight)}</strong>;
    case "em":
      return <em key={key}>{renderInlines(inline.children, source, highlight)}</em>;
    case "link":
      return inline.href ? (
        // Opens in the User's browser: the main process sends every new window there.
        <a key={key} href={inline.href} target="_blank" rel="noreferrer">
          {renderInlines(inline.children, source, highlight)}
        </a>
      ) : (
        <span key={key}>{renderInlines(inline.children, source, highlight)}</span>
      );
  }
}

function MarkdownBlock({
  block,
  source,
  highlight,
  heading,
}: {
  block: Block;
  source: string;
  highlight: readonly TextRange[];
  /** A heading's index among the file's headings, which the outline and Citations go to. */
  heading: number | null;
}) {
  switch (block.kind) {
    case "heading": {
      const Heading = `h${block.level}` as const;
      return (
        <Heading data-heading={heading ?? undefined}>
          {renderInlines(block.inlines, source, highlight)}
        </Heading>
      );
    }
    case "paragraph":
      return <p>{renderInlines(block.inlines, source, highlight)}</p>;
    case "quote":
      return <blockquote>{renderInlines(block.inlines, source, highlight)}</blockquote>;
    case "list": {
      const List = block.ordered ? "ol" : "ul";
      return (
        <List>
          {block.items.map((item, index) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: the items never reorder
            <li key={index}>{renderInlines(item, source, highlight)}</li>
          ))}
        </List>
      );
    }
    case "verbatim":
      return (
        <pre>
          <Slice source={source} start={block.start} end={block.end} highlight={highlight} />
        </pre>
      );
    case "rule":
      return <hr />;
  }
}
