import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Document } from "../../../core/api";
import { readWebPage, type WebPage } from "../../../core/documents/formats/web";
import type { XmlNode } from "../../../core/documents/formats/xml";
import type { MessageKey } from "../../../shared/i18n";
import { useLanguage, useT } from "../i18n";
import type { ViewerTarget } from "../store";
import { pageDate } from "../webPages";
import { markQuote, textNodes, unmarkQuote } from "./domQuote";
import { FileStates } from "./FileStates";
import { CitationMark, marginRoom, markTopFor, quoteMarkOf, quoteTone } from "./quoteMark";
import { OutlineButton, type OutlineEntry, UnitOutline } from "./UnitOutline";
import { useDocumentFile } from "./useDocumentFile";
import { ViewerHeader } from "./ViewerHeader";

const read = (bytes: Uint8Array): WebPage => readWebPage(new TextDecoder().decode(bytes));

/** The elements a snapshot is drawn with; any other is drawn as its content alone. */
const DRAWN = new Set([
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "p",
  "ul",
  "ol",
  "li",
  "blockquote",
  "pre",
  "thead",
  "tbody",
  "tfoot",
  "tr",
  "th",
  "td",
  "dl",
  "dt",
  "dd",
  "figure",
  "figcaption",
  "hr",
  "br",
  "strong",
  "em",
  "code",
  "sub",
  "sup",
  "s",
  "mark",
]);

const count = (value: string | undefined) => {
  const number = Number(value);
  return Number.isInteger(number) && number > 0 && number < 1000 ? String(number) : null;
};

/**
 * Draws a snapshot's article with DOM calls, never as markup: only the
 * elements in `DRAWN`, links to web addresses (which open in the browser),
 * and a placeholder for each picture, which isn't fetched.
 */
function draw(
  nodes: readonly XmlNode[],
  parent: HTMLElement,
  blocks: ReadonlyMap<XmlNode, number>,
  t: (key: MessageKey) => string,
): void {
  const owner = parent.ownerDocument;
  for (const node of nodes) {
    if (typeof node === "string") {
      parent.append(owner.createTextNode(node));
      continue;
    }
    const name = node.name.toLowerCase();
    let element: HTMLElement | null = null;
    let inner: HTMLElement | null = null;
    if (name === "img") {
      element = owner.createElement("span");
      element.className = "document-image-placeholder";
      element.dataset.quoteSkip = "";
      const label = owner.createElement("span");
      label.className = "document-image-placeholder-label";
      label.textContent = node.attrs.alt?.trim() || t("viewer.markdown.image");
      const note = owner.createElement("span");
      note.className = "document-image-placeholder-note";
      note.textContent = t("viewer.markdown.imageWeb");
      element.append(label, note);
      parent.append(element);
      continue;
    }
    if (name === "a") {
      const href = node.attrs.href ?? "";
      if (/^https?:\/\//i.test(href)) {
        const link = owner.createElement("a");
        link.href = href;
        link.target = "_blank";
        link.rel = "noreferrer";
        element = link;
      }
    } else if (name === "table") {
      element = owner.createElement("div");
      element.className = "document-table";
      inner = owner.createElement("table");
      element.append(inner);
    } else if (DRAWN.has(name)) {
      element = owner.createElement(name);
      const start = name === "ol" ? count(node.attrs.start) : null;
      if (start) element.setAttribute("start", start);
      for (const span of name === "td" || name === "th" ? ["colspan", "rowspan"] : []) {
        const value = count(node.attrs[span]);
        if (value) element.setAttribute(span, value);
      }
    }
    const block = blocks.get(node);
    if (element && block !== undefined) element.dataset.block = String(block);
    draw(node.children, inner ?? element ?? parent, blocks, t);
    if (element) parent.append(element);
  }
}

/**
 * A saved web page (#261): its snapshot as a clean reading page, under its
 * title and byline. Opened at a Citation, the quote is highlighted in the
 * cited section, else anywhere on the page, with the Citation's mark beside it.
 */
export function WebView({ document, target }: { document: Document; target: ViewerTarget }) {
  const loaded = useDocumentFile(document.id, read);
  return (
    <FileStates
      loaded={loaded}
      quote={target.quote}
      ready={(page) => <WebPageContent page={page} document={document} target={target} />}
    />
  );
}

function WebPageContent({
  page,
  document,
  target,
}: {
  page: WebPage;
  document: Document;
  target: ViewerTarget;
}) {
  const t = useT();
  const language = useLanguage();
  const scroller = useRef<HTMLDivElement>(null);
  const column = useRef<HTMLDivElement>(null);
  const host = useRef<HTMLDivElement>(null);
  const [outlineOpen, setOutlineOpen] = useState(false);
  const [current, setCurrent] = useState<number | null>(null);
  // Counts the highlights made, none while 0: the mark is placed again for each.
  const [highlighted, setHighlighted] = useState(0);
  const [markTop, setMarkTop] = useState<number | null>(null);
  const [markRoom, setMarkRoom] = useState(0);
  const mark = quoteMarkOf(target.citation);
  const blocks = useMemo(
    () => new Map<XmlNode, number>(page.blocks.map((block, index) => [block.element, index])),
    [page],
  );
  const outline: OutlineEntry[] = useMemo(
    () =>
      page.blocks.flatMap((block, index) =>
        block.level === null
          ? []
          : [{ key: index, title: block.text, depth: Math.min(block.level - 1, 3) }],
      ),
    [page],
  );
  const site = page.meta.siteName ?? document.web?.siteName ?? null;
  const byline = [page.meta.author, page.meta.published && pageDate(page.meta.published, language)]
    .filter(Boolean)
    .join(" · ");

  // Drawn once per snapshot read.
  useLayoutEffect(() => {
    const body = host.current;
    if (!body || !page.article) return;
    draw(page.article.children, body, blocks, t);
    return () => body.replaceChildren();
  }, [page, blocks, t]);

  // Each open request highlights its quote: in the cited section first, then anywhere.
  // biome-ignore lint/correctness/useExhaustiveDependencies: each open request goes there again
  useLayoutEffect(() => {
    const body = host.current;
    const container = scroller.current;
    if (!body || !container) return;
    unmarkQuote(body);
    const cited = page.units.filter(
      (unit) =>
        target.pageFrom !== undefined &&
        unit.page >= target.pageFrom &&
        unit.page <= (target.pageTo ?? target.pageFrom),
    );
    const first = cited[0] && body.querySelector(`[data-block="${cited[0].blocks.first}"]`);
    const last = cited.at(-1) && body.querySelector(`[data-block="${cited.at(-1)?.blocks.last}"]`);
    let range: Range | undefined;
    if (first && last) {
      range = body.ownerDocument.createRange();
      range.setStartBefore(first);
      range.setEndAfter(last);
    }
    const quote = target.quote;
    const marks = quote
      ? ((range && markQuote(textNodes(body, range), quote)) ?? markQuote(textNodes(body), quote))
      : null;
    setHighlighted((count) => (marks ? count + 1 : 0));
    const anchor = marks?.[0] ?? first ?? null;
    const top = anchor
      ? anchor.getBoundingClientRect().top -
        container.getBoundingClientRect().top +
        container.scrollTop
      : 0;
    container.scrollTop = Math.max(0, top - (marks ? container.clientHeight / 3 : 24));
  }, [page, target.request, target.quote, target.pageFrom, target.pageTo]);

  // The Citation's mark sits beside the quote's first line, wherever the text wraps it.
  useLayoutEffect(() => {
    const element = column.current;
    if (!element || !mark || highlighted === 0) {
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
  }, [mark, highlighted]);

  const goTo = (key: number) => {
    const container = scroller.current;
    const element = host.current?.querySelector(`[data-block="${key}"]`);
    if (!container || !element) return;
    container.scrollTop =
      element.getBoundingClientRect().top -
      container.getBoundingClientRect().top +
      container.scrollTop -
      24;
    setCurrent(key);
  };

  const onScroll = () => {
    const container = scroller.current;
    if (!container || !outlineOpen) return;
    const top = container.getBoundingClientRect().top + 48;
    let heading: number | null = null;
    for (const entry of outline) {
      const element = host.current?.querySelector(`[data-block="${entry.key}"]`);
      if (element && element.getBoundingClientRect().top <= top) heading = entry.key;
    }
    setCurrent(heading);
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
          data-testid="viewer-web"
          data-quote-tone={quoteTone(target.citation)}
          onScroll={onScroll}
          className="viewer-backdrop viewer-text-backdrop min-h-0 min-w-0 flex-1 overflow-auto select-text"
        >
          <article className="viewer-page viewer-text-page" lang={page.meta.language ?? undefined}>
            <div ref={column} className="relative">
              <header data-quote-skip="" className="document-web-header">
                {site && <p className="document-web-site">{site}</p>}
                <h1 data-testid="viewer-web-title" className="document-web-title">
                  {page.meta.title ?? document.name}
                </h1>
                {byline && <p className="document-web-byline">{byline}</p>}
              </header>
              <div ref={host} data-testid="viewer-web-article" className="document-markdown" />
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
    </>
  );
}
