/**
 * A JATS article (the XML PubMed Central keeps its articles in) as text by
 * section: its title, its abstract, and each section of its body under its
 * heading, nested as the article nests them. Figures and tables keep their
 * captions; references, footnotes and other back matter are left out. Also
 * the licence and the version date the article states.
 */
import {
  child,
  descendants,
  elements,
  is,
  parseXml,
  textOf,
  type XmlElement,
  type XmlNode,
} from "../documents/formats/xml";
import { SourceError } from "./http";
import type { TextSection } from "./types";

/** Elements whose text is never the article's running text. */
const SKIPPED = new Set(["title", "label", "sec-meta", "object-id", "alternatives", "graphic"]);

const tidy = (text: string) => text.replace(/\s+/g, " ").trim();

/** An element's text, with a space where block elements meet, so words don't run together. */
function inline(node: XmlNode): string {
  if (typeof node === "string") return node;
  if (is(node, "xref") || is(node, "ext-link")) return textOf(node);
  return node.children.map(inline).join(is(node, "list-item") ? " " : "");
}

/** The paragraphs of a section's own content, not its sub-sections'. */
function paragraphsOf(section: XmlElement): string[] {
  const paragraphs: string[] = [];
  const walk = (node: XmlElement) => {
    for (const element of elements(node)) {
      if (is(element, "sec") || SKIPPED.has(element.name)) continue;
      if (is(element, "p") || is(element, "disp-quote")) {
        const text = tidy(inline(element));
        if (text) paragraphs.push(text);
      } else if (is(element, "list")) {
        for (const item of elements(element).filter((each) => is(each, "list-item"))) {
          const text = tidy(inline(item));
          if (text) paragraphs.push(`- ${text}`);
        }
      } else if (is(element, "fig") || is(element, "table-wrap")) {
        const label = tidy(
          textOf(child(element, "label") ?? { name: "", attrs: {}, children: [] }),
        );
        const caption = child(element, "caption");
        const text = tidy(caption ? inline(caption) : "");
        if (text) paragraphs.push(label ? `${label} ${text}` : text);
      } else {
        walk(element);
      }
    }
  };
  walk(section);
  return paragraphs;
}

function sectionsOf(parent: XmlElement, level: number, into: TextSection[]): void {
  for (const section of elements(parent).filter((each) => is(each, "sec"))) {
    const title = child(section, "title");
    const label = child(section, "label");
    const heading = tidy(
      [label ? textOf(label) : "", title ? textOf(title) : ""].filter(Boolean).join(" "),
    );
    into.push({ heading: heading || "Section", level, paragraphs: paragraphsOf(section) });
    sectionsOf(section, level + 1, into);
  }
}

/** A version date written in JATS: "2023-03-09" from <pub-date pub-type="epub"> or the like. */
function dateOf(element: XmlElement | undefined): string | null {
  if (!element) return null;
  const year = textOf(child(element, "year") ?? element).trim();
  if (!/^\d{4}$/.test(year)) return null;
  const pad = (part: XmlElement | undefined) => {
    const value = part ? Number(textOf(part)) : Number.NaN;
    return Number.isInteger(value) && value > 0 ? String(value).padStart(2, "0") : null;
  };
  const month = pad(child(element, "month"));
  const day = month ? pad(child(element, "day")) : null;
  return [year, month, day].filter(Boolean).join("-");
}

export interface JatsArticle {
  title: string;
  sections: TextSection[];
  /** The licence's address (or its words, if it gives no address); null if none. */
  licence: string | null;
  versionDate: string | null;
}

/**
 * Reads a JATS article (one <article>, or PMC's <pmc-articleset> around
 * one). Its DOCTYPE, which names NLM's DTD, is dropped before parsing: no
 * entity it could declare is expanded. Throws `SourceError` ("not-open")
 * when it has no body: the publisher doesn't let PMC give its full text.
 */
export function readJats(xml: string): JatsArticle {
  const declaration = /<!DOCTYPE[^>[]*>/i;
  if (/<!DOCTYPE[^>]*\[/i.test(xml)) {
    throw new SourceError("service-error", "The article declares entities of its own.");
  }
  let root: XmlElement;
  try {
    root = parseXml(xml.replace(declaration, ""));
  } catch {
    throw new SourceError("service-error", "PMC's answer isn't readable.");
  }
  const article = is(root, "article") ? root : descendants(root, "article")[0];
  if (!article) throw new SourceError("not-open", "PMC has no article by that id.");
  const meta = descendants(article, "article-meta")[0];
  const title = tidy(textOf(meta ? (descendants(meta, "article-title")[0] ?? meta) : article));
  const body = descendants(article, "body")[0];
  if (!body || elements(body).length === 0) {
    throw new SourceError("not-open", "PMC doesn't give this article's full text.");
  }
  const sections: TextSection[] = [];
  const abstract = meta ? descendants(meta, "abstract")[0] : undefined;
  if (abstract) {
    const own = paragraphsOf(abstract);
    const parts: TextSection[] = [];
    sectionsOf(abstract, 2, parts);
    sections.push({
      heading: "Abstract",
      level: 1,
      paragraphs: [...own, ...parts.flatMap((part) => [part.heading, ...part.paragraphs])],
    });
  }
  // Text before the first section (some articles have no sections at all).
  const opening = paragraphsOf(body);
  if (opening.length > 0) sections.push({ heading: "Main text", level: 1, paragraphs: opening });
  sectionsOf(body, 1, sections);
  const licence = meta ? descendants(meta, "license")[0] : undefined;
  // ALI's <license_ref> holds the licence's address; older articles put it on <license>, or link
  // it in the licence's words. Words alone ("available for text mining…") name no licence.
  const reference = licence ? elements(licence).find((each) => is(each, "license_ref")) : undefined;
  const links = licence
    ? [licence, ...descendants(licence, "ext-link")]
        .map((each) => each.attrs["xlink:href"] ?? "")
        .filter(Boolean)
    : [];
  const licenceText = reference
    ? tidy(textOf(reference))
    : (links.find((link) => /creativecommons\.org/i.test(link)) ?? links[0] ?? "");
  const dates = meta ? descendants(meta, "pub-date") : [];
  const published =
    dates.find(
      (each) =>
        each.attrs["pub-type"] === "epub" || each.attrs["publication-format"] === "electronic",
    ) ?? dates[0];
  return { title, sections, licence: licenceText || null, versionDate: dateOf(published) };
}
