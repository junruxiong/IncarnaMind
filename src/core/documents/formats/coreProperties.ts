/**
 * The core properties of an Office package (.docx, .pptx, .xlsx): the part
 * the package's relationships name, usually docProps/core.xml. Only the
 * creation date is read (#53), never dcterms:modified.
 */
import { child, elements, parseXml, textOf } from "./xml";
import { openPackage, resolvePart } from "./zip";

/**
 * The end of the relationship type that names the part: writers use the
 * package namespace (…/package/2006/relationships/…), the officeDocument
 * one, or Strict's.
 */
const CORE_PROPERTIES = "/metadata/core-properties";

/** The part's usual place, for a package whose relationships don't say. */
const DEFAULT_PART = "docProps/core.xml";

/** The core properties part is a few hundred bytes: more than this isn't one. */
const MAX_BYTES = 1024 * 1024;

/** The package's creation date (dcterms:created) and title (dc:title) as written, each null if absent. */
export async function officeProperties(
  bytes: Uint8Array,
): Promise<{ created: string | null; title: string | null }> {
  const zip = openPackage(bytes);
  const relsXml = await zip.readText("_rels/.rels", MAX_BYTES);
  const target = relsXml
    ? elements(parseXml(relsXml)).find((rel) => rel.attrs.Type?.endsWith(CORE_PROPERTIES))?.attrs
        .Target
    : undefined;
  const xml = await zip.readText(target ? resolvePart("", target) : DEFAULT_PART, MAX_BYTES);
  if (xml === undefined) return { created: null, title: null };
  const root = parseXml(xml);
  const created = child(root, "created");
  const title = child(root, "title");
  return {
    created: created ? textOf(created).trim() : null,
    title: title ? textOf(title).trim() : null,
  };
}
