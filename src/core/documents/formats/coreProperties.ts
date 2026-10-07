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

/**
 * The package's creation date (dcterms:created) as written, or null if it
 * has none. Throws `ExtractionError` for a file that isn't a package, or
 * whose core properties can't be read.
 */
export async function officeCreated(bytes: Uint8Array): Promise<string | null> {
  const zip = openPackage(bytes);
  const relsXml = await zip.readText("_rels/.rels", MAX_BYTES);
  const target = relsXml
    ? elements(parseXml(relsXml)).find((rel) => rel.attrs.Type?.endsWith(CORE_PROPERTIES))?.attrs
        .Target
    : undefined;
  const xml = await zip.readText(target ? resolvePart("", target) : DEFAULT_PART, MAX_BYTES);
  if (xml === undefined) return null;
  const created = child(parseXml(xml), "created");
  return created ? textOf(created).trim() : null;
}
