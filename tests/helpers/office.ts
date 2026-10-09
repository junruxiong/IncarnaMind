/**
 * Small Office files built in the tests, for cases the committed fixtures
 * (tests/fixtures/formats/) don't cover: huge sheets, long sections, files
 * that can't be read, and core properties of any kind (creation dates).
 * Written with the test zip helper; nothing here
 * is a real Office library.
 */
import { buildZip } from "./zip";

const escapeXml = (text: string) =>
  text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

const CONTENT_TYPES = (overrides: string) =>
  `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>${overrides}</Types>`;

const columnName = (index: number) => {
  let name = "";
  for (let n = index + 1; n > 0; n = Math.floor((n - 1) / 26)) {
    name = String.fromCharCode(65 + ((n - 1) % 26)) + name;
  }
  return name;
};

/** docProps/core.xml as Office writes it, with these dates (W3CDTF), if given. */
export function corePropertiesXml(dates: { created?: string; modified?: string }): string {
  const date = (name: string, value: string | undefined) =>
    value === undefined
      ? ""
      : `<dcterms:${name} xsi:type="dcterms:W3CDTF">${escapeXml(value)}</dcterms:${name}>`;
  return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:creator>IncarnaMind tests</dc:creator>${date("created", dates.created)}${date("modified", dates.modified)}</cp:coreProperties>`;
}

/** What a package holds besides its content. */
export interface PackageOptions {
  /** The core properties part as written (see `corePropertiesXml`), with the package's relationship to it. */
  coreXml?: string;
  /** Where the core properties part is, if not docProps/core.xml. */
  corePath?: string;
}

/** The core properties part, and the package relationships that point to it. */
const coreEntries = ({ coreXml, corePath = "docProps/core.xml" }: PackageOptions) =>
  coreXml === undefined
    ? []
    : [
        {
          name: "_rels/.rels",
          data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId9" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="${corePath}"/></Relationships>`,
        },
        { name: corePath, data: coreXml },
      ];

export interface SheetInput {
  name: string;
  /** Rows from 1; numbers are stored as numbers, strings inline. */
  rows: (string | number)[][];
}

/** A sheet of an .xlsx written as XML, for the look the sheet preview reads. */
export interface RawSheet {
  name: string;
  /** What goes inside <worksheet>: <sheetViews>, <cols>, <sheetData>, <drawing>… */
  xml: string;
  /** The sheet's relationships, as <Relationship> elements. */
  rels?: string;
  state?: "hidden";
}

/**
 * An .xlsx put together from raw parts: its sheets, styles.xml, a theme, shared
 * strings and any other parts (drawings, media, charts, comments).
 */
export function xlsxPackage(input: {
  sheets: readonly RawSheet[];
  styles?: string;
  theme?: string;
  sharedStrings?: readonly string[];
  /** The sheet the workbook opens at, by its place among all sheets. */
  activeTab?: number;
  parts?: readonly { name: string; data: string | Buffer }[];
}): Buffer {
  const ns = `xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"`;
  const rel = (id: string, type: string, target: string) =>
    `<Relationship Id="${id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/${type}" Target="${target}"/>`;
  const relationships = (body: string) =>
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${body}</Relationships>`;
  const workbookRels = [
    ...input.sheets.map((_, index) =>
      rel(`rId${index + 1}`, "worksheet", `worksheets/sheet${index + 1}.xml`),
    ),
    input.styles ? rel("rIdS", "styles", "styles.xml") : "",
    input.theme ? rel("rIdT", "theme", "theme/theme1.xml") : "",
    input.sharedStrings ? rel("rIdSS", "sharedStrings", "sharedStrings.xml") : "",
  ].join("");
  const view =
    input.activeTab === undefined
      ? ""
      : `<bookViews><workbookView activeTab="${input.activeTab}"/></bookViews>`;
  const entries: { name: string; data: string | Buffer }[] = [
    { name: "[Content_Types].xml", data: CONTENT_TYPES("") },
    {
      name: "xl/workbook.xml",
      data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook ${ns}>${view}<sheets>${input.sheets
        .map(
          (sheet, index) =>
            `<sheet name="${escapeXml(sheet.name)}" sheetId="${index + 1}"${sheet.state ? ` state="${sheet.state}"` : ""} r:id="rId${index + 1}"/>`,
        )
        .join("")}</sheets></workbook>`,
    },
    { name: "xl/_rels/workbook.xml.rels", data: relationships(workbookRels) },
    ...input.sheets.flatMap((sheet, index) => [
      {
        name: `xl/worksheets/sheet${index + 1}.xml`,
        data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet ${ns}>${sheet.xml}</worksheet>`,
      },
      ...(sheet.rels
        ? [
            {
              name: `xl/worksheets/_rels/sheet${index + 1}.xml.rels`,
              data: relationships(sheet.rels),
            },
          ]
        : []),
    ]),
  ];
  if (input.styles) {
    entries.push({
      name: "xl/styles.xml",
      data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet ${ns}>${input.styles}</styleSheet>`,
    });
  }
  if (input.theme) entries.push({ name: "xl/theme/theme1.xml", data: input.theme });
  if (input.sharedStrings) {
    entries.push({
      name: "xl/sharedStrings.xml",
      data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><sst ${ns}>${input.sharedStrings.join("")}</sst>`,
    });
  }
  entries.push(...(input.parts ?? []));
  return buildZip(entries);
}

/** An .xlsx with these sheets, in order, values unformatted. */
export function xlsxOf(sheets: readonly SheetInput[], options: PackageOptions = {}): Buffer {
  const sheetXml = (sheet: SheetInput) => {
    const rows = sheet.rows
      .map((cells, index) => {
        const row = index + 1;
        const xml = cells
          .map((value, column) => {
            const ref = `${columnName(column)}${row}`;
            return typeof value === "number"
              ? `<c r="${ref}"><v>${value}</v></c>`
              : `<c r="${ref}" t="inlineStr"><is><t>${escapeXml(value)}</t></is></c>`;
          })
          .join("");
        return `<row r="${row}">${xml}</row>`;
      })
      .join("");
    return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>${rows}</sheetData></worksheet>`;
  };
  return buildZip([
    {
      name: "[Content_Types].xml",
      data: CONTENT_TYPES(
        `<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>`,
      ),
    },
    {
      name: "xl/workbook.xml",
      data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>${sheets
        .map(
          (sheet, index) =>
            `<sheet name="${escapeXml(sheet.name)}" sheetId="${index + 1}" r:id="rId${index + 1}"/>`,
        )
        .join("")}</sheets></workbook>`,
    },
    {
      name: "xl/_rels/workbook.xml.rels",
      data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${sheets
        .map(
          (_, index) =>
            `<Relationship Id="rId${index + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet${index + 1}.xml"/>`,
        )
        .join("")}</Relationships>`,
    },
    ...sheets.map((sheet, index) => ({
      name: `xl/worksheets/sheet${index + 1}.xml`,
      data: sheetXml(sheet),
    })),
    ...coreEntries(options),
  ]);
}

/** A .docx of these paragraphs; a heading has its level. */
export function docxOf(
  paragraphs: readonly { text: string; heading?: number }[],
  options: PackageOptions = {},
): Buffer {
  const body = paragraphs
    .map(({ text, heading }) => {
      const style = heading ? `<w:pPr><w:pStyle w:val="Heading${heading}"/></w:pPr>` : "";
      return `<w:p>${style}<w:r><w:t xml:space="preserve">${escapeXml(text)}</w:t></w:r></w:p>`;
    })
    .join("");
  const styles = [1, 2, 3]
    .map(
      (level) =>
        `<w:style w:type="paragraph" w:styleId="Heading${level}"><w:name w:val="heading ${level}"/></w:style>`,
    )
    .join("");
  return buildZip([
    { name: "[Content_Types].xml", data: CONTENT_TYPES("") },
    {
      name: "word/document.xml",
      data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>${body}</w:body></w:document>`,
    },
    {
      name: "word/styles.xml",
      data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">${styles}</w:styles>`,
    },
    ...coreEntries(options),
  ]);
}

/**
 * The start of an OLE compound file holding an "EncryptedPackage" stream:
 * what Office writes when a .docx, .pptx or .xlsx is saved with a password.
 */
export function encryptedOfficeFile(): Buffer {
  const header = Buffer.from([0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1]);
  const name = Buffer.from("EncryptedPackage", "utf16le");
  return Buffer.concat([header, Buffer.alloc(504), name, Buffer.alloc(64)]);
}

/** An OLE compound file without one: a Word 97–2003 file renamed .docx. */
export function legacyOfficeFile(): Buffer {
  const header = Buffer.from([0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1]);
  return Buffer.concat([header, Buffer.alloc(504), Buffer.from("WordDocument", "utf16le")]);
}
