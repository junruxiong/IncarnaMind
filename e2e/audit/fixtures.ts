/**
 * Fixture folders for the UI audit, written to a temporary folder:
 *
 * - `formats/`: the repo's Office fixtures and the example Documents, plus a
 *   300-page PDF, a ~50k-cell .xlsx, a .docx with images, Markdown, plain
 *   text, a Chinese Document and one with a very long name.
 * - `scale/`: 2,000 small Documents (Markdown, text, CSV and PDF) in nested
 *   folders, for the sidebar, the Library and indexing at scale.
 */
import { copyFileSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { deflateSync } from "node:zlib";
import { xlsxOf } from "../../tests/helpers/office";
import { buildPdf } from "../../tests/helpers/pdf";
import { buildZip } from "../../tests/helpers/zip";

const REPO = join(__dirname, "..", "..");

// A seeded random, so every run writes the same files.
let seed = 7;
const random = () => {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return seed / 2147483648;
};
const pick = <T>(items: readonly T[]): T => items[Math.floor(random() * items.length)] as T;

const SUBJECTS = {
  "Coastal ecology": [
    "tide",
    "estuary",
    "salt marsh",
    "seagrass",
    "sediment",
    "shoreline",
    "harbour",
    "erosion",
  ],
  "Climate policy": [
    "carbon",
    "emissions",
    "mitigation",
    "adaptation",
    "levy",
    "subsidy",
    "target",
    "treaty",
  ],
  "Tea trade": ["tea", "plantation", "auction", "export", "leaf", "oxidation", "blend", "harvest"],
  "Machine learning": [
    "model",
    "training",
    "dataset",
    "benchmark",
    "transformer",
    "embedding",
    "scaling",
    "evaluation",
  ],
  "Contract law": [
    "clause",
    "liability",
    "indemnity",
    "breach",
    "warranty",
    "termination",
    "jurisdiction",
    "remedy",
  ],
  "Public health": [
    "cohort",
    "vaccine",
    "incidence",
    "trial",
    "outcome",
    "exposure",
    "screening",
    "mortality",
  ],
  "Urban planning": [
    "zoning",
    "transit",
    "housing",
    "density",
    "corridor",
    "permit",
    "footprint",
    "heritage",
  ],
  Finance: [
    "revenue",
    "margin",
    "forecast",
    "liquidity",
    "covenant",
    "dividend",
    "valuation",
    "ledger",
  ],
} as const;

const FILLER = [
  "the survey found",
  "in the second quarter",
  "according to the field notes",
  "compared with last year",
  "as the committee noted",
  "under the revised method",
  "the team recorded",
  "across all sites",
  "with a small sample",
  "before the review",
];

function sentence(words: readonly string[]): string {
  const parts = [
    pick(FILLER),
    pick(words),
    pick(["rose", "fell", "held", "shifted", "doubled"]),
    pick(FILLER),
    pick(words),
  ];
  const text = parts.join(" ");
  return `${text.charAt(0).toUpperCase()}${text.slice(1)}.`;
}

function paragraph(words: readonly string[], sentences = 4): string {
  return Array.from({ length: sentences }, () => sentence(words)).join(" ");
}

// --- A tiny PNG encoder, for the .docx's figures -------------------------

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
const crc32 = (data: Buffer) => {
  let c = 0xffffffff;
  for (const byte of data) c = (CRC_TABLE[(c ^ byte) & 0xff] as number) ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};
const chunk = (type: string, data: Buffer) => {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([length, body, crc]);
};

/** A width×height RGB chart-like picture with some noise (so it doesn't compress to nothing). */
function png(width: number, height: number, hue: number): Buffer {
  const raw = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * 3 + 1)] = 0;
    for (let x = 0; x < width; x++) {
      const i = y * (width * 3 + 1) + 1 + x * 3;
      const bar = Math.floor(x / (width / 12));
      const top = height * (0.2 + 0.6 * Math.abs(Math.sin(bar * 1.3 + hue)));
      const inBar = y > top && x % Math.floor(width / 12) > 8;
      const noise = Math.floor(random() * 24);
      raw[i] = inBar ? 40 + hue * 20 + noise : 244 - noise / 3;
      raw[i + 1] = inBar ? 90 + noise : 245 - noise / 3;
      raw[i + 2] = inBar ? 200 - hue * 30 + noise : 247 - noise / 3;
    }
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 2;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", header),
    chunk("IDAT", deflateSync(raw)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

const escapeXml = (text: string) =>
  text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

/** A .docx with headings, paragraphs and `images` inline figures. */
function docxWithImages(title: string, sections: number, images: number): Buffer {
  const words = SUBJECTS["Coastal ecology"];
  const body: string[] = [];
  const p = (text: string, style?: string) =>
    `<w:p>${style ? `<w:pPr><w:pStyle w:val="${style}"/></w:pPr>` : ""}<w:r><w:t xml:space="preserve">${escapeXml(text)}</w:t></w:r></w:p>`;
  body.push(p(title, "Title"));
  let image = 0;
  for (let s = 1; s <= sections; s++) {
    body.push(
      p(
        `${s}. ${pick(["Methods", "Results", "Site survey", "Discussion", "Sediment cores", "Seagrass cover"])} ${s}`,
        "Heading1",
      ),
    );
    for (let k = 0; k < 4; k++) body.push(p(paragraph(words, 5)));
    if (image < images) {
      image++;
      const cx = 5486400;
      const cy = 3086100;
      body.push(
        `<w:p><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="${cx}" cy="${cy}"/><wp:docPr id="${image}" name="Figure ${image}"/><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="${image}" name="image${image}.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="rIdImg${image}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="${cx}" cy="${cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>`,
      );
      body.push(p(`Figure ${image}: seagrass cover by transect, survey ${image}.`, "Caption"));
    }
  }
  const styles = ["Title", "Heading1", "Caption"]
    .map(
      (id) =>
        `<w:style w:type="paragraph" w:styleId="${id}"><w:name w:val="${id === "Heading1" ? "heading 1" : id.toLowerCase()}"/></w:style>`,
    )
    .join("");
  const rels = Array.from(
    { length: image },
    (_, i) =>
      `<Relationship Id="rIdImg${i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image${i + 1}.png"/>`,
  ).join("");
  return buildZip([
    {
      name: "[Content_Types].xml",
      data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>`,
    },
    {
      name: "_rels/.rels",
      data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>`,
    },
    {
      name: "word/_rels/document.xml.rels",
      data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>${rels}</Relationships>`,
    },
    {
      name: "word/document.xml",
      data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"><w:body>${body.join("")}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body></w:document>`,
    },
    {
      name: "word/styles.xml",
      data: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">${styles}</w:styles>`,
    },
    ...Array.from({ length: image }, (_, i) => ({
      name: `word/media/image${i + 1}.png`,
      data: png(1200, 675, i % 4),
    })),
  ]);
}

const LONG_NAME =
  "Harbour sediment survey, east basin transects 1–14, combined field notes, lab results and the committee's comments on the revised method — final version for the board (3)";

export interface Fixtures {
  root: string;
  formats: string;
  scale: string;
  names: {
    pdf: string;
    bigPdf: string;
    docx: string;
    imageDocx: string;
    xlsx: string;
    bigXlsx: string;
    csv: string;
    pptx: string;
    md: string;
    txt: string;
    chinese: string;
    long: string;
  };
  scaleCount: number;
}

/** Writes the `formats/` folder (and `scale/` when `scaleCount` > 0) under `root`. */
export function writeFixtures(root: string, scaleCount = 0): Fixtures {
  const formats = join(root, "formats");
  mkdirSync(formats, { recursive: true });
  const repoFormats = join(REPO, "tests", "fixtures", "formats");
  for (const name of readdirSync(repoFormats))
    copyFileSync(join(repoFormats, name), join(formats, name));
  const examples = join(REPO, "resources", "examples");
  for (const name of readdirSync(examples).filter((n) => n.endsWith(".md"))) {
    copyFileSync(join(examples, name), join(formats, name));
  }

  const words = SUBJECTS["Coastal ecology"];
  writeFileSync(
    join(formats, "Tide report.pdf"),
    buildPdf([
      { lines: ["Tides and the Moon", "The Moon raises two bulges of water on the Earth."] },
      { lines: ["Spring and neap tides", "Spring tides happen at new moon and at full moon."] },
      { lines: ["Tide tables", "Harbours publish the times of high water every year."] },
    ]),
  );
  // 300 pages of a long report, ~40 lines each.
  writeFileSync(
    join(formats, "Annual survey (300 pages).pdf"),
    buildPdf(
      Array.from({ length: 300 }, (_, page) => ({
        lines: [
          `Chapter ${Math.floor(page / 20) + 1}, page ${page + 1}`,
          ...Array.from({ length: 38 }, () => sentence(words).slice(0, 90)),
        ],
      })),
    ),
  );
  writeFileSync(
    join(formats, "Seagrass field study (figures).docx"),
    docxWithImages("Seagrass field study", 12, 8),
  );
  // 2,000 rows × 25 columns = 50,000 cells.
  writeFileSync(
    join(formats, "Transect measurements (50k cells).xlsx"),
    xlsxOf([
      {
        name: "Measurements",
        rows: [
          ["Site", "Date", ...Array.from({ length: 23 }, (_, c) => `Depth ${c + 1} (cm)`)],
          ...Array.from({ length: 1999 }, (_, r) => [
            `Transect ${(r % 14) + 1}`,
            `2026-0${(r % 9) + 1}-1${r % 10}`,
            ...Array.from({ length: 23 }, () => Math.round(random() * 10000) / 100),
          ]),
        ],
      },
      { name: "Notes", rows: [["Note"], ...Array.from({ length: 30 }, () => [sentence(words)])] },
    ]),
  );
  writeFileSync(
    join(formats, "Reading notes.md"),
    [
      "# Reading notes",
      "",
      "## Tides",
      "",
      paragraph(words, 5),
      "",
      "- Spring tides happen at new moon and at full moon.",
      "- Neap tides happen at the quarter moons.",
      "",
      "## Sediment",
      "",
      paragraph(words, 6),
      "",
      "```python",
      "def tidal_range(high, low):",
      "    return high - low",
      "```",
      "",
      "| Site | Range (m) |",
      "|---|---|",
      "| Dover | 6.1 |",
      "| Calais | 6.9 |",
      "",
      ...Array.from({ length: 30 }, () => `${paragraph(words, 3)}\n`),
    ].join("\n"),
  );
  writeFileSync(
    join(formats, "Interview transcript.txt"),
    Array.from(
      { length: 400 },
      (_, i) => `${i % 2 ? "Interviewer" : "Harbour master"}: ${sentence(words)}`,
    ).join("\n"),
  );
  writeFileSync(
    join(formats, "潮汐与港口 · 研究笔记.md"),
    [
      "# 潮汐与港口",
      "",
      "## 大潮与小潮",
      "",
      "大潮发生在新月和满月时，此时太阳和月球的引力叠加，潮差最大。小潮发生在上弦月和下弦月时，潮差最小。",
      "",
      "## 港口的潮汐表",
      "",
      "港口每年都会公布高潮和低潮的时间。航运公司根据潮汐表安排大型船舶进出港，以避免搁浅。",
      "",
      ...Array.from(
        { length: 20 },
        (_, i) =>
          `第${i + 1}段：海岸侵蚀与沉积物的变化需要长期观测，研究团队在东港区设立了十四条断面。\n`,
      ),
    ].join("\n"),
  );
  writeFileSync(join(formats, `${LONG_NAME}.md`), `# ${LONG_NAME}\n\n${paragraph(words, 8)}\n`);

  let written = 0;
  const scale = join(root, "scale");
  if (scaleCount > 0) {
    const subjects = Object.entries(SUBJECTS);
    const years = ["2023", "2024", "2025", "2026"];
    for (let i = 0; i < scaleCount; i++) {
      const [subject, subjectWords] = subjects[i % subjects.length] as [string, readonly string[]];
      const year = pick(years);
      const dir = join(scale, subject, year, i % 3 === 0 ? "Drafts" : "");
      mkdirSync(dir, { recursive: true });
      const kind = i % 40 === 0 ? "pdf" : i % 12 === 0 ? "csv" : i % 3 === 0 ? "txt" : "md";
      const title = `${year}-${String((i % 12) + 1).padStart(2, "0")} ${pick(["Notes on", "Review of", "Memo:", "Draft:", "Summary of", "Minutes —"])} ${pick(subjectWords)} ${pick(subjectWords)} ${i}`;
      const name =
        i % 97 === 0
          ? `${title} ${LONG_NAME.slice(0, 80)}`
          : i % 50 === 0
            ? `${year} 研究笔记 ${i}`
            : title;
      const file = join(dir, `${name}.${kind}`);
      if (kind === "pdf") {
        writeFileSync(
          file,
          buildPdf([
            {
              lines: [
                title.replace(/[^\x20-\xff]/g, "-"),
                ...Array.from({ length: 12 }, () => sentence(subjectWords).slice(0, 90)),
              ],
            },
            { lines: Array.from({ length: 12 }, () => sentence(subjectWords).slice(0, 90)) },
          ]),
        );
      } else if (kind === "csv") {
        writeFileSync(
          file,
          [
            "Item,Value,Note",
            ...Array.from(
              { length: 40 },
              (_, r) => `${pick(subjectWords)} ${r},${Math.round(random() * 1000)},${pick(FILLER)}`,
            ),
          ].join("\n"),
        );
      } else {
        const body = Array.from({ length: 3 + (i % 5) }, () => paragraph(subjectWords, 4)).join(
          "\n\n",
        );
        writeFileSync(file, kind === "md" ? `# ${title}\n\n${body}\n` : `${title}\n\n${body}\n`);
      }
      written++;
    }
  }

  return {
    root,
    formats,
    scale,
    names: {
      pdf: "Tide report",
      bigPdf: "Annual survey (300 pages)",
      docx: "Coastal Flood Risk Review",
      imageDocx: "Seagrass field study (figures)",
      xlsx: "Regional Revenue",
      bigXlsx: "Transect measurements (50k cells)",
      csv: "Orders",
      pptx: "Quarterly Research Update",
      md: "Reading notes",
      txt: "Interview transcript",
      chinese: "潮汐与港口 · 研究笔记",
      long: LONG_NAME,
    },
    scaleCount: written,
  };
}
