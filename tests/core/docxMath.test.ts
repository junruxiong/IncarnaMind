import { describe, expect, test } from "vitest";
import { latexToOmml, MATH_NAMESPACE } from "../../src/core/exports/math";
import { checkWellFormed } from "../helpers/docx";

/** The OMML for some LaTeX, checked to be well-formed XML; it must convert. */
function omml(latex: string, display = false): string {
  const xml = latexToOmml(latex, display);
  if (xml === null) throw new Error(`${latex} didn't convert.`);
  checkWellFormed(
    `<root xmlns:m="${MATH_NAMESPACE}" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">${xml}</root>`,
  );
  return xml;
}

/** The text of a math run, as Word reads it. */
const runs = (xml: string) =>
  [...xml.matchAll(/<m:t xml:space="preserve">([^<]*)<\/m:t>/g)].map((m) => m[1]);

/** OMML with its runs' properties left out, to read its structure. */
const shape = (xml: string) =>
  xml
    .replace(/<w:rPr>[\s\S]*?<\/w:rPr>/g, "")
    .replace(/<m:rPr>[\s\S]*?<\/m:rPr>/g, "")
    .replace(/<m:r><m:t xml:space="preserve">([^<]*)<\/m:t><\/m:r>/g, "[$1]");

describe("LaTeX as a Word equation (OMML)", () => {
  test("letters, operators and a superscript", () => {
    const xml = omml("E = mc^2");
    expect(xml.startsWith("<m:oMath>")).toBe(true);
    expect(xml.endsWith("</m:oMath>")).toBe(true);
    expect(shape(xml)).toBe(
      "<m:oMath>[E][=][m]<m:sSup><m:e>[c]</m:e><m:sup>[2]</m:sup></m:sSup></m:oMath>",
    );
    // Runs are in Word's maths font.
    expect(xml).toContain(
      '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>',
    );
  });

  test("fractions, roots, subscripts and both scripts", () => {
    expect(shape(omml("\\frac{a}{b}"))).toBe(
      "<m:oMath><m:f><m:num>[a]</m:num><m:den>[b]</m:den></m:f></m:oMath>",
    );
    expect(shape(omml("\\binom{n}{k}"))).toContain('<m:fPr><m:type m:val="noBar"/></m:fPr>');
    expect(shape(omml("\\sqrt{x}"))).toBe(
      '<m:oMath><m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg></m:deg><m:e>[x]</m:e></m:rad></m:oMath>',
    );
    expect(shape(omml("\\sqrt[3]{x}"))).toBe(
      "<m:oMath><m:rad><m:deg>[3]</m:deg><m:e>[x]</m:e></m:rad></m:oMath>",
    );
    expect(shape(omml("x_i^2"))).toBe(
      "<m:oMath><m:sSubSup><m:e>[x]</m:e><m:sub>[i]</m:sub><m:sup>[2]</m:sup></m:sSubSup></m:oMath>",
    );
  });

  test("sums and integrals are n-ary operators over what follows them", () => {
    expect(shape(omml("\\sum_{i=1}^n x_i", true))).toBe(
      '<m:oMath><m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr><m:sub>[i][=][1]</m:sub><m:sup>[n]</m:sup><m:e><m:sSub><m:e>[x]</m:e><m:sub>[i]</m:sub></m:sSub></m:e></m:nary></m:oMath>',
    );
    const integral = shape(omml("\\int_0^\\infty e^{-x}\\,dx"));
    expect(integral).toContain(
      '<m:nary><m:naryPr><m:chr m:val="∫"/><m:limLoc m:val="subSup"/></m:naryPr><m:sub>[0]</m:sub><m:sup>[∞]</m:sup><m:e><m:sSup><m:e>[e]</m:e><m:sup>[−][x]</m:sup></m:sSup></m:e></m:nary>',
    );
    // The rest follows it: a thin space, then dx.
    expect(integral).toMatch(/<\/m:nary>\[.\]\[d\]\[x\]<\/m:oMath>$/u);
    // A limit left out is hidden.
    expect(shape(omml("\\oint_C f"))).toContain(
      '<m:naryPr><m:chr m:val="∮"/><m:limLoc m:val="subSup"/><m:supHide m:val="1"/></m:naryPr><m:sub>[C]</m:sub><m:sup></m:sup><m:e>[f]</m:e>',
    );
  });

  test("brackets that stretch, and matrices", () => {
    expect(shape(omml("\\left( \\frac{1}{2} \\right)"))).toBe(
      '<m:oMath><m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e><m:f><m:num>[1]</m:num><m:den>[2]</m:den></m:f></m:e></m:d></m:oMath>',
    );
    expect(shape(omml("\\left. x \\right|_0^1"))).toContain(
      '<m:d><m:dPr><m:begChr m:val=""/><m:endChr m:val="∣"/></m:dPr><m:e>[x]</m:e></m:d>',
    );
    expect(shape(omml("\\left( a \\middle| b \\right)"))).toContain(
      '<m:dPr><m:begChr m:val="("/><m:sepChr m:val="|"/><m:endChr m:val=")"/></m:dPr><m:e>[a]</m:e><m:e>[b]</m:e>',
    );
    const matrix = shape(omml("\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}"));
    expect(matrix).toContain(
      '<m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e><m:m>',
    );
    expect(matrix).toContain(
      "<m:mr><m:e>[a]</m:e><m:e>[b]</m:e></m:mr><m:mr><m:e>[c]</m:e><m:e>[d]</m:e></m:mr></m:m>",
    );
    // Cases are left-aligned columns after a brace.
    expect(shape(omml("\\begin{cases} 1 & x > 0 \\\\ 0 & \\text{else} \\end{cases}"))).toContain(
      '<m:mcs><m:mc><m:mcPr><m:count m:val="1"/><m:mcJc m:val="left"/></m:mcPr></m:mc><m:mc><m:mcPr><m:count m:val="1"/><m:mcJc m:val="left"/></m:mcPr></m:mc></m:mcs>',
    );
  });

  test("accents, bars, braces and boxes", () => {
    // Accents are Word's combining marks: a circumflex, an arrow.
    expect(shape(omml("\\hat{x}"))).toBe(
      `<m:oMath><m:acc><m:accPr><m:chr m:val="̂"/></m:accPr><m:e>[x]</m:e></m:acc></m:oMath>`,
    );
    expect(shape(omml("\\vec{v}"))).toContain(`<m:chr m:val="⃗"/>`);
    expect(shape(omml("\\bar{x}"))).toContain(`<m:chr m:val="̄"/>`);
    expect(shape(omml("\\overline{x+y}"))).toBe(
      '<m:oMath><m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>[x][+][y]</m:e></m:bar></m:oMath>',
    );
    expect(shape(omml("\\underline{x}"))).toContain(
      '<m:bar><m:barPr><m:pos m:val="bot"/></m:barPr>',
    );
    expect(shape(omml("\\underbrace{a+b}_{n}"))).toBe(
      '<m:oMath><m:limLow><m:e><m:groupChr><m:groupChrPr><m:chr m:val="⏟"/><m:pos m:val="bot"/><m:vertJc m:val="top"/></m:groupChrPr><m:e>[a][+][b]</m:e></m:groupChr></m:e><m:lim>[n]</m:lim></m:limLow></m:oMath>',
    );
    expect(shape(omml("\\boxed{x}"))).toBe(
      "<m:oMath><m:borderBox><m:e>[x]</m:e></m:borderBox></m:oMath>",
    );
    expect(shape(omml("\\lim_{x \\to 0} f(x)", true))).toContain(
      "<m:limLow><m:e>[lim]</m:e><m:lim>[x][→][0]</m:lim></m:limLow>",
    );
  });

  test("letters keep their style: upright names, double-struck, bold, and text", () => {
    const xml = omml("\\sin x + \\mathbb{R} + \\mathbf{v} + \\text{if } y");
    expect(xml).toContain(
      '<m:r><m:rPr><m:sty m:val="p"/></m:rPr><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr><m:t xml:space="preserve">sin</m:t></m:r>',
    );
    expect(xml).toContain(
      '<m:rPr><m:scr m:val="double-struck"/><m:sty m:val="p"/></m:rPr><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr><m:t xml:space="preserve">R</m:t>',
    );
    expect(xml).toContain('<m:rPr><m:sty m:val="b"/></m:rPr>');
    // Text in maths is the document's text, spaces and all.
    // (KaTeX keeps a space in text as a no-break space.)
    expect(xml).toMatch(
      /<m:r><m:rPr><m:nor\/><\/m:rPr><m:t xml:space="preserve">if[  ]<\/m:t><\/m:r>/,
    );
    // The invisible function application after "sin" isn't written.
    expect(xml).not.toContain("\u2061");
  });

  test("characters XML reserves are escaped", () => {
    expect(runs(omml("a < b \\& c > d"))).toEqual(["a", "&lt;", "b", "&amp;", "c", "&gt;", "d"]);
  });

  test("LaTeX it can't convert gives nothing, so the export keeps it as text", () => {
    expect(latexToOmml("x \\undefinedmacro", false)).toBeNull();
    expect(latexToOmml("\\frac{1}{", false)).toBeNull();
    expect(latexToOmml("\\href{https://example.com}{x}", false)).toBeNull();
    expect(latexToOmml("", false)).toBeNull();
    expect(latexToOmml("   ", true)).toBeNull();
  });
});
