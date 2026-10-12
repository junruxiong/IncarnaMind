#!/usr/bin/env python3
"""Writes the synthetic .docx files of the Word corpus (eval/word/README.md) into documents/synthetic/.

Standard library only, and deterministic: the same script writes the same bytes. Every file is
invented text with invented names; nothing in it is real contract language or a real person.

    python3 eval/word/generate-synthetic.py
"""
import os
import sys
import zipfile
from xml.sax.saxutils import escape

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents", "synthetic")
W = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" mc:Ignorable="w14"'
)
DATE = "2026-01-15T09:00:00Z"
XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'


def t(text):
    return '<w:t xml:space="preserve">%s</w:t>' % escape(text)


def run(text, b=False, i=False, u=False, size=None, rstyle=None, deleted=False):
    props = ""
    if rstyle:
        props += '<w:rStyle w:val="%s"/>' % rstyle
    if b:
        props += "<w:b/>"
    if i:
        props += "<w:i/>"
    if u:
        props += '<w:u w:val="single"/>'
    if size:
        props += '<w:sz w:val="%d"/>' % size
    rpr = "<w:rPr>%s</w:rPr>" % props if props else ""
    tag = (
        '<w:delText xml:space="preserve">%s</w:delText>' % escape(text) if deleted else t(text)
    )
    return "<w:r>%s%s</w:r>" % (rpr, tag)


class Doc:
    def __init__(self, lang="en"):
        self.lang = lang
        self.body = []
        self.footnotes = []
        self.endnotes = []
        self.comments = []
        self.rev = 100
        self.para = 0
        self.headers = {}
        self.footers = {}
        self.track = False
        self.sect_extra = ""
        self.bookmarks = 0

    # ---- inline pieces (return strings)
    def r(self, text, **kw):
        return run(text, **kw)

    def footnote(self, text):
        self.footnotes.append(text)
        n = len(self.footnotes)
        return (
            '<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr>'
            '<w:footnoteReference w:id="%d"/></w:r>' % n
        )

    def endnote(self, text):
        self.endnotes.append(text)
        n = len(self.endnotes)
        return (
            '<w:r><w:rPr><w:rStyle w:val="EndnoteReference"/></w:rPr>'
            '<w:endnoteReference w:id="%d"/></w:r>' % n
        )

    def field(self, instr, result="1"):
        return (
            '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            '<w:r><w:instrText xml:space="preserve"> %s </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            "<w:r>%s</w:r>"
            '<w:r><w:fldChar w:fldCharType="end"/></w:r>' % (escape(instr), t(result))
        )

    def comment(self, author, text, inner):
        cid = len(self.comments)
        self.comments.append((author, text))
        return (
            '<w:commentRangeStart w:id="%d"/>%s<w:commentRangeEnd w:id="%d"/>'
            '<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr>'
            '<w:commentReference w:id="%d"/></w:r>' % (cid, inner, cid, cid)
        )

    def ins(self, author, inner):
        self.rev += 1
        return '<w:ins w:id="%d" w:author="%s" w:date="%s">%s</w:ins>' % (
            self.rev,
            escape(author),
            DATE,
            inner,
        )

    def delete(self, author, text, **kw):
        self.rev += 1
        return '<w:del w:id="%d" w:author="%s" w:date="%s">%s</w:del>' % (
            self.rev,
            escape(author),
            DATE,
            run(text, deleted=True, **kw),
        )

    def format_change(self, author, text):
        self.rev += 1
        return (
            "<w:r><w:rPr><w:b/>"
            '<w:rPrChange w:id="%d" w:author="%s" w:date="%s"><w:rPr/></w:rPrChange>'
            "</w:rPr>%s</w:r>" % (self.rev, escape(author), DATE, t(text))
        )

    def bookmark(self, name, inner):
        self.bookmarks += 1
        return '<w:bookmarkStart w:id="%d" w:name="%s"/>%s<w:bookmarkEnd w:id="%d"/>' % (
            self.bookmarks,
            name,
            inner,
            self.bookmarks,
        )

    def control(self, alias, tag, text):
        self.bookmarks += 1
        return (
            "<w:sdt><w:sdtPr><w:alias w:val=\"%s\"/><w:tag w:val=\"%s\"/>"
            '<w:id w:val="%d"/><w:text/></w:sdtPr><w:sdtContent>%s</w:sdtContent></w:sdt>'
            % (escape(alias), tag, 4000 + self.bookmarks, run(text))
        )

    # ---- blocks
    def p(self, content="", style=None, num=None, align=None, section=None, keep=False):
        if isinstance(content, str) and not content.startswith("<"):
            content = run(content) if content else ""
        self.para += 1
        ppr = ""
        if style:
            ppr += '<w:pStyle w:val="%s"/>' % style
        if num:
            ppr += '<w:numPr><w:ilvl w:val="%d"/><w:numId w:val="%d"/></w:numPr>' % (
                num[1],
                num[0],
            )
        if align:
            ppr += '<w:jc w:val="%s"/>' % align
        if section:
            ppr += section
        ppr = "<w:pPr>%s</w:pPr>" % ppr if ppr else ""
        self.body.append(
            '<w:p w14:paraId="%08X" w14:textId="77777777">%s%s</w:p>' % (0x10000000 + self.para, ppr, content)
        )

    def table(self, rows, widths, header=True, merges=None):
        """rows: list of lists of cell text or (text, span)."""
        grid = "".join('<w:gridCol w:w="%d"/>' % w for w in widths)
        out = [
            '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="%d" w:type="dxa"/>'
            '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" '
            'w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>%s</w:tblGrid>'
            % (sum(widths), grid)
        ]
        for ri, row in enumerate(rows):
            out.append("<w:tr>")
            if ri == 0 and header:
                out[-1] = "<w:tr><w:trPr><w:tblHeader/></w:trPr>"
            col = 0
            for cell in row:
                span = 1
                vmerge = ""
                if isinstance(cell, tuple):
                    cell, span = cell[0], cell[1]
                    if span == 0:
                        vmerge = "<w:vMerge/>"
                        span = 1
                    elif span < 0:
                        vmerge = '<w:vMerge w:val="restart"/>'
                        span = 1
                w = sum(widths[col : col + span])
                grid_span = '<w:gridSpan w:val="%d"/>' % span if span > 1 else ""
                bold = ri == 0 and header
                inner = run(cell, b=bold) if cell else ""
                out.append(
                    '<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>%s%s</w:tcPr>'
                    "<w:p>%s</w:p></w:tc>" % (w, grid_span, vmerge, inner)
                )
                col += span
            out.append("</w:tr>")
        out.append("</w:tbl>")
        self.body.append("".join(out))
        self.body.append("<w:p/>")

    # ---- headers and footers
    def set_header(self, content, kind="default"):
        self.headers[kind] = content

    def set_footer(self, content, kind="default"):
        self.footers[kind] = content

    def section(self, first=False, last=False):
        refs = ""
        ids = 10
        for kind in self.headers:
            refs += '<w:headerReference w:type="%s" r:id="rIdH%s"/>' % (kind, kind)
        for kind in self.footers:
            refs += '<w:footerReference w:type="%s" r:id="rIdF%s"/>' % (kind, kind)
        title = "<w:titlePg/>" if "first" in self.headers or "first" in self.footers else ""
        return (
            "<w:sectPr>%s<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
            'w:header="708" w:footer="708" w:gutter="0"/>%s</w:sectPr>' % (refs, title)
        )

    # ---- package
    def save(self, name):
        zh = self.lang == "zh"
        rels = [
            ("rIdStyles", "styles", "styles.xml"),
            ("rIdNum", "numbering", "numbering.xml"),
            ("rIdSet", "settings", "settings.xml"),
            ("rIdFn", "footnotes", "footnotes.xml"),
            ("rIdEn", "endnotes", "endnotes.xml"),
        ]
        types = [
            ("/word/document.xml", "document.main"),
            ("/word/styles.xml", "styles"),
            ("/word/numbering.xml", "numbering"),
            ("/word/settings.xml", "settings"),
            ("/word/footnotes.xml", "footnotes"),
            ("/word/endnotes.xml", "endnotes"),
        ]
        files = {}
        files["word/styles.xml"] = styles_xml(zh)
        files["word/numbering.xml"] = numbering_xml()
        files["word/settings.xml"] = (
            XML + '<w:settings %s>%s<w:defaultTabStop w:val="720"/>'
            '<w:footnotePr><w:footnote w:id="-1"/><w:footnote w:id="0"/></w:footnotePr>'
            '<w:endnotePr><w:endnote w:id="-1"/><w:endnote w:id="0"/></w:endnotePr>'
            '<w:compat><w:compatSetting w:name="compatibilityMode" '
            'w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat>'
            "</w:settings>" % (W, "<w:trackRevisions/>" if self.track else "")
        )
        files["word/footnotes.xml"] = notes_xml("footnote", self.footnotes)
        files["word/endnotes.xml"] = notes_xml("endnote", self.endnotes)
        if self.comments:
            rels.append(("rIdCm", "comments", "comments.xml"))
            types.append(("/word/comments.xml", "comments"))
            files["word/comments.xml"] = comments_xml(self.comments)
        for kind, content in self.headers.items():
            fn = "header_%s.xml" % kind
            rels.append(("rIdH" + kind, "header", fn))
            types.append(("/word/" + fn, "header"))
            files["word/" + fn] = (
                XML + '<w:hdr %s><w:p><w:pPr><w:pStyle w:val="Header"/></w:pPr>%s</w:p></w:hdr>' % (W, content)
            )
        for kind, content in self.footers.items():
            fn = "footer_%s.xml" % kind
            rels.append(("rIdF" + kind, "footer", fn))
            types.append(("/word/" + fn, "footer"))
            files["word/" + fn] = (
                XML + '<w:ftr %s><w:p><w:pPr><w:pStyle w:val="Footer"/></w:pPr>%s</w:p></w:ftr>' % (W, content)
            )
        files["word/document.xml"] = (
            XML + "<w:document %s><w:body>%s%s</w:body></w:document>"
            % (W, "".join(self.body), self.section())
        )
        rel_base = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
        files["word/_rels/document.xml.rels"] = (
            XML
            + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                '<Relationship Id="%s" Type="%s%s" Target="%s"/>' % (i, rel_base, k, tg)
                for i, k, tg in rels
            )
            + "</Relationships>"
        )
        ct = {
            "document.main": "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
            "styles": "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml",
            "numbering": "application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml",
            "settings": "application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml",
            "footnotes": "application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml",
            "endnotes": "application/vnd.openxmlformats-officedocument.wordprocessingml.endnotes+xml",
            "comments": "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml",
            "header": "application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml",
            "footer": "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml",
        }
        files["[Content_Types].xml"] = (
            XML
            + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            + "".join('<Override PartName="%s" ContentType="%s"/>' % (p, ct[k]) for p, k in types)
            + '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            "</Types>"
        )
        files["_rels/.rels"] = (
            XML
            + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            "</Relationships>"
        )
        files["docProps/core.xml"] = (
            XML
            + '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            "<dc:title>%s</dc:title><dc:creator>Synthetic corpus</dc:creator>"
            "<cp:lastModifiedBy>Synthetic corpus</cp:lastModifiedBy>"
            '<dcterms:created xsi:type="dcterms:W3CDTF">%s</dcterms:created>'
            '<dcterms:modified xsi:type="dcterms:W3CDTF">%s</dcterms:modified>'
            "</cp:coreProperties>" % (escape(name), DATE, DATE)
        )
        files["docProps/app.xml"] = (
            XML
            + '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
            "<Application>Synthetic corpus generator</Application></Properties>"
        )
        order = ["[Content_Types].xml", "_rels/.rels"] + sorted(
            k for k in files if k not in ("[Content_Types].xml", "_rels/.rels")
        )
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, name + ".docx")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for k in order:
                info = zipfile.ZipInfo(k, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, files[k].encode("utf-8"))
        print("wrote", path)


def notes_xml(kind, notes):
    tag = kind
    ref = "footnoteRef" if kind == "footnote" else "endnoteRef"
    style = "FootnoteText" if kind == "footnote" else "EndnoteText"
    rstyle = "FootnoteReference" if kind == "footnote" else "EndnoteReference"
    out = [
        XML,
        "<w:%ss %s>" % (tag, W),
        '<w:%s w:type="separator" w:id="-1"><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:r><w:separator/></w:r></w:p></w:%s>' % (tag, tag),
        '<w:%s w:type="continuationSeparator" w:id="0"><w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:r><w:continuationSeparator/></w:r></w:p></w:%s>' % (tag, tag),
    ]
    for n, text in enumerate(notes, 1):
        out.append(
            '<w:%s w:id="%d"><w:p><w:pPr><w:pStyle w:val="%s"/></w:pPr>'
            '<w:r><w:rPr><w:rStyle w:val="%s"/></w:rPr><w:%s/></w:r>%s</w:p></w:%s>'
            % (tag, n, style, rstyle, ref, run(" " + text), tag)
        )
    out.append("</w:%ss>" % tag)
    return "".join(out)


def comments_xml(comments):
    out = [XML, "<w:comments %s>" % W]
    for cid, (author, text) in enumerate(comments):
        initials = "".join(w[0] for w in author.split())[:3] or "C"
        out.append(
            '<w:comment w:id="%d" w:author="%s" w:date="%s" w:initials="%s">'
            '<w:p><w:pPr><w:pStyle w:val="CommentText"/></w:pPr>'
            '<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:annotationRef/></w:r>%s</w:p>'
            "</w:comment>" % (cid, escape(author), DATE, escape(initials), run(text))
        )
    out.append("</w:comments>")
    return "".join(out)


def numbering_xml():
    def lvl(i, fmt, text, left, hanging, bold=False):
        return (
            '<w:lvl w:ilvl="%d"><w:start w:val="1"/><w:numFmt w:val="%s"/><w:lvlText w:val="%s"/>'
            '<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="%d" w:hanging="%d"/></w:pPr>%s</w:lvl>'
            % (i, fmt, text, left, hanging, "<w:rPr><w:b/></w:rPr>" if bold else "")
        )

    clauses = "".join(
        [
            lvl(0, "decimal", "%1.", 567, 567, True),
            lvl(1, "decimal", "%1.%2", 1134, 567),
            lvl(2, "lowerLetter", "(%3)", 1701, 567),
            lvl(3, "lowerRoman", "(%4)", 2268, 567),
        ]
    )
    bullets = "".join(
        [
            '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl>',
            '<w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="–"/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="1440" w:hanging="360"/></w:pPr></w:lvl>',
        ]
    )
    chinese = "".join(
        [
            lvl(0, "chineseCounting", "第%1条", 0, 0, True),
            lvl(1, "decimal", "%1.%2", 567, 567),
            lvl(2, "decimal", "(%3)", 1134, 567),
        ]
    )
    chinese_outline = "".join(
        [
            lvl(0, "chineseCounting", "%1、", 480, 480, True),
            lvl(1, "chineseCounting", "（%2）", 960, 480),
            lvl(2, "decimal", "%3.", 1440, 480),
        ]
    )
    heading = "".join(
        [
            lvl(0, "decimal", "%1", 432, 432),
            lvl(1, "decimal", "%1.%2", 576, 576),
            lvl(2, "decimal", "%1.%2.%3", 720, 720),
        ]
    )
    nums = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)]
    return (
        XML
        + "<w:numbering %s>" % W
        + '<w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="multilevel"/>%s</w:abstractNum>' % clauses
        + '<w:abstractNum w:abstractNumId="1"><w:multiLevelType w:val="hybridMultilevel"/>%s</w:abstractNum>' % bullets
        + '<w:abstractNum w:abstractNumId="2"><w:multiLevelType w:val="multilevel"/>%s</w:abstractNum>' % chinese
        + '<w:abstractNum w:abstractNumId="3"><w:multiLevelType w:val="multilevel"/>%s</w:abstractNum>' % chinese_outline
        + '<w:abstractNum w:abstractNumId="4"><w:multiLevelType w:val="multilevel"/>%s</w:abstractNum>' % heading
        + "".join('<w:num w:numId="%d"><w:abstractNumId w:val="%d"/></w:num>' % n for n in nums)
        + "</w:numbering>"
    )


def styles_xml(zh):
    fonts = (
        '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="SimSun" w:cs="Times New Roman"/>'
        if zh
        else '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="SimSun" w:cs="Times New Roman"/>'
    )
    lang = '<w:lang w:val="en-GB" w:eastAsia="zh-CN"/>' if not zh else '<w:lang w:val="en-GB" w:eastAsia="zh-CN"/>'

    def para(sid, name, ppr="", rpr="", based="Normal", nxt=None):
        return (
            '<w:style w:type="paragraph" w:styleId="%s"><w:name w:val="%s"/>%s%s'
            "<w:qFormat/>%s%s</w:style>"
            % (
                sid,
                name,
                '<w:basedOn w:val="%s"/>' % based if based else "",
                '<w:next w:val="%s"/>' % nxt if nxt else "",
                "<w:pPr>%s</w:pPr>" % ppr if ppr else "",
                "<w:rPr>%s</w:rPr>" % rpr if rpr else "",
            )
        )

    def char(sid, name, rpr):
        return (
            '<w:style w:type="character" w:styleId="%s"><w:name w:val="%s"/><w:rPr>%s</w:rPr></w:style>'
            % (sid, name, rpr)
        )

    return (
        XML
        + "<w:styles %s>" % W
        + "<w:docDefaults><w:rPrDefault><w:rPr>%s<w:sz w:val=\"22\"/><w:szCs w:val=\"22\"/>%s</w:rPr></w:rPrDefault>"
        '<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>'
        % (fonts, lang)
        + '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>'
        + '<w:style w:type="character" w:default="1" w:styleId="DefaultParagraphFont"><w:name w:val="Default Paragraph Font"/><w:uiPriority w:val="1"/><w:semiHidden/></w:style>'
        + '<w:style w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/><w:semiHidden/><w:tblPr><w:tblInd w:w="0" w:type="dxa"/><w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/><w:bottom w:w="0" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>'
        + '<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:basedOn w:val="TableNormal"/><w:tblPr><w:tblBorders>'
        + "".join(
            '<w:%s w:val="single" w:sz="4" w:space="0" w:color="808080"/>' % s
            for s in ("top", "left", "bottom", "right", "insideH", "insideV")
        )
        + "</w:tblBorders></w:tblPr></w:style>"
        + para("Title", "Title", '<w:spacing w:before="240" w:after="240"/><w:jc w:val="center"/>', '<w:b/><w:sz w:val="40"/>', nxt="Normal")
        + para("Heading1", "heading 1", '<w:keepNext/><w:spacing w:before="360" w:after="120"/><w:outlineLvl w:val="0"/>', '<w:b/><w:sz w:val="32"/>', nxt="Normal")
        + para("Heading2", "heading 2", '<w:keepNext/><w:spacing w:before="240" w:after="80"/><w:outlineLvl w:val="1"/>', '<w:b/><w:sz w:val="26"/>', nxt="Normal")
        + para("Heading3", "heading 3", '<w:keepNext/><w:spacing w:before="200" w:after="60"/><w:outlineLvl w:val="2"/>', '<w:b/><w:i/><w:sz w:val="24"/>', nxt="Normal")
        + para("ListParagraph", "List Paragraph", '<w:ind w:left="720"/>')
        + para("Header", "header", '<w:tabs><w:tab w:val="right" w:pos="9026"/></w:tabs><w:spacing w:after="0"/>', "<w:sz w:val=\"18\"/>")
        + para("Footer", "footer", '<w:tabs><w:tab w:val="right" w:pos="9026"/></w:tabs><w:spacing w:after="0"/>', "<w:sz w:val=\"18\"/>")
        + para("FootnoteText", "footnote text", '<w:spacing w:after="0" w:line="240" w:lineRule="auto"/>', "<w:sz w:val=\"20\"/>")
        + para("EndnoteText", "endnote text", '<w:spacing w:after="0" w:line="240" w:lineRule="auto"/>', "<w:sz w:val=\"20\"/>")
        + para("CommentText", "annotation text", "", "<w:sz w:val=\"20\"/>")
        + para("TOC1", "toc 1", '<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9016"/></w:tabs>')
        + char("FootnoteReference", "footnote reference", '<w:vertAlign w:val="superscript"/>')
        + char("EndnoteReference", "endnote reference", '<w:vertAlign w:val="superscript"/>')
        + char("CommentReference", "annotation reference", '<w:sz w:val="16"/>')
        + char("Hyperlink", "Hyperlink", '<w:color w:val="0563C1"/><w:u w:val="single"/>')
        + "</w:styles>"
    )


# ------------------------------------------------------------------ documents
CL = (1, None)  # numId 1: English clauses


def clause(d, text, level=0, num=1, inner=None):
    d.p(inner if inner is not None else text, num=(num, level))


def page_footer(d, label="Page", of="of"):
    d.set_footer(
        run(label + " ") + d.field("PAGE", "1") + run(" " + of + " ") + d.field("NUMPAGES", "3")
    )


def en_contract_services():
    d = Doc()
    d.set_header(run("Master Services Agreement") + "<w:r><w:tab/></w:r>" + run("Draft for review"))
    page_footer(d)
    d.p("Master Services Agreement", "Title")
    d.p(
        run("This agreement is made on 15 January 2026 between ")
        + d.control("Client", "client", "Northwind Example Ltd")
        + run(" (the Client) and ")
        + d.control("Supplier", "supplier", "Contoso Sample Services Ltd")
        + run(" (the Supplier).")
    )
    d.p(d.field('TOC \\o "1-2" \\h \\z \\u', "1. Definitions  2. Services  3. Fees  4. Liability  5. Term"), "TOC1")
    d.p("Definitions", "Heading1", num=(1, 0))
    clause(d, "", 1, inner=run('"Services" means the services described in Schedule 1.') + d.footnote("Schedule 1 is attached to the signed copy of this agreement."))
    clause(d, "", 1, inner=run('"Fees" means the amounts set out in clause ') + d.bookmark("fees_ref", run("3")) + run("."))
    clause(d, "", 1, inner=run('"Business Day" means a day other than a Saturday, Sunday or public holiday in England.'))
    d.p("Services", "Heading1", num=(1, 0))
    clause(d, "The Supplier shall provide the Services with reasonable skill and care.", 1)
    clause(d, "The Supplier shall:", 1)
    clause(d, "meet the milestones in Schedule 1;", 2)
    clause(d, "report progress each month; and", 2)
    clause(d, "tell the Client at once if a milestone is at risk.", 2)
    clause(d, "The Client shall give the Supplier the access and information it reasonably needs.", 1)
    d.p("Fees", "Heading1", num=(1, 0))
    clause(d, "The Client shall pay the Fees within 30 days of an invoice.", 1)
    d.table(
        [["Milestone", "Due", "Fee (GBP)"], ["Discovery", "28 February 2026", "12,000"], ["Build", "30 April 2026", "40,000"], ["Handover", "30 June 2026", "8,000"], [("Total", 2), "60,000"]],
        [3200, 3000, 2800],
    )
    d.p("Liability", "Heading1", num=(1, 0))
    clause(d, "", 1, inner=run("Neither party limits liability for death or personal injury caused by negligence.") + d.endnote("This carve-out follows the usual statutory position and is shown for the exercise only."))
    clause(d, "Each party's total liability is limited to the Fees paid in the preceding 12 months.", 1)
    d.p("Term", "Heading1", num=(1, 0))
    clause(d, "This agreement starts on the date above and runs for 12 months.", 1)
    d.table([["Signed for the Client", "Signed for the Supplier"], ["Name:", "Name:"], ["Date:", "Date:"]], [4500, 4500])
    d.save("en-contract-services")


def en_contract_nda():
    d = Doc()
    d.set_header(run("Mutual confidentiality agreement"))
    page_footer(d)
    d.p("Mutual Confidentiality Agreement", "Title")
    d.p("The parties wish to share confidential information to discuss a possible project, and agree as follows.")
    d.p("Confidential information", "Heading1", num=(1, 0))
    clause(d, "", 1, inner=d.comment("Reviewer", "Should this cover oral disclosures too?", d.bookmark("conf_def", run("Confidential Information means all information disclosed by one party to the other, in any form."))))
    clause(d, "It does not include information that:", 1)
    clause(d, "is or becomes public through no fault of the recipient;", 2)
    clause(d, "was known to the recipient before disclosure; or", 2)
    clause(d, "", 2, inner=d.comment("Reviewer", "Add independent development here?", run("is lawfully received from a third party.")))
    d.p("Obligations", "Heading1", num=(1, 0))
    clause(d, "", 1, inner=run("The recipient shall keep Confidential Information secret and use it only for the project, as clause ") + d.field("REF conf_def \\r \\h", "1.1") + run(" requires."))
    clause(d, "The recipient may disclose it to staff who need it and are bound to keep it secret.", 1)
    d.p("Term", "Heading1", num=(1, 0))
    clause(d, "", 1, inner=d.comment("Counsel", "Five years is long for this kind of project.", run("The obligations last five years from the date of this agreement.")))
    d.save("en-contract-nda")


def en_contract_lease():
    d = Doc()
    d.set_header(run("Lease of Unit 4, Example Court") + "<w:r><w:tab/></w:r>" + d.field("DATE \\@ \"d MMMM yyyy\"", "15 January 2026"))
    page_footer(d)
    d.p("Commercial Lease", "Title")
    d.p("Particulars", "Heading1")
    d.table(
        [["Item", "Detail"], ["Landlord", "Fabrikam Example Estates Ltd"], ["Tenant", "Adventure Sample Works Ltd"], [("Premises", -1), "Unit 4, Example Court"], [("", 0), "Ground floor, 120 square metres"], ["Term", "5 years"]],
        [2600, 6400],
    )
    d.p("Rent", "Heading1")
    d.p(run("The tenant shall pay the rent in the table below") + d.footnote("Rent is exclusive of VAT, which the tenant also pays.") + run(", quarterly in advance."))
    d.p(run("Table ") + d.field("SEQ Table \\* ARABIC", "1") + run(": Rent schedule"), align="center")
    d.table([["Year", "Annual rent (GBP)"], ["1", "24,000"], ["2", "24,000"], ["3", "26,400"], ["4", "26,400"], ["5", "26,400"]], [2000, 3000])
    d.p("Repair", "Heading1")
    d.p("The tenant shall keep the premises in good repair, and shall not make structural changes.", num=(1, 1))
    d.p("The landlord shall insure the building and may recover a fair share of the cost from the tenant.", num=(1, 1))
    d.p(run("Break right") + d.endnote("The break clause needs six months' written notice."), "Heading1")
    d.p("Either party may end the lease at the end of the third year.")
    d.save("en-contract-lease")


def en_contract_redline():
    d = Doc()
    d.track = True
    A, B = "Counterparty Counsel", "Reviewer A"
    d.set_header(run("Supply Agreement: counterparty mark-up"))
    page_footer(d)
    d.p("Supply Agreement", "Title")
    d.p("Delivery", "Heading1", num=(1, 0))
    d.p(run("The Supplier shall deliver the goods within ") + d.delete(A, "ten") + d.ins(A, run("thirty")) + run(" days of each order."), num=(1, 1))
    d.p(run("Risk passes to the Buyer ") + d.delete(B, "on delivery") + d.ins(B, run("on signature of the delivery note")) + run("."), num=(1, 1))
    d.p("Payment", "Heading1", num=(1, 0))
    d.p(d.comment(A, "We need 60 days; our finance team pays monthly.", run("The Buyer shall pay within ") + d.delete(A, "14") + d.ins(A, run("60")) + run(" days of invoice.")), num=(1, 1))
    d.p(run("Late payment carries interest at ") + d.ins(B, run("4% above base rate")) + run("."), num=(1, 1))
    d.p("Warranty", "Heading1", num=(1, 0))
    d.p(run("The Supplier warrants that the goods are ") + d.format_change(A, "fit for purpose") + run(" and ") + d.delete(B, "free from defects for 24 months") + d.ins(B, run("free from defects for 12 months")) + run("."), num=(1, 1))
    d.p(d.ins(A, run("The Supplier shall replace defective goods at its own cost.")), num=(1, 1))
    d.save("en-contract-redline")


def en_memo_board():
    d = Doc()
    d.track = True
    A = "Finance Lead"
    d.set_header(run("Memorandum"))
    page_footer(d)
    d.p("Memorandum", "Title")
    d.table([["To", "The Board"], ["From", "Chief Operating Officer"], ["Date", "15 January 2026"], ["Subject", "Warehouse move"]], [1800, 7200], header=False)
    d.p("Summary", "Heading1")
    d.p(run("We propose to move the warehouse to the new site in ") + d.delete(A, "April") + d.ins(A, run("June")) + run(", at a one-off cost of ") + d.delete(A, "GBP 90,000") + d.ins(A, run("GBP 115,000")) + run("."))
    d.p("Reasons", "Heading1")
    for line in ["Lower rent per pallet", "Shorter delivery routes", "Room to grow for five years"]:
        d.p(line, "ListParagraph", num=(2, 0))
    d.p(d.ins(A, run("The extra cost covers racking that the first estimate left out.")))
    d.p("Decision needed", "Heading1")
    d.p("Please approve the move at the next meeting.")
    d.save("en-memo-board")


def en_memo_policy():
    d = Doc()
    d.track = True
    d.set_header(run("Policy memo") + "<w:r><w:tab/></w:r>" + run("Internal"))
    page_footer(d)
    d.p("Remote Working Policy: Memo", "Title")
    d.p("Background", "Heading1")
    d.p(run("Staff may work from home up to three days a week") + d.footnote("Based on the 2025 staff survey, in which 71% asked for this.") + run("."))
    d.p("Changes proposed", "Heading1")
    d.p(d.comment("Legal", "Check this against the employment terms.", run("Managers may refuse a request only for a business reason ") + d.ins("Legal", run("given in writing")) + run(".")))
    d.p(run("Equipment is ") + d.delete("HR", "bought by staff") + d.ins("HR", run("provided by the company")) + run("."))
    d.p(d.format_change("HR", "A review takes place after six months."))
    d.save("en-memo-policy")


def en_memo_plain():
    d = Doc()
    d.set_header(run("Office notice"))
    d.set_footer(run("Example Office"))
    d.p("Memo: Office closure", "Title")
    d.table([["To", "All staff"], ["From", "Office manager"], ["Date", "15 January 2026"]], [1800, 7200], header=False)
    d.p("The office will be closed on Monday 26 January for floor repairs. Please take your laptops home on Friday.")
    d.p("Contact the office manager with any questions.")
    d.save("en-memo-plain")


def en_report_quarterly():
    d = Doc()
    d.set_header(run("Quarterly Report, Q4 2025") + "<w:r><w:tab/></w:r>" + d.control("Company", "company", "Northwind Example Ltd"))
    page_footer(d)
    d.p("Quarterly Report", "Title")
    d.p(d.field('TOC \\o "1-3" \\h \\z \\u', "Overview 1  Results 2  Outlook 3"), "TOC1")
    d.p("Overview", "Heading1")
    d.p(run("Revenue grew 8% on the quarter") + d.footnote("Revenue is stated before refunds.") + run(" and costs were flat."))
    d.p(d.bookmark("results", run("Results")), "Heading1")
    d.p("Revenue by region", "Heading2")
    d.table([["Region", "Q3", "Q4", "Change"], ["North", "4.1", "4.5", "+10%"], ["South", "3.2", "3.3", "+3%"], ["Overseas", "1.9", "2.1", "+11%"]], [2400, 2000, 2000, 2000])
    d.p("Costs", "Heading2")
    d.p(run("Staff costs rose 2%, while hosting fell 6%") + d.endnote("The hosting contract was renegotiated in October."), )
    d.p("Outlook", "Heading1")
    d.p(run("See ") + d.field("REF results \\h", "Results") + run(" for the figures behind this."))
    d.save("en-report-quarterly")


def en_policy_numbered():
    d = Doc()
    d.set_header(run("Information Security Policy"))
    page_footer(d)
    d.p("Information Security Policy", "Title")
    for text, lvl, style in [
        ("Purpose", 0, "Heading1"),
        ("Scope", 1, "Heading2"),
        ("This policy applies to all staff and contractors.", None, None),
        ("Responsibilities", 1, "Heading2"),
        ("Managers", 2, "Heading3"),
        ("Managers make sure their teams complete security training each year.", None, None),
        ("Staff", 2, "Heading3"),
        ("Staff report lost devices within one hour.", None, None),
        ("Access", 0, "Heading1"),
        ("Passwords", 1, "Heading2"),
        ("Use a password manager and a second factor.", None, None),
    ]:
        if lvl is None:
            d.p(text)
        else:
            d.p(text, style, num=(5, lvl))
    d.save("en-policy-numbered")


def en_report_sections():
    d = Doc()
    d.set_header(run("First page header"), "first")
    d.set_header(run("Annual Review"), "default")
    d.set_footer(run("Confidential"), "first")
    d.set_footer(run("Page ") + d.field("PAGE", "2"), "default")
    d.p("Annual Review", "Title")
    d.p("Chapter 1", "Heading1")
    d.p("The first page has its own header and footer, as the section sets.")
    d.p("A page break comes next.")
    d.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    d.p("Chapter 2", "Heading1")
    d.p(run("Chapter two starts on a new page") + d.footnote("Footnotes restart on each page in some templates; here they run on."))
    d.save("en-report-sections")


# ---- Chinese
def zh_contract_service():
    d = Doc("zh")
    d.set_header(run("服务合同") + "<w:r><w:tab/></w:r>" + run("草稿"))
    page_footer(d, "第", "页，共")
    d.p("技术服务合同", "Title")
    d.p(run("甲方：") + d.control("甲方", "party_a", "示例北方科技有限公司") + run("　乙方：") + d.control("乙方", "party_b", "示例南方信息服务有限公司"))
    d.p("服务内容", "Heading1", num=(3, 0))
    d.p(run("乙方按附件一所列内容向甲方提供技术服务。") + d.footnote("附件一为双方签字盖章后的合同附件。"), num=(3, 1))
    d.p("乙方应当按时完成各阶段工作，并及时向甲方报告进度。", num=(3, 1))
    d.p("费用与支付", "Heading1", num=(3, 0))
    d.p("服务费共计人民币陆万元整，分三期支付：", num=(3, 1))
    d.table([["阶段", "时间", "金额（元）"], ["需求调研", "2026年2月28日", "12,000"], ["开发实施", "2026年4月30日", "40,000"], ["验收交付", "2026年6月30日", "8,000"]], [3000, 3200, 2800])
    d.p("违约责任", "Heading1", num=(3, 0))
    d.p(run("任何一方违约，应当赔偿对方因此遭受的损失。") + d.endnote("赔偿范围以实际发生的直接损失为限。"), num=(3, 1))
    d.p("争议解决", "Heading1", num=(3, 0))
    d.p("因本合同发生的争议，双方应协商解决；协商不成的，提交甲方所在地人民法院诉讼解决。", num=(3, 1))
    d.table([["甲方（盖章）", "乙方（盖章）"], ["代表签字：", "代表签字："], ["日期：", "日期："]], [4500, 4500])
    d.save("zh-contract-service")


def zh_contract_sale():
    d = Doc("zh")
    d.set_header(run("采购合同"))
    page_footer(d, "第", "页，共")
    d.p("设备采购合同", "Title")
    d.p("交货", "Heading1", num=(3, 0))
    d.p(d.comment("法务", "交货期限是否与附件的排期一致？", run("乙方应在合同签订后三十日内将设备送至甲方指定地点。")), num=(3, 1))
    d.p("验收", "Heading1", num=(3, 0))
    d.p(run("甲方应在收货后七日内完成验收。") + d.footnote("逾期未提出异议的，视为验收合格。"), num=(3, 1))
    d.p("付款", "Heading1", num=(3, 0))
    d.p(d.comment("财务", "付款比例需要和预算对齐。", run("甲方在验收合格后十五日内支付全部货款。")), num=(3, 1))
    d.save("zh-contract-sale")


def zh_contract_redline():
    d = Doc("zh")
    d.track = True
    A, B = "对方律师", "审核人甲"
    d.set_header(run("采购合同：对方修改稿"))
    page_footer(d, "第", "页，共")
    d.p("货物采购合同", "Title")
    d.p("交货", "Heading1", num=(3, 0))
    d.p(run("乙方应在订单发出后") + d.delete(A, "十") + d.ins(A, run("三十")) + run("日内交货。"), num=(3, 1))
    d.p("付款", "Heading1", num=(3, 0))
    d.p(d.comment(A, "我方财务按月付款。", run("甲方应在收到发票后") + d.delete(A, "十四") + d.ins(A, run("六十")) + run("日内付款。")), num=(3, 1))
    d.p("质量保证", "Heading1", num=(3, 0))
    d.p(run("质保期为") + d.delete(B, "二十四") + d.ins(B, run("十二")) + run("个月。") + d.ins(A, run("乙方应自费更换有缺陷的货物。")), num=(3, 1))
    d.save("zh-contract-redline")


def zh_memo_notice():
    d = Doc("zh")
    d.track = True
    d.set_header(run("通知"))
    d.set_footer(run("示例公司办公室"))
    d.p("关于仓库搬迁的通知", "Title")
    d.table([["收件人", "全体员工"], ["发件人", "运营部"], ["日期", "2026年1月15日"]], [1800, 7200], header=False)
    d.p(run("公司仓库将于") + d.delete("财务负责人", "四月") + d.ins("财务负责人", run("六月")) + run("搬至新址，请各部门提前做好准备。"))
    d.p("搬迁原因", "Heading1")
    for line in ["每托盘租金更低", "配送路线更短", "未来五年有扩展空间"]:
        d.p(line, "ListParagraph", num=(2, 0))
    d.save("zh-memo-notice")


def zh_memo_meeting():
    d = Doc("zh")
    d.set_header(run("会议纪要"))
    page_footer(d, "第", "页，共")
    d.p("项目例会纪要", "Title")
    d.table([["时间", "2026年1月15日 10:00"], ["地点", "三楼会议室"], ["参加人", "张三、李四、王五（示例人名）"]], [1800, 7200], header=False)
    d.p("议题", "Heading1")
    d.p("第一阶段进度", "ListParagraph", num=(2, 0))
    d.p("预算调整", "ListParagraph", num=(2, 0))
    d.p("行动项", "Heading1")
    d.table([["事项", "负责人", "期限"], ["提交进度报告", "张三", "1月20日"], ["更新预算表", "李四", "1月22日"], ["预订下次会议室", "王五", "1月18日"]], [4000, 2400, 2600])
    d.save("zh-memo-meeting")


def zh_report_annual():
    d = Doc("zh")
    d.set_header(run("年度报告") + "<w:r><w:tab/></w:r>" + d.field("DATE \\@ \"yyyy年M月d日\"", "2026年1月15日"))
    page_footer(d, "第", "页，共")
    d.p("2025年度报告", "Title")
    d.p(d.field('TOC \\o "1-3" \\h \\z \\u', "概述 1  经营情况 2  展望 3"), "TOC1")
    d.p("概述", "Heading1")
    d.p(run("全年营业收入同比增长百分之八。") + d.footnote("营业收入为扣除退款前的金额。"))
    d.p("经营情况", "Heading1")
    d.p("各地区收入", "Heading2")
    d.table([["地区", "第三季度", "第四季度", "变化"], ["北方", "4.1", "4.5", "+10%"], ["南方", "3.2", "3.3", "+3%"], ["海外", "1.9", "2.1", "+11%"]], [2400, 2000, 2000, 2000])
    d.p(run("人员成本上升百分之二，托管成本下降百分之六。") + d.endnote("托管合同已于十月重新议价。"))
    d.p("展望", "Heading1")
    d.p("明年将继续控制成本，并扩大海外业务。")
    d.save("zh-report-annual")


def zh_policy_rules():
    d = Doc("zh")
    d.set_header(run("信息安全管理制度"))
    page_footer(d, "第", "页，共")
    d.p("信息安全管理制度", "Title")
    d.p("总则", "Heading1", num=(4, 0))
    d.p("本制度适用于全体员工和外包人员。", num=(4, 1))
    d.p(d.comment("信息安全部", "外包人员是否需要单独的培训？", run("各部门负责人应保证本部门员工每年完成安全培训。")), num=(4, 1))
    d.p("账号与口令", "Heading1", num=(4, 0))
    d.p("口令应由密码管理器生成，并启用二次验证。", num=(4, 1))
    d.p("设备丢失应在一小时内报告。", num=(4, 2))
    d.p("附则", "Heading1", num=(4, 0))
    d.p("本制度自发布之日起施行。", num=(4, 1))
    d.save("zh-policy-rules")


def zh_bilingual_contract():
    d = Doc("zh")
    d.set_header(run("Bilingual Agreement / 双语协议"))
    page_footer(d, "Page / 第", "of / 共")
    d.p("Software Licence / 软件许可协议", "Title")
    d.table(
        [
            ["English", "中文"],
            ["The Licensor grants the Licensee a non-exclusive licence.", "许可方授予被许可方非独占许可。"],
            ["The licence lasts one year.", "许可期限为一年。"],
            ["Fees are USD 5,000 per year.", "许可费为每年5,000美元。"],
        ],
        [4500, 4500],
    )
    d.p(run("Notices go to ") + d.control("Contact", "contact", "legal@example.invalid") + run("，通知应发送至该邮箱。") + d.footnote("Example address; it does not exist. 示例地址，并不存在。"))
    d.p(d.comment("Counsel / 律师", "Which text prevails if they differ? 两种文本不一致时以哪种为准？", run("Both texts are equally authentic. 两种文本具有同等效力。")))
    d.save("zh-bilingual-contract")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith(("en_", "zh_")) and callable(fn):
            fn()


if __name__ == "__main__":
    sys.exit(main())
