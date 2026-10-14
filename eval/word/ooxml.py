"""A small OOXML writer for the synthetic files of the Word corpus (eval/word/README.md).

Standard library only, and deterministic: the same calls write the same bytes. Used by
generate-synthetic.py and generate-domains.py.
"""
import os
import struct
import zlib
import zipfile
from xml.sax.saxutils import escape

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents", "synthetic")
W = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" mc:Ignorable="w14"'
)
WX = W.replace(
    ' mc:Ignorable="w14"',
    ' xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
    ' xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office"'
    ' xmlns:w10="urn:schemas-microsoft-com:office:word"'
    ' xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"'
    ' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
    ' xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
    ' xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"'
    ' xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml"'
    ' mc:Ignorable="w14 w15"',
)
DATE = "2026-01-15T09:00:00Z"
XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'


def t(text):
    return '<w:t xml:space="preserve">%s</w:t>' % escape(text)


def run(text, b=False, i=False, u=False, size=None, rstyle=None, deleted=False, rtl=False):
    props = ""
    if rstyle:
        props += '<w:rStyle w:val="%s"/>' % rstyle
    if b:
        props += "<w:b/>"
    if i:
        props += "<w:i/>"
    if size:
        props += '<w:sz w:val="%d"/>' % size
    if u:
        props += '<w:u w:val="single"/>'
    if rtl:
        props += "<w:rtl/>"
    rpr = "<w:rPr>%s</w:rPr>" % props if props else ""
    tag = (
        '<w:delText xml:space="preserve">%s</w:delText>' % escape(text) if deleted else t(text)
    )
    return "<w:r>%s%s</w:r>" % (rpr, tag)


class Raw:
    """Block-level XML to put in a table cell."""

    def __init__(self, xml):
        self.xml = xml


def png(width, height, rgb):
    """A small PNG of a vertical gradient from `rgb` to white; the same arguments give the same bytes."""

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    rows = b""
    for y in range(height):
        mix = y / max(1, height - 1)
        pixel = bytes(int(c + (255 - c) * mix) for c in rgb)
        rows += b"\x00" + pixel * width
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )


# Office Math (OMML) pieces
def m_r(text):
    return '<m:r><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr><m:t>%s</m:t></m:r>' % escape(text)


def m_f(numerator, denominator):
    return "<m:f><m:num>%s</m:num><m:den>%s</m:den></m:f>" % (numerator, denominator)


def m_sup(base, sup):
    return "<m:sSup><m:e>%s</m:e><m:sup>%s</m:sup></m:sSup>" % (base, sup)


def m_sub(base, sub):
    return "<m:sSub><m:e>%s</m:e><m:sub>%s</m:sub></m:sSub>" % (base, sub)


def m_sum(lower, upper, body):
    return (
        '<m:nary><m:naryPr><m:chr m:val="\u2211"/><m:limLoc m:val="undOvr"/></m:naryPr>'
        "<m:sub>%s</m:sub><m:sup>%s</m:sup><m:e>%s</m:e></m:nary>" % (lower, upper, body)
    )


def m_rad(body):
    return '<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>%s</m:e></m:rad>' % body


def m_display(*parts):
    return "<m:oMathPara><m:oMath>%s</m:oMath></m:oMathPara>" % "".join(parts)


def m_inline(*parts):
    return "<m:oMath>%s</m:oMath>" % "".join(parts)


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
        self.ns = W
        self.rtl = lang in ("ar", "he")
        self.compat = 15
        self.pages = 0
        self.final_cols = 1
        self.extra_files = {}
        self.extra_rels = []
        self.extra_types = []
        self.extra_defaults = {}
        self.media = 0
        self.charts = 0
        self.sects = 0
        self.shapes = 0
        self.settings_extra = ""

    def extended(self):
        """Switches on the drawing, maths and comment-thread namespaces for this document."""
        self.ns = WX
        return self

    def run(self, text, **kw):
        return run(text, rtl=self.rtl, **kw)

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

    def comment(self, author, text, inner, replies=()):
        """A comment on `inner`; `replies` is a list of (author, text) in the same thread."""
        base = len(self.comments)
        self.comments.append((author, text, None))
        for reply_author, reply_text in replies:
            self.comments.append((reply_author, reply_text, base))
        ids = range(base, len(self.comments))
        starts = "".join('<w:commentRangeStart w:id="%d"/>' % i for i in ids)
        ends = "".join(
            '<w:commentRangeEnd w:id="%d"/>'
            '<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr>'
            '<w:commentReference w:id="%d"/></w:r>' % (i, i)
            for i in ids
        )
        return starts + inner + ends

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
            content = self.run(content) if content else ""
        self.para += 1
        ppr = ""
        if style:
            ppr += '<w:pStyle w:val="%s"/>' % style
        if num:
            ppr += '<w:numPr><w:ilvl w:val="%d"/><w:numId w:val="%d"/></w:numPr>' % (
                num[1],
                num[0],
            )
        if self.rtl:
            ppr += "<w:bidi/>"
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
        cols = '<w:cols w:num="%d" w:space="720"/>' % self.final_cols if self.final_cols > 1 else ""
        return (
            "<w:sectPr>%s<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
            'w:header="708" w:footer="708" w:gutter="0"/>%s%s</w:sectPr>' % (refs, cols, title)
        )

    # ---- richer pieces (documents made with .extended())
    def sect(self, header=None, footer=None, cols=1, landscape=False, kind="nextPage"):
        """The section properties that end a section: pass them to `p(section=...)`."""
        self.sects += 1
        n = self.sects
        refs = ""
        for what, content, tag in (("header", header, "hdr"), ("footer", footer, "ftr")):
            if content is None:
                continue
            name = "%s_s%d.xml" % (what, n)
            style = "Header" if what == "header" else "Footer"
            self.extra_files["word/" + name] = (
                XML + '<w:%s %s><w:p><w:pPr><w:pStyle w:val="%s"/></w:pPr>%s</w:p></w:%s>' % (tag, self.ns, style, content, tag)
            )
            self.extra_rels.append(("rIdS%s%d" % (what[0].upper(), n), what, name))
            self.extra_types.append(("/word/" + name, what))
            refs += '<w:%sReference w:type="default" r:id="rIdS%s%d"/>' % (what, what[0].upper(), n)
        size = (
            '<w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/>'
            if landscape
            else '<w:pgSz w:w="11906" w:h="16838"/>'
        )
        columns = '<w:cols w:num="%d" w:space="720"/>' % cols if cols > 1 else ""
        return (
            '<w:sectPr>%s<w:type w:val="%s"/>%s<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" '
            'w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>%s</w:sectPr>' % (refs, kind, size, columns)
        )

    def image(self, rgb=(60, 120, 200), size=(160, 100), wrap="square", alt="Figure"):
        """A picture; `wrap` is "square", "tight", "topAndBottom", or None for one in the line."""
        self.media += 1
        n = self.media
        self.extra_files["word/media/image%d.png" % n] = png(size[0], size[1], rgb)
        self.extra_defaults["png"] = "image/png"
        rid = "rIdImg%d" % n
        self.extra_rels.append((rid, "image", "media/image%d.png" % n))
        cx, cy = size[0] * 9525 * 2, size[1] * 9525 * 2
        graphic = (
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic><pic:nvPicPr><pic:cNvPr id="%d" name="image%d.png" descr="%s"/><pic:cNvPicPr/></pic:nvPicPr>'
            '<pic:blipFill><a:blip r:embed="%s"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
            % (n, n, escape(alt), rid, cx, cy)
        )
        doc_pr = '<wp:docPr id="%d" name="Picture %d" descr="%s"/><wp:cNvGraphicFramePr/>' % (n, n, escape(alt))
        if wrap is None:
            return (
                '<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="%d" cy="%d"/>'
                '<wp:effectExtent l="0" t="0" r="0" b="0"/>%s%s</wp:inline></w:drawing></w:r>' % (cx, cy, doc_pr, graphic)
            )
        wrapping = {
            "square": '<wp:wrapSquare wrapText="bothSides"/>',
            "tight": '<wp:wrapTight wrapText="bothSides"><wp:wrapPolygon edited="0"><wp:start x="0" y="0"/>'
            '<wp:lineTo x="0" y="21600"/><wp:lineTo x="21600" y="21600"/><wp:lineTo x="21600" y="0"/>'
            '<wp:lineTo x="0" y="0"/></wp:wrapPolygon></wp:wrapTight>',
            "topAndBottom": "<wp:wrapTopAndBottom/>",
        }[wrap]
        return (
            '<w:r><w:drawing><wp:anchor distT="0" distB="0" distL="114300" distR="114300" simplePos="0" '
            'relativeHeight="251659264" behindDoc="0" locked="0" layoutInCell="1" allowOverlap="1">'
            '<wp:simplePos x="0" y="0"/><wp:positionH relativeFrom="margin"><wp:align>right</wp:align></wp:positionH>'
            '<wp:positionV relativeFrom="paragraph"><wp:posOffset>0</wp:posOffset></wp:positionV>'
            '<wp:extent cx="%d" cy="%d"/><wp:effectExtent l="0" t="0" r="0" b="0"/>%s%s%s</wp:anchor></w:drawing></w:r>'
            % (cx, cy, wrapping, doc_pr, graphic)
        )

    def chart(self, title, categories, series):
        """A column chart with cached values; `series` maps a name to a list of numbers."""
        self.charts += 1
        n = self.charts
        rows = len(categories)

        def strs(values):
            return '<c:strCache><c:ptCount val="%d"/>%s</c:strCache>' % (
                len(values),
                "".join('<c:pt idx="%d"><c:v>%s</c:v></c:pt>' % (i, escape(v)) for i, v in enumerate(values)),
            )

        sers = ""
        for k, (name, values) in enumerate(series.items()):
            col = "BCDEFG"[k]
            sers += (
                '<c:ser><c:idx val="%d"/><c:order val="%d"/><c:tx><c:strRef><c:f>Sheet1!$%s$1</c:f>%s</c:strRef></c:tx>'
                '<c:cat><c:strRef><c:f>Sheet1!$A$2:$A$%d</c:f>%s</c:strRef></c:cat>'
                '<c:val><c:numRef><c:f>Sheet1!$%s$2:$%s$%d</c:f><c:numCache><c:formatCode>General</c:formatCode>'
                '<c:ptCount val="%d"/>%s</c:numCache></c:numRef></c:val></c:ser>'
                % (
                    k, k, col, strs([name]), rows + 1, strs(categories), col, col, rows + 1, rows,
                    "".join('<c:pt idx="%d"><c:v>%s</c:v></c:pt>' % (i, v) for i, v in enumerate(values)),
                )
            )
        self.extra_files["word/charts/chart%d.xml" % n] = (
            XML
            + '<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<c:chart><c:title><c:tx><c:rich><a:bodyPr/><a:p><a:r><a:t>%s</a:t></a:r></a:p></c:rich></c:tx>'
            '<c:overlay val="0"/></c:title><c:autoTitleDeleted val="0"/><c:plotArea><c:layout/>'
            '<c:barChart><c:barDir val="col"/><c:grouping val="clustered"/><c:varyColors val="0"/>%s'
            '<c:gapWidth val="150"/><c:axId val="111"/><c:axId val="222"/></c:barChart>'
            '<c:catAx><c:axId val="111"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:delete val="0"/>'
            '<c:axPos val="b"/><c:crossAx val="222"/></c:catAx>'
            '<c:valAx><c:axId val="222"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:delete val="0"/>'
            '<c:axPos val="l"/><c:crossAx val="111"/></c:valAx></c:plotArea><c:plotVisOnly val="1"/></c:chart>'
            "</c:chartSpace>" % (escape(title), sers)
        )
        rid = "rIdChart%d" % n
        self.extra_rels.append((rid, "chart", "charts/chart%d.xml" % n))
        self.extra_types.append(("/word/charts/chart%d.xml" % n, "chart"))
        return (
            '<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="5486400" cy="2743200"/>'
            '<wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="%d" name="Chart %d"/><wp:cNvGraphicFramePr/>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart">'
            '<c:chart r:id="%s"/></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>' % (100 + n, n, rid)
        )

    def textbox(self, paragraphs, width=220, height=80, fill="#f2f2f2"):
        """A text box (VML) holding paragraphs of text."""
        self.shapes += 1
        inner = "".join("<w:p>%s</w:p>" % self.run(text) for text in paragraphs)
        return (
            '<w:r><w:pict><v:rect id="textbox%d" style="position:absolute;margin-left:0;margin-top:0;'
            'width:%dpt;height:%dpt;z-index:%d;mso-position-horizontal:right;mso-position-vertical-relative:text" '
            'fillcolor="%s" strokecolor="#7f7f7f"><v:textbox inset="5pt,5pt,5pt,5pt"><w:txbxContent>%s'
            '</w:txbxContent></v:textbox><w10:wrap type="square"/></v:rect></w:pict></w:r>'
            % (self.shapes, width, height, 251659000 + self.shapes, fill, inner)
        )

    def shape(self, text, kind="roundrect", fill="#dae8fc"):
        """A drawn shape (VML) with a label."""
        self.shapes += 1
        return (
            '<w:r><w:pict><v:%s id="shape%d" style="width:140pt;height:36pt;z-index:%d" fillcolor="%s" '
            'strokecolor="#6c8ebf"><v:textbox><w:txbxContent><w:p>%s</w:p></w:txbxContent></v:textbox></v:%s></w:pict></w:r>'
            % (kind, self.shapes, 251660000 + self.shapes, fill, self.run(text), kind)
        )

    def form_text(self, name, value="", ):
        self.bookmarks += 1
        return (
            '<w:r><w:fldChar w:fldCharType="begin"><w:ffData><w:name w:val="%s"/><w:enabled/>'
            '<w:calcOnExit w:val="0"/><w:textInput><w:default w:val="%s"/></w:textInput></w:ffData></w:fldChar></w:r>'
            '<w:r><w:instrText xml:space="preserve"> FORMTEXT </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r>%s</w:r><w:r><w:fldChar w:fldCharType="end"/></w:r>'
            % (name, escape(value), t(value or "\u2002\u2002\u2002"))
        )

    def form_check(self, name, checked=False):
        return (
            '<w:r><w:fldChar w:fldCharType="begin"><w:ffData><w:name w:val="%s"/><w:enabled/>'
            '<w:calcOnExit w:val="0"/><w:checkBox><w:sizeAuto/><w:default w:val="%d"/></w:checkBox></w:ffData></w:fldChar></w:r>'
            '<w:r><w:instrText xml:space="preserve"> FORMCHECKBOX </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r>'
            % (name, 1 if checked else 0)
        )

    def form_dropdown(self, name, items):
        entries = "".join('<w:listEntry w:val="%s"/>' % escape(i) for i in items)
        return (
            '<w:r><w:fldChar w:fldCharType="begin"><w:ffData><w:name w:val="%s"/><w:enabled/>'
            '<w:calcOnExit w:val="0"/><w:ddList><w:result w:val="0"/>%s</w:ddList></w:ffData></w:fldChar></w:r>'
            '<w:r><w:instrText xml:space="preserve"> FORMDROPDOWN </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r>%s</w:r><w:r><w:fldChar w:fldCharType="end"/></w:r>'
            % (name, entries, t(items[0]))
        )

    def control_kind(self, kind, alias, tag, text, items=()):
        """A content control: kind is "dropdown", "date" or "checkbox"."""
        self.bookmarks += 1
        choice = {
            "dropdown": "<w:dropDownList>%s</w:dropDownList>"
            % "".join('<w:listItem w:displayText="%s" w:value="%s"/>' % (escape(i), escape(i)) for i in items),
            "date": '<w:date><w:dateFormat w:val="yyyy-MM-dd"/><w:lid w:val="en-US"/>'
            '<w:storeMappedDataAs w:val="dateTime"/><w:calendar w:val="gregorian"/></w:date>',
            "checkbox": '<w14:checkbox><w14:checked w14:val="0"/><w14:checkedState w14:val="2612" w14:font="MS Gothic"/>'
            '<w14:uncheckedState w14:val="2610" w14:font="MS Gothic"/></w14:checkbox>',
        }[kind]
        return (
            '<w:sdt><w:sdtPr><w:alias w:val="%s"/><w:tag w:val="%s"/><w:id w:val="%d"/>%s</w:sdtPr>'
            "<w:sdtContent>%s</w:sdtContent></w:sdt>"
            % (escape(alias), tag, 4000 + self.bookmarks, choice, self.run(text))
        )

    def block_control(self, alias, tag, paragraphs):
        self.bookmarks += 1
        return (
            '<w:sdt><w:sdtPr><w:alias w:val="%s"/><w:tag w:val="%s"/><w:id w:val="%d"/><w:showingPlcHdr/>'
            "</w:sdtPr><w:sdtContent>%s</w:sdtContent></w:sdt>" % (escape(alias), tag, 4000 + self.bookmarks, paragraphs)
        )

    def moved(self, author, text, label):
        """A move: the text deleted here is `moved_from`, and `moved_to` puts it elsewhere."""
        self.rev += 1
        n = self.rev
        return (
            '<w:moveFromRangeStart w:id="%d" w:name="%s" w:author="%s" w:date="%s"/>'
            '<w:moveFrom w:id="%d" w:author="%s" w:date="%s">%s</w:moveFrom><w:moveFromRangeEnd w:id="%d"/>'
            % (n, label, escape(author), DATE, n + 1000, escape(author), DATE, run(text, deleted=True), n)
        ), (
            '<w:moveToRangeStart w:id="%d" w:name="%s" w:author="%s" w:date="%s"/>'
            '<w:moveTo w:id="%d" w:author="%s" w:date="%s">%s</w:moveTo><w:moveToRangeEnd w:id="%d"/>'
            % (n + 2000, label, escape(author), DATE, n + 3000, escape(author), DATE, run(text), n + 2000)
        )

    def table_xml(self, rows, widths, header=True, style="TableGrid"):
        """Like `table`, but returns the XML (to nest in a cell). A cell is text, (text, span), or Raw."""
        grid = "".join('<w:gridCol w:w="%d"/>' % w for w in widths)
        out = [
            '<w:tbl><w:tblPr><w:tblStyle w:val="%s"/><w:tblW w:w="%d" w:type="dxa"/>%s'
            '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" '
            'w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>%s</w:tblGrid>'
            % (style, sum(widths), "<w:bidiVisual/>" if self.rtl else "", grid)
        ]
        for ri, row in enumerate(rows):
            out.append("<w:tr><w:trPr><w:tblHeader/></w:trPr>" if ri == 0 and header else "<w:tr>")
            col = 0
            for cell in row:
                span, vmerge = 1, ""
                if isinstance(cell, tuple):
                    cell, span = cell
                    if span == 0:
                        vmerge, span = "<w:vMerge/>", 1
                    elif span < 0:
                        vmerge, span = '<w:vMerge w:val="restart"/>', 1
                w = sum(widths[col : col + span])
                grid_span = '<w:gridSpan w:val="%d"/>' % span if span > 1 else ""
                if isinstance(cell, Raw):
                    inner = cell.xml + "<w:p/>"
                else:
                    self.para += 1
                    ppr = "<w:pPr><w:bidi/></w:pPr>" if self.rtl else ""
                    text = self.run(cell, b=ri == 0 and header) if cell else ""
                    inner = "<w:p>%s%s</w:p>" % (ppr, text)
                out.append(
                    '<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>%s%s</w:tcPr>%s</w:tc>' % (w, grid_span, vmerge, inner)
                )
                col += span
            out.append("</w:tr>")
        out.append("</w:tbl>")
        return "".join(out)

    def raw(self, xml):
        """Adds block-level XML to the body."""
        self.body.append(xml)

    # ---- package
    def save(self, name, out=None):
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
        files["word/styles.xml"] = styles_xml(self.lang)
        files["word/numbering.xml"] = numbering_xml()
        files["word/settings.xml"] = (
            XML + '<w:settings %s>%s<w:defaultTabStop w:val="720"/>'
            '<w:footnotePr><w:footnote w:id="-1"/><w:footnote w:id="0"/></w:footnotePr>'
            '<w:endnotePr><w:endnote w:id="-1"/><w:endnote w:id="0"/></w:endnotePr>'
            "%s</w:settings>"
            % (
                self.ns,
                "<w:trackRevisions/>" if self.track else "",
                (
                    '<w:compat><w:compatSetting w:name="compatibilityMode" '
                    'w:uri="http://schemas.microsoft.com/office/word" w:val="%d"/></w:compat>'
                    % self.compat
                    if self.compat
                    else ""
                ),
            )
        )
        files["word/footnotes.xml"] = notes_xml("footnote", self.footnotes, self.ns)
        files["word/endnotes.xml"] = notes_xml("endnote", self.endnotes, self.ns)
        if self.comments:
            threaded = any(parent is not None for _, _, parent in self.comments)
            rels.append(("rIdCm", "comments", "comments.xml"))
            types.append(("/word/comments.xml", "comments"))
            files["word/comments.xml"] = comments_xml(self.comments, self.ns, threaded)
            if threaded:
                rels.append(("rIdCmEx", "http://schemas.microsoft.com/office/2011/relationships/commentsExtended", "commentsExtended.xml"))
                types.append(("/word/commentsExtended.xml", "commentsExtended"))
                files["word/commentsExtended.xml"] = XML + "<w15:commentsEx %s>%s</w15:commentsEx>" % (
                    self.ns,
                    "".join(
                        '<w15:commentEx w15:paraId="%08X"%s w15:done="0"/>'
                        % (0x20000000 + i, ' w15:paraIdParent="%08X"' % (0x20000000 + parent) if parent is not None else "")
                        for i, (_, _, parent) in enumerate(self.comments)
                    ),
                )
        for kind, content in self.headers.items():
            fn = "header_%s.xml" % kind
            rels.append(("rIdH" + kind, "header", fn))
            types.append(("/word/" + fn, "header"))
            files["word/" + fn] = (
                XML + '<w:hdr %s><w:p><w:pPr><w:pStyle w:val="Header"/></w:pPr>%s</w:p></w:hdr>' % (self.ns, content)
            )
        for kind, content in self.footers.items():
            fn = "footer_%s.xml" % kind
            rels.append(("rIdF" + kind, "footer", fn))
            types.append(("/word/" + fn, "footer"))
            files["word/" + fn] = (
                XML + '<w:ftr %s><w:p><w:pPr><w:pStyle w:val="Footer"/></w:pPr>%s</w:p></w:ftr>' % (self.ns, content)
            )
        for name_, data_ in self.extra_files.items():
            files[name_] = data_
        rels.extend(self.extra_rels)
        types.extend(self.extra_types)
        files["word/document.xml"] = (
            XML + "<w:document %s><w:body>%s%s</w:body></w:document>"
            % (self.ns, "".join(self.body), self.section())
        )
        rel_base = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
        files["word/_rels/document.xml.rels"] = (
            XML
            + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                '<Relationship Id="%s" Type="%s%s" Target="%s"/>' % (i, rel_base, k, tg)
                for i, k, tg in rels
            ).replace(rel_base + "http", "http")
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
            "commentsExtended": "application/vnd.openxmlformats-officedocument.wordprocessingml.commentsExtended+xml",
            "chart": "application/vnd.openxmlformats-officedocument.drawingml.chart+xml",
        }
        files["[Content_Types].xml"] = (
            XML
            + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            + "".join('<Default Extension="%s" ContentType="%s"/>' % e for e in sorted(self.extra_defaults.items()))
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
            "<Application>Synthetic corpus generator</Application>%s</Properties>"
            % ("<Pages>%d</Pages>" % self.pages if self.pages else "")
        )
        order = ["[Content_Types].xml", "_rels/.rels"] + sorted(
            k for k in files if k not in ("[Content_Types].xml", "_rels/.rels")
        )
        out = out or OUT
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, name + ".docx")
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for k in order:
                info = zipfile.ZipInfo(k, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, files[k] if isinstance(files[k], bytes) else files[k].encode("utf-8"))
        print("wrote", path)


def notes_xml(kind, notes, ns=W):
    tag = kind
    ref = "footnoteRef" if kind == "footnote" else "endnoteRef"
    style = "FootnoteText" if kind == "footnote" else "EndnoteText"
    rstyle = "FootnoteReference" if kind == "footnote" else "EndnoteReference"
    out = [
        XML,
        "<w:%ss %s>" % (tag, ns),
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


def comments_xml(comments, ns=W, threaded=False):
    out = [XML, "<w:comments %s>" % ns]
    for cid, (author, text, _) in enumerate(comments):
        initials = "".join(w[0] for w in author.split())[:3] or "C"
        para = ' w14:paraId="%08X" w14:textId="77777777"' % (0x20000000 + cid) if threaded else ""
        out.append(
            '<w:comment w:id="%d" w:author="%s" w:date="%s" w:initials="%s">'
            '<w:p%s><w:pPr><w:pStyle w:val="CommentText"/></w:pPr>'
            '<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr><w:annotationRef/></w:r>%s</w:p>'
            "</w:comment>" % (cid, escape(author), DATE, escape(initials), para, run(text))
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


# language -> (east Asian font, complex-script font, language tags)
_LATIN = ("SimSun", "Times New Roman")
PROFILES = {
    "en": (_LATIN, "en-GB", "zh-CN", "ar-SA"),
    "zh": (_LATIN, "en-GB", "zh-CN", "ar-SA"),
    "zh-TW": (("PMingLiU", "Times New Roman"), "en-GB", "zh-TW", "ar-SA"),
    "ja": (("MS Mincho", "Times New Roman"), "ja-JP", "ja-JP", "ar-SA"),
    "ko": (("Malgun Gothic", "Times New Roman"), "ko-KR", "ko-KR", "ar-SA"),
    "de": (_LATIN, "de-DE", "zh-CN", "ar-SA"),
    "fr": (_LATIN, "fr-FR", "zh-CN", "ar-SA"),
    "es": (_LATIN, "es-ES", "zh-CN", "ar-SA"),
    "ar": ((_LATIN[0], "Arial"), "ar-SA", "zh-CN", "ar-SA"),
    "he": ((_LATIN[0], "Arial"), "he-IL", "zh-CN", "he-IL"),
}


def styles_xml(language="en"):
    (east, complex_), latin_lang, east_lang, bidi_lang = PROFILES[language]
    fonts = '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="%s" w:cs="%s"/>' % (east, complex_)
    lang = '<w:lang w:val="%s" w:eastAsia="%s"%s/>' % (
        latin_lang,
        east_lang,
        "" if language in ("en", "zh") else ' w:bidi="%s"' % bidi_lang,
    )

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


