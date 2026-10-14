#!/usr/bin/env python3
"""Writes the synthetic domain documents of the Word corpus (eval/word/README.md).

Legal, finance, medical, academic, government, engineering, HR and education documents, in ten
languages, with the Word features a round-trip has to keep. Standard library only; the same script
writes the same bytes. The text is invented (domain_text.py); the features are put in by flags.

    python3 eval/word/generate-domains.py core                 # into breadth/domains/ (committed)
    python3 eval/word/generate-domains.py extended OUT_DIR      # the rest, for the extended set
"""
import os
import sys

import ooxml
from domain_text import FILLER, PACKS, SECTIONS
from ooxml import Doc, Raw, m_display, m_f, m_inline, m_r, m_rad, m_sub, m_sum, m_sup, run

HERE = os.path.dirname(os.path.abspath(__file__))

THANKS = {
    "en": "Thanks, noted.", "zh": "好的，已知悉。", "zh-TW": "好的，已知悉。", "ja": "了解しました。",
    "ko": "알겠습니다.", "de": "Danke, zur Kenntnis genommen.", "fr": "Merci, noté.", "es": "Gracias, anotado.",
    "ar": "شكراً، تم العلم.", "he": "תודה, נרשם.",
}
# (name, language of the text, flags, committed)
SPECS = [
    ("legal-en-court-complaint", "en", "notes tables textbox signature", True),
    ("legal-fr-contrat", "fr", "notes tables signature xref", True),
    ("legal-de-kaufvertrag", "de", "notes tables signature merged", False),
    ("legal-zh-complaint", "zh", "tracked comments tables form", False),
    ("legal-ar-contract", "ar", "notes tables signature", True),
    ("finance-en-prospectus", "en", "toc columns merged caption xref", True),
    ("finance-de-audit-letter", "de", "comments tracked notes", False),
    ("finance-ja-securities-report", "ja", "tables chart caption", False),
    ("finance-es-informe", "es", "tables chart caption notes", False),
    ("medical-en-clinical-protocol", "en", "toc merged nested form headers", True),
    ("medical-en-sop", "en", "form controls tracked comments", False),
    ("medical-ko-protocol", "ko", "tables merged form notes", True),
    ("medical-zh-TW-sop", "zh-TW", "tables form controls", False),
    ("academic-en-paper", "en", "columns math image caption notes", True),
    ("academic-he-paper", "he", "math notes caption", False),
    ("gov-zh-notice", "zh", "nonum tracked comments", True),
    ("gov-fr-reglement", "fr", "tables notes", False),
    ("gov-ja-form", "ja", "form tables controls", False),
    ("gov-zh-TW-announcement", "zh-TW", "nonum tables", True),
    ("gov-en-form", "en", "form controls tables", False),
    ("engineering-en-spec", "en", "landscape shapes math merged compat12 caption", True),
    ("engineering-de-norm", "de", "tables math caption", False),
    ("hr-en-handbook", "en", "toc controls image comments", False),
    ("hr-ja-contract", "ja", "controls tables signature", True),
    ("hr-he-contract", "he", "tracked tables signature", True),
    ("education-en-syllabus", "en", "merged textbox image tables", False),
    ("education-es-lesson-plan", "es", "tables shapes", False),
    ("education-ar-course", "ar", "tables merged", False),
    ("mixed-en-zh-report", "zh", "tables chart tracked", True),
    ("mixed-en-ja-ko-manual", "ja", "tables shapes", False),
    ("mixed-en-ar-contract", "ar", "tables signature comments", False),
]
PACK_OF = {"mixed-en-zh-report": "en", "mixed-en-ja-ko-manual": "en", "mixed-en-ar-contract": "en"}
# Long documents: (name, language, title, subtitle, chapters, committed)
LONG = [
    ("legal-en-msa-long", "en", "Master Services Agreement", "With schedules, about 125 pages", 12, True),
    ("finance-en-annual-report-long", "en", "Annual Report and Accounts", "For the year ended 31 December", 12, False),
    ("engineering-en-standard-long", "en", "Standard for Order Processing Systems", "Requirements and test methods", 12, False),
    ("academic-zh-thesis-long", "zh", "基于稀疏订单数据的配送时间估计", "硕士学位论文（长文档）", 12, False),
]


def numbering(lang, flags):
    if "nonum" in flags:
        return None
    return (3, 0) if lang in ("zh", "zh-TW", "ja") else (5, 0)


def header_footer(d, P, title):
    d.set_header(d.run(title) + "<w:r><w:tab/></w:r>" + d.run(P["conf"]))
    d.set_footer(d.run(P["page"] + " ") + d.field("PAGE", "1") + d.run(" " + P["of"] + " ") + d.field("NUMPAGES", "3"))


def caption(d, P, kind, text):
    label = P[kind]
    d.p(d.run(label + " ") + d.field("SEQ %s \\* ARABIC" % label, "1") + d.run(": " + text), align="center")


def equations(d, P):
    quadratic = m_display(
        m_r("x="),
        m_f(m_r("-b\u00b1") + m_rad(m_sup(m_r("b"), m_r("2")) + m_r("-4ac")), m_r("2a")),
    )
    total = m_display(m_sum(m_r("i=1"), m_r("n"), m_sub(m_r("x"), m_r("i"))), m_r("="), m_r("n"), m_sub(m_r("\u03bc"), m_r("0")))
    d.p(d.run(P["math"] + ":"))
    d.p(quadratic)
    d.p(total)
    d.p(d.run(P["equation"] + " ") + d.field("SEQ %s \\* ARABIC" % P["equation"], "1") + d.run(": ") + m_inline(m_sup(m_r("e"), m_r("i\u03c0")), m_r("+1=0")))


def blocks(d, P, name, lang, flags, sections):
    head, rows = P["head"], P["rows"]
    if "tables" in flags:
        caption(d, P, "table", head[0])
        d.table([list(head)] + [list(r) for r in rows], [3000, 3000, 3000])
    if "merged" in flags:
        caption(d, P, "table", P["group"][0] + " / " + P["group"][1])
        d.table(
            [
                [(P["group"][0], 2), (P["group"][1], 1)],
                [head[0], head[1], head[2]],
                [(rows[0][0], -1), rows[0][1], rows[0][2]],
                [("", 0), rows[1][1], rows[1][2]],
                [rows[2][0], (rows[2][1], 2)],
            ],
            [3000, 3000, 3000],
        )
    if "nested" in flags:
        inner = d.table_xml([[head[0], head[1]], [rows[0][0], rows[0][1]]], [1800, 1800])
        d.raw(d.table_xml([[P["nested"], head[2]], [Raw(inner), rows[0][2]]], [4500, 4500]))
        d.p()
    if "chart" in flags:
        title, cats, series = P["chart"]
        d.p(d.chart(title, cats, {series: [120, 135, 150, 170]}), align="center")
        caption(d, P, "figure", title)
    if "math" in flags:
        equations(d, P)
    if "image" in flags:
        d.p(d.image(wrap="square", alt=P["figure"]) + d.run(sections[0][1][0] + " " + sections[1][1][0]))
        caption(d, P, "figure", P["figure"])
    if "textbox" in flags:
        d.p(d.textbox([P["box"], P["conf"]]) + d.run(sections[0][1][1]))
    if "shapes" in flags:
        d.p("".join(d.shape(s) for s in P["shapes"]))
    if "form" in flags:
        d.p(d.run(P["form"][0] + ": ") + d.form_text("name", "") + d.run("  " + P["form"][1] + ": ") + d.form_text("date", "2026-01-15"))
        d.p(d.form_check("agree") + d.run(" " + P["form"][2] + "   ") + d.form_dropdown("choice", P["form"][3]))
    if "controls" in flags:
        d.p(
            d.control(P["ctrl"][0], "title", P["ctrl"][1])
            + d.run("  ")
            + d.control_kind("date", P["form"][1], "date", "2026-01-15")
            + d.run("  ")
            + d.control_kind("dropdown", P["form"][0], "choice", P["form"][3][0], P["form"][3])
            + d.run("  ")
            + d.control_kind("checkbox", P["form"][2], "agree", "\u2610")
        )
        d.raw(d.block_control(P["ctrl"][0], "block", "<w:p>%s</w:p>" % d.run(P["ctrl"][1])))
    if "xref" in flags:
        d.p(d.run(sections[0][0] + ": ") + d.field("REF sec0 \\h", sections[0][0]) + d.run(" ") + d.field("PAGEREF sec0 \\h", "1"))
    if "columns" in flags:
        d.p(section=d.sect(header=d.run(P["conf"]), footer=d.run(P["page"])))
        for _, paras in sections:
            for text in paras:
                d.p(text * 2)
        d.p(section=d.sect(cols=2))
    if "landscape" in flags:
        d.p(section=d.sect())
        caption(d, P, "table", head[0])
        d.table([list(head) * 2] + [list(r) * 2 for r in rows], [2000] * 6)
        d.p(section=d.sect(landscape=True, header=d.run(P["conf"])))
    if "signature" in flags:
        d.table([[P["sig"][0], P["sig"][1]], ["", ""], ["", ""]], [4500, 4500])


def short(name, lang, flags, out):
    P = PACKS[PACK_OF.get(name, lang)]
    title, subtitle, sections = SECTIONS[name]
    d = Doc(lang).extended()
    flags = flags.split()
    if "compat12" in flags:
        d.compat = 12
    if "tracked" in flags:
        d.track = True
    header_footer(d, P, title)
    d.final_cols = 1
    d.p(title, "Title")
    d.p(subtitle, align="center")
    if "toc" in flags:
        d.p(d.run(P["toc"], b=True))
        d.p(d.field('TOC \\o "1-2" \\h \\z \\u', " ".join(h for h, _ in sections)), "TOC1")
    num = numbering(lang, flags)
    for idx, (heading, paras) in enumerate(sections):
        d.p(d.bookmark("sec%d" % idx, d.run(heading)), "Heading1", num=num)
        for j, text in enumerate(paras):
            inline = d.run(text)
            if "notes" in flags and idx == 0:
                inline += d.footnote(P["fn"][0]) if j == 0 else d.endnote(P["fn"][1])
            if "comments" in flags and idx == 1 and j == 0:
                inline = d.comment(P["cm"][0], P["cm"][1], inline, replies=[P["cm"][2], (P["cm"][0], THANKS[lang if lang in THANKS else "en"])])
            if "tracked" in flags and idx >= 1:
                who, old, new = P["tr"]
                who2, old2, new2 = P["tr2"]
                inline = d.run(text) + d.delete(who, old) + d.ins(who, d.run(new)) + d.run(" ") + d.delete(who2, old2) + d.ins(who2, d.run(new2))
                if idx == 2 and j == 0:
                    gone, there = d.moved(who2, paras[1], "move1")
                    inline += gone
                    d.p(inline)
                    d.p(there)
                    continue
                if idx == 2 and j == 1:
                    inline = d.format_change(who2, text)
            d.p(inline)
    blocks(d, P, name, lang, flags, sections)
    d.save(name, out)


def long_doc(name, lang, title, subtitle, chapters, out):
    F = FILLER[lang]
    P = PACKS[lang]
    d = Doc(lang).extended()
    d.pages = 125
    if name.startswith("legal"):
        d.compat = 12
    header_footer(d, P, title)
    d.p(title, "Title")
    d.p(subtitle, align="center")
    d.p(d.run(P["toc"], b=True))
    d.p(d.field('TOC \\o "1-2" \\h \\z \\u', title), "TOC1")
    num = numbering(lang, "")
    chapter_names = {
        "en": ["Definitions", "Scope", "Delivery", "Charges", "Warranties", "Liability", "Confidentiality", "Records", "Change control", "Disputes", "Termination", "General"],
        "zh": ["绪论", "文献综述", "问题描述", "方法设计", "模型推导", "数据说明", "实验设置", "结果分析", "讨论", "局限性", "结论", "展望"],
    }[lang]
    n = 0

    def sentence(k):
        s = F["subjects"][k % 6]
        v = F["verbs"][(k * 5 + 1) % 6]
        o = F["objects"][(k * 7 + 2) % 6]
        t = F["tails"][(k * 11 + 3) % 6]
        return (s + " " + v + " " + o + " " + t + ".") if lang == "en" else (s + v + o + t + "。")

    for c in range(chapters):
        d.p(d.bookmark("ch%d" % c, d.run(chapter_names[c])), "Heading1", num=num)
        for sub in range(4):
            d.p(d.run("%s %d.%d" % (chapter_names[c], c + 1, sub + 1)), "Heading2", num=(num[0], 1))
            for k in range(20):
                n += 1
                text = (" " if lang == "en" else "").join(sentence(n * 4 + i) for i in range(4))
                inline = d.run(text)
                if k == 3 and sub == 0:
                    inline += d.footnote(P["fn"][0])
                if k == 9 and sub == 2:
                    inline += d.endnote(P["fn"][1])
                d.p(inline)
            if sub == 1:
                caption(d, P, "table", chapter_names[c])
                d.table([list(P["head"])] + [list(r) for r in P["rows"]], [3000, 3000, 3000])
    d.save(name, out)


def main(argv):
    mode = argv[1] if len(argv) > 1 else "core"
    if mode == "core":
        out, committed = os.path.join(HERE, "breadth", "domains"), (True,)
    elif mode == "extended" and len(argv) > 2:
        out, committed = argv[2], (False,)
    else:
        print(__doc__)
        return 2
    for name, lang, flags, is_core in SPECS:
        if (is_core,) == committed:
            short(name, lang, flags, out)
    for name, lang, title, subtitle, chapters, is_core in LONG:
        if (is_core,) == committed:
            long_doc(name, lang, title, subtitle, chapters, out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
