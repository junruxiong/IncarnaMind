#!/usr/bin/env python3
"""Writes the first 18 synthetic .docx files of the Word corpus (eval/word/README.md) into documents/synthetic/.

Standard library only, and deterministic: the same script writes the same bytes. Every file is
invented text with invented names; nothing in it is real contract language or a real person.

    python3 eval/word/generate-synthetic.py
"""
import sys

from ooxml import Doc, run

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
