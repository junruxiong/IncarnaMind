"""Test workbooks with every hard feature, written by XlsxWriter (BSD-2). Test data only."""
import sys, os, random, datetime
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell as rc

OUT = sys.argv[1]
random.seed(7)

def finance_model():
    wb = xlsxwriter.Workbook(os.path.join(OUT, "xw-finance-model.xlsx"))
    wb.set_properties({"title": "Five-year model", "author": "Test"})
    title = wb.add_format({"bold": True, "font_size": 16, "align": "center", "valign": "vcenter", "bg_color": "#1F3864", "font_color": "#FFFFFF"})
    hdr = wb.add_format({"bold": True, "bottom": 2, "bg_color": "#D9E1F2"})
    gbp = wb.add_format({"num_format": "£#,##0;[Red]-£#,##0"})
    acct = wb.add_format({"num_format": '_-£* #,##0.00_-;-£* #,##0.00_-;_-£* "-"??_-;_-@_-'})
    pct = wb.add_format({"num_format": "0.0%"})
    mill = wb.add_format({"num_format": '#,##0.0,,"m"'})
    date = wb.add_format({"num_format": "dd mmm yyyy"})
    inp = wb.add_format({"bg_color": "#FFF2CC", "border": 1})
    inp_pct = wb.add_format({"bg_color": "#FFF2CC", "border": 1, "num_format": "0.00%"})
    bold_gbp = wb.add_format({"bold": True, "num_format": "£#,##0", "top": 1, "bottom": 6})

    i = wb.add_worksheet("Inputs")
    i.merge_range("A1:D1", "Assumptions", title)
    i.set_row(0, 30)
    i.set_column("A:A", 28); i.set_column("B:D", 14)
    rows = [("Start revenue", 1250000, inp), ("Growth rate", 0.085, inp_pct), ("Tax rate", 0.25, inp_pct),
            ("Discount rate", 0.09, inp_pct), ("Cost of sales %", 0.42, inp_pct), ("Opex per year", 310000, inp),
            ("Loan amount", 500000, inp), ("Loan rate", 0.061, inp_pct), ("Loan years", 5, inp)]
    for r, (label, v, f) in enumerate(rows, start=2):
        i.write(r, 0, label); i.write(r, 1, v, f)
    i.write(12, 0, "Start date"); i.write_datetime(12, 1, datetime.datetime(2027, 1, 1), date)
    i.write(13, 0, "Scenario"); i.write(13, 1, "Base", inp)
    i.data_validation("B14", {"validate": "list", "source": ["Low", "Base", "High"], "input_title": "Scenario", "input_message": "Pick one"})
    i.data_validation("B4", {"validate": "decimal", "criteria": "between", "minimum": 0, "maximum": 0.5, "error_message": "0 to 50%"})
    i.data_validation("B11", {"validate": "integer", "criteria": ">", "value": 0})
    i.write(15, 0, "Scenario uplift"); i.write_formula(15, 1, '=IF(B14="High",0.03,IF(B14="Low",-0.03,0))', pct)
    i.write(16, 0, "Effective growth"); i.write_formula(16, 1, "=B4+B16", pct)
    wb.define_name("StartRevenue", "=Inputs!$B$3")
    wb.define_name("Growth", "=Inputs!$B$17")
    wb.define_name("TaxRate", "=Inputs!$B$5")
    wb.define_name("DiscountRate", "=Inputs!$B$6")
    wb.define_name("Inputs!LocalNote", "=Inputs!$A$1")
    wb.define_name("Years", "=Revenue!$B$3:$F$3")

    rv = wb.add_worksheet("Revenue")
    rv.freeze_panes(3, 1)
    rv.set_column("A:A", 22); rv.set_column("B:G", 14)
    rv.merge_range("A1:G1", "Revenue build", title)
    rv.write_row(2, 0, ["", 2027, 2028, 2029, 2030, 2031, "Total"], hdr)
    rv.write(3, 0, "Revenue")
    rv.write_formula(3, 1, "=StartRevenue", gbp)
    for c in range(2, 6):
        rv.write_formula(3, c, f"={rc(3, c-1)}*(1+Growth)", gbp)
    rv.write_formula(3, 6, "=SUM(B4:F4)", bold_gbp)
    rv.write(4, 0, "Cost of sales")
    for c in range(1, 6):
        rv.write_formula(4, c, f"=-{rc(3, c)}*Inputs!$B$7", gbp)
    rv.write_formula(4, 6, "=SUM(B5:F5)", bold_gbp)
    rv.write(5, 0, "Gross profit")
    for c in range(1, 7):
        rv.write_formula(5, c, f"={rc(3, c)}+{rc(4, c)}", gbp)
    rv.write(6, 0, "Gross margin")
    for c in range(1, 7):
        rv.write_formula(6, c, f"=IFERROR({rc(5, c)}/{rc(3, c)},0)", pct)
    rv.write(7, 0, "Revenue (m)")
    for c in range(1, 6):
        rv.write_formula(7, c, f"={rc(3, c)}", mill)
    rv.conditional_format("B7:F7", {"type": "data_bar", "bar_color": "#63BE7B", "data_bar_2010": True})
    rv.conditional_format("B6:F6", {"type": "3_color_scale"})
    rv.conditional_format("B4:F4", {"type": "icon_set", "icon_style": "3_arrows"})
    rv.conditional_format("B5:F5", {"type": "cell", "criteria": "<", "value": -700000, "format": wb.add_format({"font_color": "#9C0006", "bg_color": "#FFC7CE"})})
    ch = wb.add_chart({"type": "column"})
    ch.add_series({"name": "=Revenue!$A$4", "categories": "=Revenue!$B$3:$F$3", "values": "=Revenue!$B$4:$F$4"})
    ln = wb.add_chart({"type": "line"})
    ln.add_series({"name": "=Revenue!$A$6", "categories": "=Revenue!$B$3:$F$3", "values": "=Revenue!$B$6:$F$6", "y2_axis": True})
    ch.combine(ln)
    ch.set_title({"name": "Revenue and gross profit"})
    rv.insert_chart("B11", ch)

    co = wb.add_worksheet("Costs")
    co.set_column("A:A", 24); co.set_column("B:F", 14)
    co.write_row(0, 0, ["Item", "Category", "Amount", "Month", "Approved"], hdr)
    cats = ["Rent", "Staff", "IT", "Travel", "Marketing"]
    for r in range(1, 61):
        co.write(r, 0, f"Cost line {r}")
        co.write(r, 1, cats[r % 5])
        co.write_number(r, 2, round(random.uniform(200, 9000), 2), acct)
        co.write_datetime(r, 3, datetime.datetime(2027, 1 + (r % 12), 1), date)
        co.write_boolean(r, 4, r % 3 != 0)
    co.data_validation("B2:B61", {"validate": "list", "source": "=$H$2:$H$6"})
    for k, cat in enumerate(cats):
        co.write(1 + k, 7, cat)
        co.write_formula(1 + k, 8, f'=SUMIF($B$2:$B$61,H{2+k},$C$2:$C$61)', acct)
        co.write_formula(1 + k, 9, f'=COUNTIFS($B$2:$B$61,H{2+k},$E$2:$E$61,TRUE)')
        co.write_formula(1 + k, 10, f'=AVERAGEIF($B$2:$B$61,H{2+k},$C$2:$C$61)', acct)
    co.write(7, 7, "Total"); co.write_formula(7, 8, "=SUM(I2:I6)", acct)
    co.write(8, 7, "Largest"); co.write_formula(8, 8, "=MAX(C2:C61)", acct)
    co.write(9, 7, "Smallest"); co.write_formula(9, 8, "=MIN(C2:C61)", acct)
    co.write(10, 7, "Lookup IT"); co.write_formula(10, 8, '=VLOOKUP("IT",H2:I6,2,FALSE)', acct)
    co.write(11, 7, "Index/match"); co.write_formula(11, 8, '=INDEX(I2:I6,MATCH("Staff",H2:H6,0))', acct)
    co.write(12, 7, "Approved total"); co.write_formula(12, 8, '=SUMPRODUCT((E2:E61=TRUE)*C2:C61)', acct)
    co.write(13, 7, "Rounded"); co.write_formula(13, 8, '=ROUND(I8/1000,1)')
    co.write(14, 7, "Up/down"); co.write_formula(14, 8, '=ROUNDUP(I9,-2)-ROUNDDOWN(I10,-2)')
    co.write(15, 7, "Lines"); co.write_formula(15, 8, '=COUNT(C2:C61)&" lines, "&COUNTA(A2:A61)&" names"')
    co.conditional_format("C2:C61", {"type": "top", "value": 5, "format": wb.add_format({"bold": True})})
    co.conditional_format("A2:E61", {"type": "formula", "criteria": "=$E2=FALSE", "format": wb.add_format({"font_color": "#808080"})})
    co.autofilter("A1:E61")

    su = wb.add_worksheet("Summary")
    su.set_column("A:A", 26); su.set_column("B:B", 16)
    su.merge_range("A1:B1", "Summary", title)
    su.write(2, 0, "Five-year revenue"); su.write_formula(2, 1, "=Revenue!G4", gbp)
    su.write(3, 0, "Five-year gross profit"); su.write_formula(3, 1, "=Revenue!G6", gbp)
    su.write(4, 0, "Annual costs"); su.write_formula(4, 1, "=Costs!I8", gbp)
    su.write(5, 0, "Opex, five years"); su.write_formula(5, 1, "=Inputs!B8*5", gbp)
    su.write(6, 0, "Profit before tax"); su.write_formula(6, 1, "=B4-B5-B6", gbp)
    su.write(7, 0, "Tax"); su.write_formula(7, 1, "=MAX(0,B7*TaxRate)", gbp)
    su.write(8, 0, "Profit after tax"); su.write_formula(8, 1, "=B7-B8", bold_gbp)
    su.write(9, 0, "NPV of gross profit"); su.write_formula(9, 1, "=NPV(DiscountRate,Revenue!B6:F6)", gbp)
    su.write(10, 0, "Loan payment"); su.write_formula(10, 1, "=PMT(Inputs!B10,Inputs!B11,-Inputs!B9)", gbp)
    su.write(11, 0, "Cash flows"); su.write_row(11, 2, [-900000, 150000, 260000, 310000, 360000, 420000])
    su.write(12, 0, "IRR"); su.write_formula(12, 1, "=IRR(C12:H12)", pct)
    su.write(13, 0, "First year end"); su.write_formula(13, 1, "=EOMONTH(Inputs!B13,11)", date)
    su.write(14, 0, "Model year"); su.write_formula(14, 1, "=YEAR(Inputs!B13)")
    su.write(15, 0, "Label"); su.write_formula(15, 1, '="Model "&TEXT(B10,"#,##0")&" NPV"')
    su.write(16, 0, "Check"); su.write_formula(16, 1, '=IF(AND(B9>0,B13>0),"OK",IF(OR(B9<0,B13<0),"Loss","?"))')
    su.write(17, 0, "Abs"); su.write_formula(17, 1, "=ABS(Revenue!B5)+SQRT(16)+POWER(2,3)+MOD(17,5)")
    su.write(18, 0, "Date"); su.write_formula(18, 1, "=DATE(2027,12,31)-Inputs!B13", wb.add_format({"num_format": "0"}))
    su.activate()
    wb.close()

def sales_10k():
    wb = xlsxwriter.Workbook(os.path.join(OUT, "xw-sales-12k-rows.xlsx"))
    ws = wb.add_worksheet("Orders")
    hdr = wb.add_format({"bold": True, "bg_color": "#DDEBF7"})
    money = wb.add_format({"num_format": "#,##0.00"})
    date = wb.add_format({"num_format": "yyyy-mm-dd"})
    ws.freeze_panes(1, 0)
    ws.set_column("A:A", 10); ws.set_column("B:C", 14); ws.set_column("D:H", 12)
    ws.write_row(0, 0, ["Order", "Region", "Product", "Qty", "Price", "Amount", "Band", "Date"], hdr)
    regions = ["North", "South", "East", "West", "华东", "华南"]
    products = ["Widget", "Gadget", "Doohickey", "Sprocket"]
    N = 12000
    for r in range(1, N + 1):
        ws.write_number(r, 0, 100000 + r)
        ws.write(r, 1, regions[r % 6])
        ws.write(r, 2, products[(r * 7) % 4])
        q = (r * 13) % 50 + 1
        p = round(5 + ((r * 31) % 400) / 4, 2)
        ws.write_number(r, 3, q)
        ws.write_number(r, 4, p, money)
        ws.write_formula(r, 5, f"=D{r+1}*E{r+1}", money, q * p)
        ws.write_formula(r, 6, f'=IF(F{r+1}>1000,"High","Low")', None, "High" if q * p > 1000 else "Low")
        ws.write_datetime(r, 7, datetime.datetime(2026, 1, 1) + datetime.timedelta(days=r % 365), date)
    t = wb.add_worksheet("Totals")
    t.write_row(0, 0, ["Region", "Amount", "Orders", "Share"], hdr)
    for k, reg in enumerate(regions):
        t.write(k + 1, 0, reg)
        t.write_formula(k + 1, 1, f"=SUMIF(Orders!B:B,A{k+2},Orders!F:F)", money)
        t.write_formula(k + 1, 2, f"=COUNTIF(Orders!B:B,A{k+2})")
        t.write_formula(k + 1, 3, f"=B{k+2}/$B$9", wb.add_format({"num_format": "0.0%"}))
    t.write(8, 0, "Total"); t.write_formula(8, 1, "=SUM(B2:B7)", money); t.write_formula(8, 2, "=SUM(C2:C7)")
    t.write(9, 0, "Grand"); t.write_formula(9, 1, f"=SUM(Orders!F2:F{N+1})", money)
    t.write(10, 0, "Avg"); t.write_formula(10, 1, f"=AVERAGE(Orders!F2:F{N+1})", money)
    t.write(11, 0, "High"); t.write_formula(11, 1, f'=COUNTIF(Orders!G2:G{N+1},"High")')
    wb.close()

def table_refs():
    wb = xlsxwriter.Workbook(os.path.join(OUT, "xw-table-structured-refs.xlsx"))
    ws = wb.add_worksheet("Stock")
    data = [["Bolt", 120, 0.15], ["Nut", 300, 0.05], ["Washer", 800, 0.02], ["Screw", 250, 0.11], ["Rivet", 90, 0.32]]
    ws.add_table("A1:D6", {"name": "Stock", "data": data, "columns": [
        {"header": "Part"}, {"header": "Qty"}, {"header": "Unit"},
        {"header": "Value", "formula": "=[@Qty]*[@Unit]", "format": wb.add_format({"num_format": "0.00"})}],
        "total_row": False, "style": "Table Style Medium 9"})
    ws.write("F1", "Total value"); ws.write_formula("G1", "=SUM(Stock[Value])")
    ws.write("F2", "Total qty"); ws.write_formula("G2", "=SUM(Stock[Qty])")
    ws.write("F3", "Max unit"); ws.write_formula("G3", "=MAX(Stock[Unit])")
    ws.set_column("A:G", 12)
    wb.close()

def cjk():
    wb = xlsxwriter.Workbook(os.path.join(OUT, "xw-cjk-门店报表.xlsx"))
    title = wb.add_format({"bold": True, "font_size": 14, "font_name": "微软雅黑", "align": "center"})
    yuan = wb.add_format({"num_format": "[$¥-804]#,##0.00"})
    wan = wb.add_format({"num_format": '0.0,"万"'})
    cn_date = wb.add_format({"num_format": 'yyyy"年"m"月"d"日"'})
    pct = wb.add_format({"num_format": "0.0%"})
    s = wb.add_worksheet("销售数据")
    s.merge_range("A1:F1", "2026年第三季度门店销售", title)
    s.write_row(1, 0, ["门店", "城市", "七月", "八月", "九月", "合计"])
    stores = [("静安店", "上海"), ("朝阳店", "北京"), ("天河店", "广州"), ("南山店", "深圳"), ("西湖店", "杭州"), ("武侯店", "成都")]
    for r, (st, city) in enumerate(stores, start=2):
        s.write(r, 0, st); s.write(r, 1, city)
        for c in range(2, 5):
            s.write_number(r, c, random.randint(80000, 400000), yuan)
        s.write_formula(r, 5, f"=SUM(C{r+1}:E{r+1})", yuan)
    s.write(8, 0, "总计"); 
    for c in range(2, 6):
        col = chr(65 + c)
        s.write_formula(8, c, f"=SUM({col}3:{col}8)", yuan)
    s.write_comment("A3", "旗舰店，2025年翻新")
    s.write_rich_string("A11", "备注：", wb.add_format({"bold": True, "color": "red"}), "数据未经审计", "，仅供内部参考")
    s.write("A12", "报告日期"); s.write_datetime("B12", datetime.datetime(2026, 10, 1), cn_date)
    s.set_column("A:B", 12); s.set_column("C:F", 16)
    s.conditional_format("F3:F8", {"type": "data_bar"})
    h = wb.add_worksheet("汇总")
    h.write("A1", "季度总额"); h.write_formula("B1", "='销售数据'!F9", yuan)
    h.write("A2", "万元"); h.write_formula("B2", "='销售数据'!F9", wan)
    h.write("A3", "最高门店"); h.write_formula("B3", "=INDEX('销售数据'!A3:A8,MATCH(MAX('销售数据'!F3:F8),'销售数据'!F3:F8,0))")
    h.write("A4", "上海占比"); h.write_formula("B4", "=SUMIF('销售数据'!B3:B8,\"上海\",'销售数据'!F3:F8)/B1", pct)
    h.write("A5", "平均"); h.write_formula("B5", "=AVERAGE('销售数据'!F3:F8)", yuan)
    h.write("A6", "说明"); h.write_formula("B6", '=LEFT(\'销售数据\'!A3,2)&"·"&RIGHT(\'销售数据\'!B3,1)&LEN(\'销售数据\'!A1)')
    h.set_column("A:A", 12); h.set_column("B:B", 18)
    ch = wb.add_chart({"type": "pie"})
    ch.add_series({"name": "合计", "categories": "='销售数据'!$A$3:$A$8", "values": "='销售数据'!$F$3:$F$8"})
    ch.set_title({"name": "门店占比"})
    h.insert_chart("D2", ch)
    wb.close()

def charts():
    wb = xlsxwriter.Workbook(os.path.join(OUT, "xw-charts-sparklines.xlsx"))
    ws = wb.add_worksheet("Data")
    ws.write_row(0, 0, ["Month", "A", "B", "C", "Trend"])
    for r in range(1, 13):
        ws.write(r, 0, datetime.date(2026, r, 1).strftime("%b"))
        for c in range(1, 4):
            ws.write_number(r, c, random.randint(10, 100))
    ws.write(13, 0, "Total")
    for c in range(1, 4):
        col = chr(65 + c); ws.write_formula(13, c, f"=SUM({col}2:{col}13)")
    for r in range(1, 13):
        ws.add_sparkline(r, 4, {"range": f"Data!B{r+1}:D{r+1}"})
    for k, typ in enumerate(["bar", "line", "area", "scatter", "radar", "doughnut"]):
        ch = wb.add_chart({"type": typ})
        for c in range(1, 4 if typ != "doughnut" else 2):
            ch.add_series({"name": ["Data", 0, c], "categories": ["Data", 1, 0, 12, 0], "values": ["Data", 1, c, 12, c]})
        ch.set_title({"name": f"{typ} chart"})
        ws.insert_chart(16 + (k // 2) * 16, 1 + (k % 2) * 8, ch)
    cs = wb.add_chartsheet("Chart sheet")
    c2 = wb.add_chart({"type": "column", "subtype": "stacked"})
    for c in range(1, 4):
        c2.add_series({"name": ["Data", 0, c], "categories": ["Data", 1, 0, 12, 0], "values": ["Data", 1, c, 12, c]})
    cs.set_chart(c2)
    ws.activate()
    wb.close()

def names_cf_dv():
    wb = xlsxwriter.Workbook(os.path.join(OUT, "xw-names-cf-validation.xlsx"))
    a = wb.add_worksheet("Rates")
    b = wb.add_worksheet("Quote")
    c = wb.add_worksheet("Hidden lists")
    c.hide()
    a.write_row(0, 0, ["Band", "Rate"])
    for r, (band, rate) in enumerate([("A", 0.05), ("B", 0.08), ("C", 0.12), ("D", 0.2)], start=1):
        a.write(r, 0, band); a.write_number(r, 1, rate)
    wb.define_name("RateTable", "=Rates!$A$2:$B$5")
    wb.define_name("VAT", "=0.2")
    wb.define_name("Quote!Discount", "=Quote!$B$3")
    wb.define_name("_xlnm.Print_Area", "=Quote!$A$1:$D$20") if False else None
    b.print_area("A1:D20"); b.repeat_rows(0)
    for r, v in enumerate(["Gold", "Silver", "Bronze"]):
        c.write(r, 0, v)
    wb.define_name("Tiers", "='Hidden lists'!$A$1:$A$3")
    b.write("A1", "Band"); b.write("B1", "B")
    b.data_validation("B1", {"validate": "list", "source": "=Rates!$A$2:$A$5"})
    b.write("A2", "Tier"); b.write("B2", "Gold")
    b.data_validation("B2", {"validate": "list", "source": "=Tiers"})
    b.write("A3", "Discount"); b.write_number("B3", 0.1)
    b.data_validation("B3", {"validate": "decimal", "criteria": "between", "minimum": 0, "maximum": 0.3})
    b.write("A4", "Date"); b.write_datetime("B4", datetime.datetime(2026, 11, 2), wb.add_format({"num_format": "d/m/yy"}))
    b.data_validation("B4", {"validate": "date", "criteria": ">", "value": datetime.date(2026, 1, 1)})
    b.write("A5", "Net"); b.write_number("B5", 1000)
    b.data_validation("B5", {"validate": "custom", "value": "=B5>0"})
    b.write("A6", "Rate"); b.write_formula("B6", "=VLOOKUP(B1,RateTable,2,FALSE)")
    b.write("A7", "Gross"); b.write_formula("B7", "=B5*(1+B6)*(1-Discount)")
    b.write("A8", "With VAT"); b.write_formula("B8", "=B7*(1+VAT)")
    b.write("A9", "Tier bonus"); b.write_formula("B9", '=IF(B2="Gold",50,IF(B2="Silver",20,0))')
    b.write("A10", "Final"); b.write_formula("B10", "=ROUND(B8+B9,2)")
    b.write("A11", "Lookup"); b.write_formula("B11", '=IFERROR(VLOOKUP("Z",RateTable,2,FALSE),"none")')
    b.write("A12", "Text"); b.write_formula("B12", '=UPPER(B2)&"-"&LOWER(B1)&"-"&TRIM("  x  ")&MID("abcdef",2,3)')
    b.write("A13", "Count"); b.write_formula("B13", '=COUNTIF(Rates!B2:B5,">0.07")')
    b.write("A14", "Not"); b.write_formula("B14", "=NOT(B10>2000)")
    b.write("A15", "Hyperlink"); b.write_url("B15", "https://example.com/terms", string="Terms")
    red = wb.add_format({"bg_color": "#FFC7CE"})
    b.conditional_format("B10", {"type": "cell", "criteria": ">", "value": 1200, "format": red})
    b.conditional_format("B5:B10", {"type": "2_color_scale"})
    b.conditional_format("A1:A14", {"type": "text", "criteria": "containing", "value": "a", "format": wb.add_format({"italic": True})})
    b.conditional_format("B4", {"type": "cell", "criteria": ">", "value": 46174, "format": red})
    b.conditional_format("B1:B14", {"type": "blanks", "format": red})
    b.conditional_format("B5:B10", {"type": "data_bar", "data_bar_2010": True, "bar_negative_color": "#FF0000"})
    b.set_column("A:A", 14); b.set_column("B:B", 16)
    b.activate()
    wb.close()

finance_model(); sales_10k(); table_refs(); cjk(); charts(); names_cf_dv()
print("ok")
