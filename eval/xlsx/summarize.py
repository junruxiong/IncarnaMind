"""Counts the spike's decision-rule measures from the judges' files, as Markdown.

  python3 -I summarize.py <rt folder with set/judge-noedit.json, judge-edit.json, lo.json, ours.json> [<others folder>]

Every count says its denominator: workbooks the judge could read, or cells compared.
"""
import json, os, sys

SETS = ("repo", "gen", "excel")
PARTS = ("charts", "pivots", "tables", "comments", "pictures", "drawings", "slicers", "vba", "externalLinks")
COUNTED = ("conditional formats", "data validation", "defined names")
RAW = ("x14 conditional formats", "x14 validations", "sparklines")

def load(path):
    return json.load(open(path)) if os.path.exists(path) else []

def judged(rows):
    return [r for r in rows if "judge_cannot_read_original" not in r]

def intact(row, feature):
    k = row.get("kept", {}).get(feature)
    if k is None: return None
    if feature in PARTS: return None if k[0] == 0 else k[0] == k[2]
    if feature in COUNTED: return None if k[0] == 0 else k[0] == k[1]
    return None if k[0] == 0 else k[0] == k[1]

def measures(rows):
    rows = judged(rows)
    opened = [r for r in rows if r.get("opens")]
    out = {"files": len(rows), "opens": len(opened)}
    out["values_and_formulas_kept"] = sum(1 for r in opened if r["values_lost"] == 0 and r["formulas_lost"] == 0)
    out["formats_kept"] = sum(1 for r in opened if r["formats_lost"] == 0 and not r["layout_problems"])
    out["cells"] = sum(r.get("cells", 0) for r in opened)
    out["cells_format_lost"] = sum(r.get("formats_lost", 0) for r in opened)
    out["layout_problem_files"] = sum(1 for r in opened if r["layout_problems"])
    features = {}
    for f in PARTS + COUNTED + RAW:
        have = [r for r in rows if intact(r, f) is not None]
        features[f] = (len(have), sum(1 for r in have if intact(r, f)))
    out["features"] = features
    out["not_opened"] = [r["file"] for r in rows if not r.get("opens")][:10]
    return out

def line(label, kept, total):
    return f"| {label} | {kept} of {total} | {100 * kept / total:.1f}% |" if total else f"| {label} | none in the corpus | |"

def section(title, copies):
    print(f"\n### {title}\n")
    print("| Measure | Count | Share |\n|---|---|---|")
    totals = {"files": 0, "opens": 0, "values_and_formulas_kept": 0, "formats_kept": 0}
    features = {}
    cells = lost = 0
    for name in SETS:
        m = measures(load(copies(name)))
        for k in totals: totals[k] += m[k]
        cells += m["cells"]; lost += m["cells_format_lost"]
        for f, (a, b) in m["features"].items():
            x = features.get(f, (0, 0)); features[f] = (x[0] + a, x[1] + b)
    print(line("Workbooks the judge opens", totals["opens"], totals["files"]))
    print(line("Every value and formula kept", totals["values_and_formulas_kept"], totals["opens"]))
    print(line("Number formats, styles, merged cells, widths, sheet names and order kept", totals["formats_kept"], totals["opens"]))
    print(line("Cells whose look is the same", cells - lost, cells))
    for f, (a, b) in features.items():
        print(line(f"{f} intact (of workbooks that have them)", b, a))

def main():
    rt = sys.argv[1]
    others = sys.argv[2] if len(sys.argv) > 2 else None
    print("## Excel editing spike: decision-rule measures\n")
    section("Save with no edit", lambda s: os.path.join(rt, s, "judge-noedit.json"))
    section("Save after one edit", lambda s: os.path.join(rt, s, "judge-edit.json"))
    if others:
        for tool in ("exceljs", "sheetjs"):
            section(f"For comparison: {tool}, no edit", lambda s, t=tool: os.path.join(others, s, f"judge-{t}.json"))
    print("\n### One edit, judged by LibreOffice\n")
    cells = agree = files = 0
    for name in SETS:
        lo = json.load(open(os.path.join(rt, name, "lo.json"))) if os.path.exists(os.path.join(rt, name, "lo.json")) else {"edits": [], "opens": {}}
        for e in lo["edits"]:
            files += 1; cells += e["cells"]; agree += e["agree"]
    print(f"{files} edited workbooks, {agree} of {cells} dependent cells agree with LibreOffice ({100 * agree / max(cells, 1):.1f}%).")

main()
