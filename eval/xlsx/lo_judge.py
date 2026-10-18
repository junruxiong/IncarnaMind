"""LibreOffice as the judge of the calculator and the copies: run on Linux CI, never on the owner's Mac.

  python3 lo_judge.py <originals folder> <round-trip folder> <out.json>

For every edited copy (<rt>/edit/<name>.xlsx with <name>.json beside it, written by roundtrip.ts):
LibreOffice opens the original and the copy, recalculates everything, and the cells that depend
on the edit are compared with what IncarnaMind's calculator stored in the copy. Every copy, edited
or not, is also opened, to see that LibreOffice reads it.
"""
import json, os, subprocess, sys, tempfile, time
import uno
from com.sun.star.beans import PropertyValue

ERRORS = {532: "#DIV/0!", 503: "#NUM!", 502: "#VALUE!", 519: "#VALUE!", 524: "#REF!", 525: "#NAME?", 32767: "#N/A", 522: "#CIRCULAR", 523: "#NUM!", 504: "#VALUE!", 511: "#VALUE!"}

def prop(name, value):
    p = PropertyValue(); p.Name = name; p.Value = value; return p

def start():
    profile = tempfile.mkdtemp(prefix="lo-profile-")
    subprocess.Popen(["soffice", f"-env:UserInstallation=file://{profile}", "--headless", "--invisible", "--norestore",
                      "--nologo", "--accept=pipe,name=excelspike;urp;"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
    for _ in range(120):
        try:
            ctx = resolver.resolve("uno:pipe,name=excelspike;urp;StarOffice.ComponentContext")
            return ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        except Exception:
            time.sleep(0.5)
    raise SystemExit("LibreOffice didn't start")

def load(desktop, path):
    return desktop.loadComponentFromURL(uno.systemPathToFileUrl(os.path.abspath(path)), "_blank", 0, (prop("Hidden", True),))

def read(doc, ref):
    sheet_name, cell_ref = ref.rsplit("!", 1)
    if not doc.Sheets.hasByName(sheet_name): return {"missing": True}
    cell = doc.Sheets.getByName(sheet_name).getCellRangeByName(cell_ref)
    code = cell.getError()
    if code: return {"error": ERRORS.get(code, f"Err:{code}")}
    from com.sun.star.table.CellContentType import TEXT, FORMULA, EMPTY
    if cell.Type == EMPTY: return None
    if cell.Type == TEXT: return cell.String
    if cell.Type == FORMULA:
        # FormulaResultType2: 1 value, 2 text.
        return cell.String if cell.FormulaResultType2 == 2 else cell.Value
    return cell.Value

def same(a, b):
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.get("error") == b.get("error")
    if isinstance(a, bool): a = 1.0 if a else 0.0
    if isinstance(b, bool): b = 1.0 if b else 0.0
    if a in (None, "") and b in (None, "", 0, 0.0): return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-9 * max(1, abs(a), abs(b))
    return a == b

def main():
    originals, rt, out = sys.argv[1:4]
    desktop = start()
    results = {"opens": {}, "edits": []}
    for folder in ("noedit", "edit"):
        for name in sorted(os.listdir(os.path.join(rt, folder))):
            if not name.endswith(".xlsx"): continue
            try:
                doc = load(desktop, os.path.join(rt, folder, name))
                results["opens"][f"{folder}/{name}"] = doc is not None
                if doc is not None: doc.close(True)
            except Exception as e:
                results["opens"][f"{folder}/{name}"] = False
    for name in sorted(os.listdir(os.path.join(rt, "edit"))):
        if not name.endswith(".json"): continue
        info = json.load(open(os.path.join(rt, "edit", name)))
        xlsx = name[:-5] + ".xlsx"
        entry = {"file": xlsx, "edit": info["edit"], "cells": 0, "agree": 0, "stale": len(info["stale"]), "disagree": []}
        try:
            before = load(desktop, os.path.join(originals, xlsx))
            stored = {ref: read(before, ref) for ref in info["expected"]}
            before.calculateAll()
            lo_before = {ref: read(before, ref) for ref in info["expected"]}
            before.close(True)
            after = load(desktop, os.path.join(rt, "edit", xlsx))
            after.calculateAll()
            stale = set(info["stale"])
            for ref, ours in info["expected"].items():
                if ref in stale: continue
                lo = read(after, ref)
                entry["cells"] += 1
                if same(ours, lo): entry["agree"] += 1
                else:
                    entry["disagree"].append({"ref": ref, "ours": ours, "libreoffice": lo,
                        "stored_before": stored.get(ref), "libreoffice_before": lo_before.get(ref)})
            after.close(True)
        except Exception as e:
            entry["problem"] = str(e)[:300]
        entry["disagree"] = entry["disagree"][:12] + ([{"more": len(entry["disagree"]) - 12}] if len(entry["disagree"]) > 12 else [])
        results["edits"].append(entry)
        print(xlsx, entry["agree"], "/", entry["cells"], flush=True)
    json.dump(results, open(out, "w"), indent=1, default=str)

main()
