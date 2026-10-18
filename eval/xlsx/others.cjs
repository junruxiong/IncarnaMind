// The same no-edit round trip through ExcelJS (MIT) and SheetJS Community Edition (Apache-2.0).
const fs = require("fs"), path = require("path");
const ExcelJS = require("exceljs");
const XLSX = require("xlsx");
const [input, output] = process.argv.slice(2);
for (const d of ["exceljs", "sheetjs"]) fs.mkdirSync(path.join(output, d), { recursive: true });
(async () => {
  const results = [];
  for (const name of fs.readdirSync(input).filter((n) => n.endsWith(".xlsx")).sort()) {
    const bytes = fs.readFileSync(path.join(input, name));
    const r = { file: name };
    try {
      let t = performance.now();
      const wb = new ExcelJS.Workbook();
      await wb.xlsx.load(bytes);
      r.exceljsOpenMs = Math.round(performance.now() - t);
      fs.writeFileSync(path.join(output, "exceljs", name), Buffer.from(await wb.xlsx.writeBuffer()));
    } catch (e) { r.exceljsError = String(e.message || e).slice(0, 200); }
    try {
      let t = performance.now();
      const wb = XLSX.read(bytes, { cellStyles: true, cellFormula: true, cellNF: true, cellDates: false, bookVBA: true });
      r.sheetjsOpenMs = Math.round(performance.now() - t);
      fs.writeFileSync(path.join(output, "sheetjs", name), XLSX.write(wb, { type: "buffer", bookType: "xlsx", cellStyles: true, bookVBA: true }));
    } catch (e) { r.sheetjsError = String(e.message || e).slice(0, 200); }
    results.push(r);
  }
  fs.writeFileSync(path.join(output, "others.json"), JSON.stringify(results, null, 1));
  console.log(results.length, "done", results.filter((r) => r.exceljsError).length, "exceljs errors", results.filter((r) => r.sheetjsError).length, "sheetjs errors");
})();
