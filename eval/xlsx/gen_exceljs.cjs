// Test workbooks written by ExcelJS (MIT): a pivot table, shared and array formulas. Test data only.
const ExcelJS = require("exceljs");
const out = process.argv[2];
async function pivot() {
  const wb = new ExcelJS.Workbook();
  const data = wb.addWorksheet("Sales");
  data.addRow(["Region", "Product", "Quarter", "Units", "Revenue"]);
  const regions = ["North", "South", "East", "West"], products = ["Alpha", "Beta", "Gamma"];
  let n = 0;
  for (const q of ["Q1", "Q2", "Q3", "Q4"]) for (const r of regions) for (const p of products) {
    n++; const units = (n * 37) % 90 + 10; data.addRow([r, p, q, units, units * (12.5 + (n % 7))]);
  }
  data.getColumn(5).numFmt = "#,##0.00";
  data.getCell("G1").value = "Total"; data.getCell("H1").value = { formula: "SUM(E2:E49)", result: 0 };
  data.getCell("G2").value = "North"; data.getCell("H2").value = { formula: 'SUMIFS(E2:E49,A2:A49,"North")', result: 0 };
  const pv = wb.addWorksheet("Pivot");
  pv.addPivotTable({ sourceSheet: data, rows: ["Region"], columns: ["Quarter"], values: ["Revenue"], metric: "sum" });
  await wb.xlsx.writeFile(out + "/ej-pivot.xlsx");
}
async function shared() {
  const wb = new ExcelJS.Workbook();
  const ws = wb.addWorksheet("Ledger");
  ws.addRow(["Item", "Qty", "Price", "Amount", "Running"]);
  for (let i = 1; i <= 200; i++) ws.addRow([`Item ${i}`, (i * 7) % 23 + 1, Math.round(((i * 13) % 97 + 3) * 100) / 100]);
  ws.getCell("D2").value = { formula: "B2*C2", result: 0, shareType: "shared", ref: "D2:D201" };
  for (let r = 3; r <= 201; r++) ws.getCell(`D${r}`).value = { sharedFormula: "D2", result: 0 };
  ws.getCell("E2").value = { formula: "D2", result: 0 };
  ws.getCell("E3").value = { formula: "E2+D3", result: 0, shareType: "shared", ref: "E3:E201" };
  for (let r = 4; r <= 201; r++) ws.getCell(`E${r}`).value = { sharedFormula: "E3", result: 0 };
  ws.getCell("G1").value = "Total"; ws.getCell("H1").value = { formula: "SUM(D2:D201)", result: 0 };
  ws.getCell("G2").value = "Weighted"; ws.getCell("H2").value = { formula: "SUM(B2:B201*C2:C201)", result: 0, shareType: "array", ref: "H2" };
  ws.getCell("G3").value = "Check"; ws.getCell("H3").value = { formula: "IF(ROUND(H1-H2,6)=0,\"OK\",\"Mismatch\")", result: "OK" };
  ws.getColumn(3).numFmt = "0.00"; ws.getColumn(4).numFmt = "#,##0.00"; ws.getColumn(5).numFmt = "#,##0.00";
  await wb.xlsx.writeFile(out + "/ej-shared-formulas.xlsx");
}
shared().then(() => console.log("ok")).catch((e) => { console.error(e); process.exit(1); });
