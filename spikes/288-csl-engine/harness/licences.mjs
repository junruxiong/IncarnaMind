// Lists every crate the wasm build resolves, its licence, and whether its code ends up in the
// .wasm file (normal dependencies) or only runs at build time (proc macros and their deps).
// usage: node licences.mjs <metadata.json>
import fs from "node:fs";

const meta = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const pkgs = new Map(meta.packages.map((p) => [p.id, p]));
const nodes = new Map(meta.resolve.nodes.map((n) => [n.id, n]));
const isProcMacro = (id) => pkgs.get(id).targets.some((t) => t.kind.includes("proc-macro"));

// Reachable through normal (non-build, non-dev) edges, without passing through a proc macro.
const inBinary = new Set();
const stack = [meta.resolve.root];
while (stack.length) {
  const id = stack.pop();
  if (inBinary.has(id)) continue;
  inBinary.add(id);
  for (const dep of nodes.get(id).deps) {
    const normal = dep.dep_kinds.some((k) => k.kind === null);
    if (normal && !isProcMacro(dep.pkg)) stack.push(dep.pkg);
  }
}

const rows = [...pkgs.values()]
  .filter((p) => p.id !== meta.resolve.root)
  .map((p) => ({
    name: p.name,
    version: p.version,
    license: p.license ?? `file: ${p.license_file}`,
    where: inBinary.has(p.id) ? "in the .wasm" : "build time only",
    repo: p.repository ?? "",
  }))
  .sort((a, b) => (a.where === b.where ? a.name.localeCompare(b.name) : a.where.localeCompare(b.where)));

const byLicence = {};
for (const r of rows) (byLicence[r.license] ??= []).push(r.name);
console.log(`${rows.length} crates; ${rows.filter((r) => r.where === "in the .wasm").length} compiled into the .wasm`);
for (const [l, names] of Object.entries(byLicence)) console.log(`- ${l} (${names.length})`);
console.log("\n| Crate | Version | Licence | Where |\n|---|---|---|---|");
for (const r of rows) console.log(`| ${r.name} | ${r.version} | ${r.license} | ${r.where} |`);
