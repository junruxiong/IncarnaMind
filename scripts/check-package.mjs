// Checks what electron-builder packed, after `npm run dist:dir` or `npm run dist`:
//   1. Every app.asar under dist/ holds only out/, node_modules/ and package.json at its
//      top level. Anything else (sources, tests, data, untracked files in the checkout)
//      means the `files` patterns in electron-builder.yml let the whole folder in.
//   2. Each app.asar and each installer is within its size budget.
// Usage: node scripts/check-package.mjs [dist-folder]   (default: dist)
// Exit code 1 when a check fails.
import { readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { listPackage } from "@electron/asar";

const ALLOWED_TOP_LEVEL = new Set(["out", "node_modules", "package.json"]);

// Set from the fixed macOS arm64 build (app.asar 130.8 MB) plus about 20%. The asar holds
// JavaScript and pure-JS dependencies; the native libraries are unpacked beside it.
const MAX_ASAR_BYTES = 157_000_000;
// The largest installer of the fixed macOS arm64 build (the dmg, 184.3 MB) plus about 20%.
// The Windows and Linux installers are checked against the same limit.
const MAX_INSTALLER_BYTES = 221_000_000;

const distDir = resolve(process.argv[2] ?? "dist");
const INSTALLER_EXT = /\.(dmg|zip|exe|AppImage)$/;

/** Every file under dir, except what is inside an app.asar.unpacked folder. */
function walk(dir) {
	const found = [];
	for (const entry of readdirSync(dir, { withFileTypes: true })) {
		const path = join(dir, entry.name);
		if (entry.isDirectory()) {
			if (entry.name === "app.asar.unpacked") continue;
			found.push(...walk(path));
		} else {
			found.push(path);
		}
	}
	return found;
}

const mb = (bytes) => `${(bytes / 1e6).toFixed(1)} MB`;
const failures = [];

let files;
try {
	files = walk(distDir);
} catch {
	console.error(`No packaged output at ${distDir}. Run npm run dist:dir first.`);
	process.exit(1);
}

const asars = files.filter((f) => f.endsWith("app.asar"));
if (asars.length === 0) {
	console.error(`No app.asar found under ${distDir}.`);
	process.exit(1);
}

for (const asar of asars) {
	const size = statSync(asar).size;
	const entries = listPackage(asar, { isPack: false });
	const topLevel = new Map();
	for (const entry of entries) {
		const name = entry.replace(/^[\\/]/, "").split(/[\\/]/)[0];
		if (name) topLevel.set(name, (topLevel.get(name) ?? 0) + 1);
	}
	const offenders = [...topLevel.keys()].filter((n) => !ALLOWED_TOP_LEVEL.has(n)).sort();
	console.log(`${asar}: ${mb(size)}, ${entries.length} entries`);
	if (offenders.length > 0) {
		failures.push(
			`${asar} holds files outside out/, node_modules/ and package.json: ${offenders.join(", ")}`,
		);
	}
	if (size > MAX_ASAR_BYTES) {
		failures.push(`${asar} is ${mb(size)}, over the ${mb(MAX_ASAR_BYTES)} budget`);
	}
}

for (const file of files.filter((f) => INSTALLER_EXT.test(f))) {
	const size = statSync(file).size;
	console.log(`${file}: ${mb(size)}`);
	if (size > MAX_INSTALLER_BYTES) {
		failures.push(`${file} is ${mb(size)}, over the ${mb(MAX_INSTALLER_BYTES)} installer budget`);
	}
}

if (failures.length > 0) {
	for (const failure of failures) console.error(`FAIL: ${failure}`);
	process.exit(1);
}
console.log("Package contents and sizes are within bounds.");
