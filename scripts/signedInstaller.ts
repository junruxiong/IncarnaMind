// Puts the Windows installer SignPath signed in place of the one electron-builder built
// (docs/releasing.md, Windows signing). Signing changes the file, so its blockmap and its sha512
// and size in latest.yml, which electron-builder wrote for the unsigned file, are made again:
// auto-update refuses a download whose hash doesn't match latest.yml.
// Usage: node scripts/signedInstaller.ts <signed installer> <dist folder>
import { createHash } from "node:crypto";
import { copyFile, readFile, stat, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildBlockMap } from "app-builder-lib/out/targets/blockmap/blockmap.js";

export async function applySignedInstaller(
  signedInstaller: string,
  distDir: string,
): Promise<void> {
  const installer = join(distDir, basename(signedInstaller));
  const metadataFile = join(distDir, "latest.yml");
  const unsignedHash = createHash("sha512")
    .update(await readFile(installer))
    .digest("base64");
  const unsignedSize = `size: ${(await stat(installer)).size}\n`;
  const metadata = await readFile(metadataFile, "utf8");
  if (!metadata.includes(unsignedHash) || !metadata.includes(unsignedSize)) {
    throw new Error(
      `${metadataFile} doesn't describe ${installer}, so it can't be updated for the signed one.`,
    );
  }

  await copyFile(signedInstaller, installer);
  const signed = await buildBlockMap(installer, "gzip", `${installer}.blockmap`);
  await writeFile(
    metadataFile,
    metadata
      .replaceAll(unsignedHash, signed.sha512)
      .replace(unsignedSize, `size: ${(await stat(installer)).size}\n`),
  );
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [signedInstaller, distDir] = process.argv.slice(2);
  if (!signedInstaller || !distDir) {
    console.error("Usage: node scripts/signedInstaller.ts <signed installer> <dist folder>");
    process.exit(1);
  }
  await applySignedInstaller(signedInstaller, distDir);
  console.log(`${join(distDir, basename(signedInstaller))} is the signed installer now.`);
}
