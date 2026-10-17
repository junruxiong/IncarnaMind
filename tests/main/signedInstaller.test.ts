import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";
import { expect, onTestFinished, test } from "vitest";
import { applySignedInstaller } from "../../scripts/signedInstaller";

const NAME = "IncarnaMind-1.2.3-win-x64.exe";

/** latest.yml as electron-builder writes it for an NSIS installer. */
function latestYml(installer: Buffer): string {
  const sha512 = createHash("sha512").update(installer).digest("base64");
  return [
    "version: 1.2.3",
    "files:",
    `  - url: ${NAME}`,
    `    sha512: ${sha512}`,
    `    size: ${installer.length}`,
    `path: ${NAME}`,
    `sha512: ${sha512}`,
    "releaseDate: '2026-10-16T09:00:00.000Z'",
    "",
  ].join("\n");
}

async function release() {
  const folder = await mkdtemp(join(tmpdir(), "incarnamind-signed-installer-"));
  onTestFinished(() => rm(folder, { recursive: true, force: true }));
  const dist = join(folder, "dist");
  const signedPath = join(folder, "signed", NAME);
  await mkdir(dist);
  await mkdir(join(folder, "signed"));
  const unsigned = Buffer.from(Array.from({ length: 300_000 }, (_, index) => (index * 7) % 251));
  const signed = Buffer.concat([unsigned, Buffer.from("an Authenticode signature")]);
  await writeFile(join(dist, NAME), unsigned);
  await writeFile(join(dist, `${NAME}.blockmap`), "the unsigned installer's blockmap");
  await writeFile(signedPath, signed);
  return { dist, signedPath, unsigned, signed };
}

test("auto-update gets the signed installer, with its own hash, size and blockmap", async () => {
  const { dist, signedPath, unsigned, signed } = await release();
  await writeFile(join(dist, "latest.yml"), latestYml(unsigned));

  await applySignedInstaller(signedPath, dist);

  expect(await readFile(join(dist, NAME))).toEqual(signed);
  expect(await readFile(join(dist, "latest.yml"), "utf8")).toBe(latestYml(signed));
  const blockmap: { files: { sizes: number[] }[] } = JSON.parse(
    gunzipSync(await readFile(join(dist, `${NAME}.blockmap`))).toString(),
  );
  const sizes = blockmap.files.flatMap((file) => file.sizes);
  expect(sizes.reduce((sum, size) => sum + size, 0)).toBe(signed.length);
});

test("a latest.yml for another installer is left alone, and so is the installer", async () => {
  const { dist, signedPath, unsigned } = await release();
  const other = latestYml(Buffer.from("another build"));
  await writeFile(join(dist, "latest.yml"), other);

  await expect(applySignedInstaller(signedPath, dist)).rejects.toThrow("doesn't describe");

  expect(await readFile(join(dist, NAME))).toEqual(unsigned);
  expect(await readFile(join(dist, "latest.yml"), "utf8")).toBe(other);
});
