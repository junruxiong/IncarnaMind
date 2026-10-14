/**
 * A password-protected PDF for tests (#763): one page of Helvetica lines,
 * encrypted with the PDF standard security handler's revision 2 (RC4, 40-bit
 * key, ISO 32000-1 §7.6.3), which pdf.js opens with the password and refuses
 * without it. No PDF library writes one, so it is written here, byte by byte.
 */
import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";

/** The standard security handler's padding (ISO 32000-1, Algorithm 2). */
const PADDING = Buffer.from(
  "28bf4e5e4e758a4164004e56fffa01082e2e00b6d0683e802f0ca9fe6453697a",
  "hex",
);

const md5 = (...parts: Buffer[]) => createHash("md5").update(Buffer.concat(parts)).digest();

function rc4(key: Buffer, data: Buffer): Buffer {
  const state = Array.from({ length: 256 }, (_, index) => index);
  let j = 0;
  for (let i = 0; i < 256; i++) {
    j = (j + (state[i] as number) + (key[i % key.length] as number)) & 255;
    [state[i], state[j]] = [state[j] as number, state[i] as number];
  }
  const out = Buffer.alloc(data.length);
  let i = 0;
  j = 0;
  for (let at = 0; at < data.length; at++) {
    i = (i + 1) & 255;
    j = (j + (state[i] as number)) & 255;
    [state[i], state[j]] = [state[j] as number, state[i] as number];
    const k = state[((state[i] as number) + (state[j] as number)) & 255] as number;
    out[at] = (data[at] as number) ^ k;
  }
  return out;
}

const padded = (password: string) =>
  Buffer.concat([Buffer.from(password, "latin1"), PADDING]).subarray(0, 32);

/** The PDF's bytes: `lines` on one page, opened only with `password`. */
export function lockedPdf(lines: readonly string[], password: string): Buffer {
  const permissions = -4;
  const id = createHash("md5").update(lines.join("\n")).digest();
  const owner = rc4(md5(padded(password)).subarray(0, 5), padded(password));
  const p = Buffer.alloc(4);
  p.writeInt32LE(permissions);
  const fileKey = md5(padded(password), owner, p, id).subarray(0, 5);
  const user = rc4(fileKey, PADDING);
  const objectKey = (number: number) => {
    const suffix = Buffer.from([number & 255, (number >> 8) & 255, (number >> 16) & 255, 0, 0]);
    return md5(fileKey, suffix).subarray(0, 10);
  };

  const literal = (text: string) => text.replace(/[\\()]/g, (char) => `\\${char}`);
  const content = Buffer.from(
    `BT /F1 12 Tf 72 720 Td ${lines.map((line) => `(${literal(line)}) Tj 0 -14 Td`).join(" ")} ET`,
    "latin1",
  );
  const stream = rc4(objectKey(5), content);
  const objects: Buffer[] = [
    Buffer.from("<< /Type /Catalog /Pages 2 0 R >>", "latin1"),
    Buffer.from("<< /Type /Pages /Kids [4 0 R] /Count 1 >>", "latin1"),
    Buffer.from(
      "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
      "latin1",
    ),
    Buffer.from(
      "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>",
      "latin1",
    ),
    Buffer.concat([
      Buffer.from(`<< /Length ${stream.length} >>\nstream\n`, "latin1"),
      stream,
      Buffer.from("\nendstream", "latin1"),
    ]),
    Buffer.from(
      `<< /Filter /Standard /V 1 /R 2 /O <${owner.toString("hex")}> /U <${user.toString("hex")}> /P ${permissions} >>`,
      "latin1",
    ),
  ];
  const parts: Buffer[] = [Buffer.from("%PDF-1.4\n%\xE2\xE3\xCF\xD3\n", "latin1")];
  const offsets: number[] = [];
  let length = parts[0]?.length ?? 0;
  objects.forEach((body, index) => {
    offsets.push(length);
    const object = Buffer.concat([
      Buffer.from(`${index + 1} 0 obj\n`, "latin1"),
      body,
      Buffer.from("\nendobj\n", "latin1"),
    ]);
    parts.push(object);
    length += object.length;
  });
  const xref = [
    "xref",
    `0 ${objects.length + 1}`,
    "0000000000 65535 f ",
    ...offsets.map((offset) => `${String(offset).padStart(10, "0")} 00000 n `),
    "trailer",
    `<< /Size ${objects.length + 1} /Root 1 0 R /Encrypt 6 0 R /ID [<${id.toString("hex")}> <${id.toString("hex")}>] >>`,
    "startxref",
    String(length),
    "%%EOF",
    "",
  ].join("\n");
  parts.push(Buffer.from(xref, "latin1"));
  return Buffer.concat(parts);
}

/** Writes a locked PDF into `folder`, a test's temporary one; returns its path. */
export async function writeLockedPdf(
  folder: string,
  name: string,
  lines: readonly string[],
  password: string,
): Promise<string> {
  const path = join(folder, name);
  await writeFile(path, lockedPdf(lines, password));
  return path;
}
