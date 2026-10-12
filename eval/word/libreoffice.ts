import { execFile, execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const TIMEOUT_MS = 3 * 60_000;

/** The LibreOffice binary: INCARNAMIND_SOFFICE, else `soffice` or `libreoffice` on the PATH. */
export function findLibreOffice(): string | undefined {
  const named = process.env.INCARNAMIND_SOFFICE;
  if (named) return named;
  for (const command of ["soffice", "libreoffice"]) {
    try {
      execFileSync(command, ["--version"], { stdio: "ignore", timeout: 60_000 });
      return command;
    } catch {
      // not this one
    }
  }
  return undefined;
}

export interface Conversion {
  opened: boolean;
  /** Size of the PDF LibreOffice wrote, when it did. */
  pdfBytes: number;
  /** What went wrong, when `opened` is false. */
  problem?: string;
}

/**
 * Opens a file in LibreOffice headless by converting it to PDF, with a profile of its own in a
 * temporary folder so the User's LibreOffice settings and documents are never touched.
 */
export function convertToPdf(soffice: string, file: string): Promise<Conversion> {
  const work = mkdtempSync(join(tmpdir(), "incarnamind-word-"));
  const args = [
    `-env:UserInstallation=file://${join(work, "profile")}`,
    "--headless",
    "--norestore",
    "--convert-to",
    "pdf",
    "--outdir",
    join(work, "out"),
    file,
  ];
  return new Promise((resolve) => {
    execFile(soffice, args, { timeout: TIMEOUT_MS }, (error, _stdout, stderr) => {
      let result: Conversion;
      try {
        const pdf = readdirSync(join(work, "out")).find((name) => name.endsWith(".pdf"));
        const pdfBytes = pdf ? statSync(join(work, "out", pdf)).size : 0;
        result =
          pdfBytes > 0
            ? { opened: true, pdfBytes }
            : {
                opened: false,
                pdfBytes,
                problem: error?.message ?? (stderr.trim() || "no PDF written"),
              };
      } catch {
        result = { opened: false, pdfBytes: 0, problem: error?.message ?? "no PDF written" };
      }
      rmSync(work, { recursive: true, force: true });
      resolve(result);
    });
  });
}
