import { execFile, execFileSync } from "node:child_process";
import {
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

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
 * temporary folder so the User's LibreOffice settings and documents are never touched. With
 * `keep`, the PDF is also copied there, as `keep`'s path says, for a person to look at.
 */
export function convertToPdf(soffice: string, file: string, keep?: string): Promise<Conversion> {
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
        if (pdf && pdfBytes > 0 && keep) {
          mkdirSync(dirname(keep), { recursive: true });
          copyFileSync(join(work, "out", pdf), keep);
        }
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

/** A Basic module that opens a file, accepts every tracked change, and saves its text. */
const ACCEPT_MODULE = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE script:module PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" "module.dtd">
<script:module xmlns:script="http://openoffice.org/2000/script" script:name="Accept" script:language="StarBasic">
Sub AcceptAll(source As String, target As String)
  Dim opening(0) As New com.sun.star.beans.PropertyValue
  opening(0).Name = "Hidden"
  opening(0).Value = True
  Dim document As Object
  document = StarDesktop.loadComponentFromURL(ConvertToURL(source), "_blank", 0, opening())
  Dim dispatcher As Object
  dispatcher = createUnoService("com.sun.star.frame.DispatchHelper")
  dispatcher.executeDispatch(document.getCurrentController().getFrame(), ".uno:AcceptAllTrackedChanges", "", 0, Array())
  Dim saving(1) As New com.sun.star.beans.PropertyValue
  saving(0).Name = "FilterName"
  saving(0).Value = "Text (encoded)"
  saving(1).Name = "FilterOptions"
  saving(1).Value = "UTF8,LF"
  document.storeToURL(ConvertToURL(target), saving())
  document.close(True)
  StarDesktop.terminate()
End Sub
</script:module>
`;

const library = (kind: "script" | "dialog") =>
  `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE library:library PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" "library.dtd">
<library:library xmlns:library="http://openoffice.org/2000/library" library:name="Standard" library:readonly="false" library:passwordprotected="false">${kind === "script" ? '\n <library:element library:name="Accept"/>' : ""}
</library:library>
`;

const libraries = (kind: "script" | "dialog") =>
  `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE library:libraries PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" "libraries.dtd">
<library:libraries xmlns:library="http://openoffice.org/2000/library" xmlns:xlink="http://www.w3.org/1999/xlink">
 <library:library library:name="Standard" xlink:href="$(USER)/basic/Standard/${kind}.xlb/" xlink:type="simple" library:link="false"/>
</library:libraries>
`;

/**
 * Opens a .docx in LibreOffice headless, accepts every tracked change in it
 * (#107), and returns its text, a paragraph a line: a Basic macro in a
 * profile of its own, in a temporary folder, does it.
 */
export function acceptAllChanges(
  soffice: string,
  file: string,
): Promise<{ text: string } | { problem: string }> {
  const work = mkdtempSync(join(tmpdir(), "incarnamind-word-accept-"));
  const basic = join(work, "profile", "user", "basic");
  mkdirSync(join(basic, "Standard"), { recursive: true });
  writeFileSync(join(basic, "script.xlc"), libraries("script"));
  writeFileSync(join(basic, "dialog.xlc"), libraries("dialog"));
  writeFileSync(join(basic, "Standard", "script.xlb"), library("script"));
  writeFileSync(join(basic, "Standard", "dialog.xlb"), library("dialog"));
  writeFileSync(join(basic, "Standard", "Accept.xba"), ACCEPT_MODULE);
  const target = join(work, "accepted.txt");
  const args = [
    `-env:UserInstallation=file://${join(work, "profile")}`,
    "--headless",
    "--norestore",
    `macro:///Standard.Accept.AcceptAll("${file}","${target}")`,
  ];
  return new Promise((resolve) => {
    execFile(soffice, args, { timeout: TIMEOUT_MS }, (error, _stdout, stderr) => {
      try {
        resolve({ text: readFileSync(target, "utf8") });
      } catch {
        resolve({ problem: error?.message ?? (stderr.trim() || "no text written") });
      } finally {
        rmSync(work, { recursive: true, force: true });
      }
    });
  });
}
