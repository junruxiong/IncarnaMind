/**
 * Serves Document files to the renderer over the `incarnamind-document:`
 * protocol: the Document viewer loads a PDF (pdf.js) or a text file from
 * `documentFileUrl(id)`, and the bytes stream straight from the file, where
 * the User keeps it. A Document whose file is missing or can't be reached is
 * a 404: the viewer then shows the text IncarnaMind kept (`readDocumentText`).
 */
import { protocol } from "electron";
import { type Core, type DocumentKind, NotFoundError } from "../core";
import { DOCUMENT_SCHEME, documentIdFromUrl } from "../shared/documentViewer";

const CONTENT_TYPES: Readonly<Record<DocumentKind, string>> = {
  pdf: "application/pdf",
  // The viewer decodes text itself, the way processing does (UTF-8, UTF-16 or GB18030).
  text: "text/plain",
  markdown: "text/markdown",
};

/** Must run before the app is ready. */
export function registerDocumentScheme(): void {
  protocol.registerSchemesAsPrivileged([
    {
      scheme: DOCUMENT_SCHEME,
      // The least the viewer needs: pdf.js (and the text view) load files with
      // XMLHttpRequest, which only works for a CORS-enabled scheme. Nothing else:
      // not standard, not secure, no fetch, no CSP bypass, no service workers, no storage.
      privileges: { corsEnabled: true },
    },
  ]);
}

/**
 * Answers requests for Document files. Only live Documents are served; anything
 * else is a 404.
 *
 * A built app's page is a local file, whose requests Chromium doesn't subject to
 * CORS. In development the page comes from the dev server, so its origin
 * (`devServerOrigin`) is the one cross-origin reader allowed.
 */
export function serveDocumentFiles(core: Core, devServerOrigin: string | null): void {
  protocol.handle(DOCUMENT_SCHEME, async (request) => {
    const headers: Record<string, string> = {
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
      // Never runs as a page, even if something navigated to it.
      "Content-Security-Policy": "default-src 'none'; sandbox",
    };
    if (devServerOrigin && request.headers.get("Origin") === devServerOrigin) {
      headers["Access-Control-Allow-Origin"] = devServerOrigin;
    }
    if (request.method !== "GET") return new Response(null, { status: 405, headers });
    const documentId = documentIdFromUrl(request.url);
    if (!documentId) return new Response(null, { status: 404, headers });
    try {
      const { document, stream, size } = await core.openDocumentFile(documentId);
      headers["Content-Type"] = CONTENT_TYPES[document.kind];
      // The file's size now: it may have changed since it was indexed.
      headers["Content-Length"] = String(size);
      return new Response(stream, { status: 200, headers });
    } catch (error) {
      if (error instanceof NotFoundError) return new Response(null, { status: 404, headers });
      console.error(error);
      return new Response(null, { status: 500, headers });
    }
  });
}
