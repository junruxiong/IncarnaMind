import { documentFileUrl } from "../../../shared/documentViewer";

/** The main process has no such live Document: it was deleted, or its file is gone. */
export class MissingDocumentError extends Error {
  override name = "MissingDocumentError";
}

/**
 * Reads a Document's whole file from the main process's document protocol.
 * XMLHttpRequest rather than fetch: the protocol is registered with only the
 * privileges pdf.js needs, and fetch isn't one of them.
 */
export function loadDocumentBytes(documentId: string, signal: AbortSignal): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("GET", documentFileUrl(documentId));
    request.responseType = "arraybuffer";
    request.onload = () => {
      if (request.status === 200) resolve(new Uint8Array(request.response as ArrayBuffer));
      else if (request.status === 404) reject(new MissingDocumentError("The Document is gone."));
      else reject(new Error(`The Document's file couldn't be read (status ${request.status}).`));
    };
    request.onerror = () => reject(new Error("The Document's file couldn't be read."));
    request.onabort = () => reject(new DOMException("Aborted", "AbortError"));
    signal.addEventListener("abort", () => request.abort(), { once: true });
    request.send();
  });
}
