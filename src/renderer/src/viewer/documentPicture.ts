import { documentImageUrl } from "../../../shared/documentViewer";

/**
 * Reads a picture a Document shows (a Markdown file's from beside it, a
 * saved web page's own) as a `data:` URL, which the app's policy allows.
 */
export function loadDocumentPicture(
  documentId: string,
  path: string,
  signal: AbortSignal,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("GET", documentImageUrl(documentId, path));
    request.responseType = "blob";
    request.onload = () => {
      if (request.status !== 200) {
        reject(new Error(`status ${request.status}`));
        return;
      }
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(request.response as Blob);
    };
    request.onerror = () => reject(new Error("The picture couldn't be read."));
    signal.addEventListener("abort", () => request.abort(), { once: true });
    request.send();
  });
}
