import type { DragEvent } from "react";

/**
 * Dragging a Document from the sidebar onto the composer, to add it to the
 * Question's Search scope. It travels as its id under a type of IncarnaMind's
 * own, so the window's file drop and the Folders ignore it. The scope picker
 * ("@") is the way to add one without dragging.
 */
export const DOCUMENT_TYPE = "application/x-incarnamind-document";

/** Starts dragging this Document. */
export function startDocumentDrag(event: DragEvent, documentId: string): void {
  event.dataTransfer.setData(DOCUMENT_TYPE, JSON.stringify([documentId]));
  event.dataTransfer.effectAllowed = "link";
}

export const carriesDocuments = (event: DragEvent): boolean =>
  event.dataTransfer.types.includes(DOCUMENT_TYPE);

/** The Documents a drop carries; none if it isn't one of IncarnaMind's. */
export function droppedDocumentIds(event: DragEvent): string[] {
  try {
    const value = JSON.parse(event.dataTransfer.getData(DOCUMENT_TYPE)) as unknown;
    return Array.isArray(value) ? value.filter((id): id is string => typeof id === "string") : [];
  } catch {
    return [];
  }
}
