import type { ChatModelChoice, ProviderError } from "../api";

/** An in-app folder. The legacy Group API name keeps saved assignments compatible. */
export interface LibraryGroup {
  id: string;
  name: string;
  description: string;
  createdAt: string;
  updatedAt: string;
}

/** Explicit selection: classification never falls back to another provider. */
export type LibraryClassifier =
  | { kind: "chat"; choice: ChatModelChoice }
  | { kind: "jev" }
  | { kind: "auto"; baseUrl: string }
  | { kind: "ollama"; baseUrl: string; modelId: string; usePageImages?: boolean };

export interface ClassificationModel {
  id: string;
  images: boolean;
  reason: "text" | "visual" | "slow" | "memory" | "unavailable" | "selected";
}

export interface LibrarySettings {
  classifier: LibraryClassifier | null;
  automatic: boolean;
}

export type ClassificationStatus = "pending" | "classifying" | "classified" | "waiting" | "failed";

export interface DocumentGroupAssignment {
  documentId: string;
  groupId: string | null;
  source: "automatic" | "user";
  status: ClassificationStatus;
  error: ProviderError | null;
  model: ClassificationModel | null;
}

export interface LibrarySnapshot {
  groups: LibraryGroup[];
  assignments: DocumentGroupAssignment[];
  settings: LibrarySettings;
}

export interface LibraryGroupInput {
  name: string;
  description: string;
}
