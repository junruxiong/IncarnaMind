/**
 * Checking a reference list (#711): each entry of a Mind's reference list
 * (./list) looked up in Crossref (./check), so a reference to a work that
 * doesn't exist, or that doesn't say what the work's record says, is
 * flagged. Only the reference is sent, as written, on its own data flow,
 * "reference-check". The User starts a check where a line by its action
 * says what goes to Crossref, so starting it is the consent (ADR-0022);
 * nothing is sent in local mode. A reference checked once isn't asked
 * about again while IncarnaMind runs; one Crossref couldn't answer is.
 */
import type { CheckReferencesResult, DataKind, ReferenceCheck, ReferenceCheckReason } from "../api";
import type { Consent } from "../consent";
import { ConsentDeclinedError } from "../errors";
import type { MindContent } from "../mindContent";
import { parseMindId } from "../minds";
import { CROSSREF } from "../referenceRecords/lookup";
import { SourceError, type SourceHttp } from "../sources/http";
import { checkReference } from "./check";
import { createCrossrefClient } from "./crossref";
import { referenceList } from "./list";
import { citedReference } from "./match";

/** What the "reference-check" flow sends: each reference, as written. */
export const REFERENCE_CHECK_SENDS: readonly DataKind[] = ["references"];

export interface ReferenceChecksOptions {
  consent: Consent;
  localOnly(): boolean;
  http: SourceHttp;
  /** Crossref's API, unless the host says otherwise (tests). */
  crossrefUrl?: string;
  content: Pick<MindContent, "read">;
  progress(event: { mindId: string; done: number; total: number }): void;
  signal: AbortSignal;
}

const reasonOf = (error: unknown): ReferenceCheckReason =>
  error instanceof SourceError
    ? error.reason === "rate-limited"
      ? "rate-limited"
      : error.reason === "unreachable"
        ? "unreachable"
        : "service-error"
    : "service-error";

export function createReferenceChecks(options: ReferenceChecksOptions) {
  const { consent } = options;
  const crossref = createCrossrefClient(options.http, options.crossrefUrl);
  const checked = new Map<string, ReferenceCheck>();
  const running = new Map<string, AbortController>();

  consent.registry.register({
    id: "reference-check",
    sends: REFERENCE_CHECK_SENDS,
    // It goes nowhere until a check asks; then the decision is listed.
    services: async () => [],
  });

  const notChecked = (reference: string, reason: ReferenceCheckReason): ReferenceCheck => ({
    reference,
    mark: "not-checked",
    by: null,
    record: null,
    differences: [],
    reason,
  });

  return {
    async check(mindIdInput: unknown): Promise<CheckReferencesResult> {
      const mindId = parseMindId(mindIdInput);
      const entries = options.content
        .read(mindId, referenceList)
        .map((entry) => ({ ...entry, cited: citedReference(entry.text) }));
      if (entries.length === 0) return { kind: "no-references" };

      running.get(mindId)?.abort();
      const controller = new AbortController();
      running.set(mindId, controller);
      const signal = AbortSignal.any([controller.signal, options.signal]);
      const all = (reason: ReferenceCheckReason): CheckReferencesResult => ({
        kind: "checked",
        references: entries.map((entry) => ({
          blockId: entry.blockId,
          ...notChecked(entry.cited.text, reason),
        })),
      });
      try {
        if (options.localOnly()) return all("local-mode");
        // The action says what goes to Crossref: starting the check allows it.
        consent.choose(["reference-check"], CROSSREF);
        try {
          await consent.ensure("reference-check", CROSSREF);
        } catch (error) {
          if (error instanceof ConsentDeclinedError) return all("declined");
          throw error;
        }
        const references: (ReferenceCheck & { blockId: string })[] = [];
        let stopped: ReferenceCheckReason | null = null;
        for (const [index, entry] of entries.entries()) {
          if (signal.aborted) return { kind: "cancelled" };
          const key = entry.cited.text.toLowerCase();
          let result = checked.get(key) ?? null;
          if (!result && stopped) result = notChecked(entry.cited.text, stopped);
          if (!result) {
            try {
              result = await checkReference(entry.cited, crossref, signal);
              checked.set(key, result);
            } catch (error) {
              if (signal.aborted) return { kind: "cancelled" };
              result = notChecked(entry.cited.text, reasonOf(error));
              // Crossref asked to slow down: the rest wait for another check.
              if (result.reason === "rate-limited") stopped = "rate-limited";
            }
          }
          references.push({ blockId: entry.blockId, ...result });
          options.progress({ mindId, done: index + 1, total: entries.length });
        }
        return { kind: "checked", references };
      } finally {
        if (running.get(mindId) === controller) running.delete(mindId);
      }
    },

    stop(mindIdInput: unknown): void {
      running.get(parseMindId(mindIdInput))?.abort();
    },
  };
}
