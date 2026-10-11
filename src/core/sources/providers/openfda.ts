/**
 * openFDA's drug labels (https://open.fda.gov/apis/drug/label/): the FDA's
 * labels with each section in a field of its own. Content is CC0 unless
 * marked (https://open.fda.gov/terms/). Without a key it allows 240 requests
 * a minute and 1,000 a day from one address; a key the User has, kept through
 * `secrets.ts` and sent in a header, never in the address, raises the day to
 * 120,000 (https://open.fda.gov/apis/authentication/). Every answer carries a
 * disclaimer, which is shown with the result. A label's sections come flat:
 * a subsection such as 5.1 is inside its section's text, which DailyMed's
 * structured label (./dailymed) keeps apart, so that gives the text first
 * when it is allowed.
 */
import { isRecord } from "../../errors";
import { SourceError } from "../http";
import type {
  FullText,
  FullTextLocation,
  ProviderContext,
  ProviderWork,
  SourceProvider,
  SourceQuery,
  TextSection,
} from "../types";
import { aboutLabel, compactDate, companyName, jsonObject, sentenceCase, tidy } from "./common";

export const OPENFDA_URL = "https://api.fda.gov";

/** 240 a minute is the limit; a search here is one request, and a label two. */
const INTERVAL_MS = 600;

const SET_ID = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i;

/** The label fields that are sections, in the order a label reads, with their headings. Tables and other fields are left out. */
const SECTIONS: readonly (readonly [field: string, heading: string])[] = [
  ["boxed_warning", "Boxed warning"],
  ["indications_and_usage", "Indications and usage"],
  ["dosage_and_administration", "Dosage and administration"],
  ["dosage_forms_and_strengths", "Dosage forms and strengths"],
  ["contraindications", "Contraindications"],
  ["warnings_and_cautions", "Warnings and precautions"],
  ["warnings", "Warnings"],
  ["precautions", "Precautions"],
  ["general_precautions", "General precautions"],
  ["adverse_reactions", "Adverse reactions"],
  ["drug_interactions", "Drug interactions"],
  ["use_in_specific_populations", "Use in specific populations"],
  ["pregnancy", "Pregnancy"],
  ["labor_and_delivery", "Labor and delivery"],
  ["nursing_mothers", "Nursing mothers"],
  ["pediatric_use", "Pediatric use"],
  ["geriatric_use", "Geriatric use"],
  ["overdosage", "Overdosage"],
  ["drug_abuse_and_dependence", "Drug abuse and dependence"],
  ["description", "Description"],
  ["clinical_pharmacology", "Clinical pharmacology"],
  ["mechanism_of_action", "Mechanism of action"],
  ["pharmacodynamics", "Pharmacodynamics"],
  ["pharmacokinetics", "Pharmacokinetics"],
  ["nonclinical_toxicology", "Nonclinical toxicology"],
  [
    "carcinogenesis_and_mutagenesis_and_impairment_of_fertility",
    "Carcinogenesis, mutagenesis, impairment of fertility",
  ],
  ["clinical_studies", "Clinical studies"],
  ["references", "References"],
  ["how_supplied", "How supplied"],
  ["storage_and_handling", "Storage and handling"],
  ["information_for_patients", "Information for patients"],
  ["spl_medguide", "Medication guide"],
  ["purpose", "Purpose"],
  ["active_ingredient", "Active ingredient"],
  ["spl_unclassified_section", "Other"],
];

const strings = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((each): each is string => typeof each === "string") : [];

/** A field's first value: openFDA gives most as lists of one, and a few (set_id, version) as text. */
const first = (value: unknown) =>
  tidy(typeof value === "string" ? value : (strings(value)[0] ?? ""));

/** A section heading with its number, when the label's text starts with it ("5 WARNINGS AND PRECAUTIONS …"). */
export function numberedHeading(heading: string, text: string): string {
  const match = /^(\d+(?:\.\d+)*)\s+(.*)$/s.exec(text.trim());
  if (!match?.[1] || !match[2]) return heading;
  // Compared by letters: a label may write "DOSAGE & ADMINISTRATION" for "Dosage and administration".
  const letters = (value: string) => value.toLowerCase().replace(/[^a-z]/g, "");
  return letters(match[2]).startsWith(letters(heading).slice(0, 6))
    ? `${match[1]} ${heading}`
    : heading;
}

/** The sections of one openFDA label result. */
export function labelSections(label: Record<string, unknown>): TextSection[] {
  const sections: TextSection[] = [];
  for (const [field, heading] of SECTIONS) {
    const paragraphs = strings(label[field]).map(tidy).filter(Boolean);
    if (paragraphs.length === 0) continue;
    sections.push({
      heading: numberedHeading(heading, paragraphs[0] ?? ""),
      level: 1,
      paragraphs,
    });
  }
  return sections;
}

/** "Zituvimet (sitagliptin and metformin hydrochloride)": the brand, then the generic name. */
function titleOf(openfda: Record<string, unknown>): string {
  const brand = sentenceCase(first(openfda.brand_name));
  const generic = sentenceCase(first(openfda.generic_name));
  if (brand && generic && brand.toLowerCase() !== generic.toLowerCase()) {
    return `${brand} (${generic.toLowerCase()})`;
  }
  return brand || generic;
}

/** The words as an openFDA search: a name, in any of the three fields that name a drug. */
export function labelSearch(query: SourceQuery): string | null {
  const words = query.text.match(/[\p{L}\p{N}][\p{L}\p{N}-]*/gu);
  if (!words) return null;
  const phrase = `"${words.join(" ")}"`;
  return ["generic_name", "brand_name", "substance_name"]
    .map((field) => `openfda.${field}:${phrase}`)
    .join(" OR ");
}

export function createOpenfda(
  baseUrl: string = OPENFDA_URL,
  key: () => Promise<string | null> = async () => null,
): SourceProvider {
  const terms = { url: "https://open.fda.gov/terms/", intervalMs: INTERVAL_MS };

  /** One query of the label endpoint: its results and the disclaimer; none for "No matches found". */
  async function labels(search: string, limit: number, context: ProviderContext) {
    const apiKey = await key();
    let body: string;
    try {
      body = await context.http.text(
        `${baseUrl}/drug/label.json?${new URLSearchParams({ search, limit: String(limit) })}`,
        {
          lane: "openfda",
          intervalMs: terms.intervalMs,
          accept: "application/json",
          signal: context.signal,
          timeoutMs: 40_000,
          ...(apiKey && { headers: { "X-Api-Key": apiKey } }),
        },
      );
    } catch (error) {
      // openFDA answers 404 when nothing matches.
      if (error instanceof SourceError && error.reason === "not-found") {
        return { results: [], notice: null };
      }
      throw error;
    }
    const answer = jsonObject(body, "openFDA");
    const meta = isRecord(answer.meta) ? answer.meta : {};
    return {
      results: Array.isArray(answer.results) ? answer.results.filter(isRecord) : [],
      notice: first([meta.disclaimer]) || null,
    };
  }

  const addressOf = (setId: string) =>
    `${baseUrl}/drug/label.json?${new URLSearchParams({ search: `set_id:"${setId}"` })}`;

  function fromLabel(label: Record<string, unknown>, notice: string | null): ProviderWork | null {
    const openfda = isRecord(label.openfda) ? label.openfda : {};
    const setId = first(label.set_id).toLowerCase();
    const title = titleOf(openfda);
    if (!setId || !title) return null;
    const manufacturer = companyName(first(openfda.manufacturer_name));
    const effective = compactDate(first(label.effective_time));
    const copy: FullTextLocation = {
      format: "sections",
      url: addressOf(setId),
      provider: "openfda",
      version: "published",
      licence: "cc0",
      versionDate: effective,
    };
    return {
      ids: { setid: setId },
      fields: {
        type: "document",
        title,
        authors: manufacturer ? [{ literal: manufacturer }] : [],
        ...(effective && { issued: effective }),
        containerTitle: "Drug label (openFDA)",
        publisher: "U.S. Food and Drug Administration",
        language: "en",
      },
      citedBy: null,
      versionDate: effective,
      fullText: [copy],
      landingUrl: `https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=${setId}`,
      ...(notice && { notice }),
    };
  }

  const worksOf = (found: { results: Record<string, unknown>[]; notice: string | null }) =>
    found.results
      .map((label) => fromLabel(label, found.notice))
      .filter((work): work is ProviderWork => work !== null);

  return {
    id: "openfda",
    name: "openFDA",
    service: { id: "https://api.fda.gov", name: "openFDA" },
    terms,
    resolves: ["setid"],
    searchedByDefault: false,
    takesKey: true,
    async search(query, context) {
      const search = labelSearch(query);
      return search ? worksOf(await labels(search, query.limit, context)) : [];
    },
    async resolve(identifier, context) {
      if (identifier.kind !== "setid") return null;
      return worksOf(await labels(`set_id:"${identifier.id}"`, 1, context))[0] ?? null;
    },
    async fetchFullText(location, context): Promise<FullText> {
      const setId = SET_ID.exec(decodeURIComponent(location.url))?.[0];
      if (!setId) throw new SourceError("not-open", `${location.url} isn't an openFDA label.`);
      const { results, notice } = await labels(`set_id:"${setId}"`, 1, context);
      const label = results[0];
      const sections = label ? labelSections(label) : [];
      if (!label || sections.length === 0) {
        throw new SourceError("not-open", "openFDA has no sections for this label.");
      }
      const work = fromLabel(label, notice);
      return {
        format: "sections",
        title: work?.fields.title ?? "Drug label",
        sections: [
          aboutLabel({
            source: "Drug label from openFDA",
            setId,
            version: first(label.version),
            effective: compactDate(first(label.effective_time)),
            notice,
          }),
          ...sections,
        ],
        url: location.url,
        licence: "cc0",
        versionDate: compactDate(first(label.effective_time)),
      };
    },
  };
}
