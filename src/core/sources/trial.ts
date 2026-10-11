/**
 * One ClinicalTrials.gov study record (API v2) as text by part: what the
 * record says of the study and who runs it, its conditions and design, its
 * arms and interventions, its outcomes, its eligibility criteria, and, when
 * results are posted, the participant flow, outcome measures and adverse
 * events. Each part is a section, so a Citation points at "Eligibility
 * criteria" or "Results: adverse events" and its quote is checked there. A
 * study with no results has none of those parts. Contacts, sites and
 * references are left out.
 */
import { isRecord } from "../errors";
import { tidy } from "./providers/common";
import type { TextSection } from "./types";

type Json = Record<string, unknown>;

const obj = (value: unknown): Json => (isRecord(value) ? value : {});
const list = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);
const str = (value: unknown): string =>
  typeof value === "string" ? tidy(value) : typeof value === "number" ? String(value) : "";
const strs = (value: unknown): string[] => list(value).map(str).filter(Boolean);

/** "RANDOMIZED", "PHASE2" as "Randomized", "Phase 2". */
export function readable(value: string): string {
  const spaced = value.replace(/_/g, " ").toLowerCase();
  return `${spaced.slice(0, 1).toUpperCase()}${spaced.slice(1)}`.replace(/phase(\d)/i, "Phase $1");
}

/** A line "Label: value", or nothing when there is no value. */
const line = (label: string, value: string) => (value ? [`${label}: ${value}`] : []);

/** A paragraph of the record's own text, whose line breaks are kept as paragraphs. */
const blocks = (value: unknown): string[] =>
  typeof value === "string"
    ? value
        .split(/\n\s*\n/)
        .map(tidy)
        .filter(Boolean)
    : [];

export interface TrialRecord {
  nct: string;
  title: string;
  sponsor: string;
  /** ISO 8601 date the record was last updated, as ClinicalTrials.gov posts it. */
  updated: string | null;
  sections: TextSection[];
}

const dateOf = (struct: unknown) => str(obj(struct).date);

function overview(protocol: Json, nct: string): TextSection {
  const id = obj(protocol.identificationModule);
  const status = obj(protocol.statusModule);
  const sponsors = obj(protocol.sponsorCollaboratorsModule);
  const description = obj(protocol.descriptionModule);
  const paragraphs = [
    ...line("NCT number", nct),
    ...line("Brief title", str(id.briefTitle)),
    ...line("Official title", str(id.officialTitle)),
    ...line("Overall status", readable(str(status.overallStatus))),
    ...line("Lead sponsor", str(obj(sponsors.leadSponsor).name)),
    ...line("Start date", dateOf(status.startDateStruct)),
    ...line("Primary completion date", dateOf(status.primaryCompletionDateStruct)),
    ...line("Results first posted", dateOf(status.resultsFirstPostDateStruct)),
    ...line("Record last updated", dateOf(status.lastUpdatePostDateStruct)),
    ...blocks(description.briefSummary),
    ...blocks(description.detailedDescription),
  ];
  return { heading: "Overview", level: 1, paragraphs };
}

function design(protocol: Json): TextSection {
  const module = obj(protocol.designModule);
  const info = obj(module.designInfo);
  const masking = obj(info.maskingInfo);
  const enrollment = obj(module.enrollmentInfo);
  const conditions = obj(protocol.conditionsModule);
  const paragraphs = [
    ...line("Conditions", strs(conditions.conditions).join("; ")),
    ...line("Keywords", strs(conditions.keywords).join("; ")),
    ...line("Study type", readable(str(module.studyType))),
    ...line("Phases", strs(module.phases).map(readable).join(", ")),
    ...line("Allocation", readable(str(info.allocation))),
    ...line("Intervention model", readable(str(info.interventionModel))),
    ...line("Primary purpose", readable(str(info.primaryPurpose))),
    ...line("Masking", readable(str(masking.masking))),
    ...line(
      "Enrollment",
      str(enrollment.count) &&
        `${str(enrollment.count)}${str(enrollment.type) ? ` (${readable(str(enrollment.type))})` : ""}`,
    ),
  ];
  return { heading: "Conditions and design", level: 1, paragraphs };
}

function arms(protocol: Json): TextSection {
  const module = obj(protocol.armsInterventionsModule);
  const paragraphs: string[] = [];
  for (const arm of list(module.armGroups).map(obj)) {
    const names = strs(arm.interventionNames).join("; ");
    paragraphs.push(
      [
        `Arm: ${str(arm.label)}${str(arm.type) ? ` (${readable(str(arm.type))})` : ""}.`,
        str(arm.description),
        names && `Interventions: ${names}.`,
      ]
        .filter(Boolean)
        .join(" "),
    );
  }
  for (const each of list(module.interventions).map(obj)) {
    paragraphs.push(
      [
        `Intervention: ${str(each.name)}${str(each.type) ? ` (${readable(str(each.type))})` : ""}.`,
        str(each.description),
      ]
        .filter(Boolean)
        .join(" "),
    );
  }
  return { heading: "Arms and interventions", level: 1, paragraphs };
}

function outcomes(protocol: Json, key: string, heading: string): TextSection {
  const paragraphs = list(obj(protocol.outcomesModule)[key])
    .map(obj)
    .map((outcome) =>
      [
        `${str(outcome.measure)}.`,
        str(outcome.description),
        str(outcome.timeFrame) && `Time frame: ${str(outcome.timeFrame)}.`,
      ]
        .filter(Boolean)
        .join(" "),
    );
  return { heading, level: 1, paragraphs };
}

function eligibility(protocol: Json): TextSection {
  const module = obj(protocol.eligibilityModule);
  const paragraphs = [
    ...blocks(module.eligibilityCriteria),
    ...line("Sex", readable(str(module.sex))),
    ...line("Ages", [str(module.minimumAge), str(module.maximumAge)].filter(Boolean).join(" to ")),
    ...line("Age groups", strs(module.stdAges).map(readable).join(", ")),
    ...line(
      "Healthy volunteers",
      module.healthyVolunteers === true ? "Yes" : module.healthyVolunteers === false ? "No" : "",
    ),
  ];
  return { heading: "Eligibility criteria", level: 1, paragraphs };
}

/** Group id to title, for the parts of the results that count by group. */
const groupTitles = (groups: unknown): Map<string, string> =>
  new Map(
    list(groups)
      .map(obj)
      .map((group): [string, string] => [str(group.id), str(group.title)]),
  );

function participantFlow(results: Json): TextSection {
  const module = obj(results.participantFlowModule);
  const titles = groupTitles(module.groups);
  const paragraphs = [...blocks(module.preAssignmentDetails), ...blocks(module.recruitmentDetails)];
  for (const period of list(module.periods).map(obj)) {
    paragraphs.push(`Period: ${str(period.title)}.`);
    for (const milestone of list(period.milestones).map(obj)) {
      const counts = list(milestone.achievements)
        .map(obj)
        .map(
          (each) =>
            `${titles.get(str(each.groupId)) ?? str(each.groupId)} ${str(each.numSubjects)}`,
        )
        .join("; ");
      paragraphs.push(`${str(milestone.type)}: ${counts}.`);
    }
  }
  return { heading: "Results: participant flow", level: 1, paragraphs };
}

function outcomeMeasures(results: Json): TextSection {
  const paragraphs: string[] = [];
  for (const measure of list(obj(results.outcomeMeasuresModule).outcomeMeasures).map(obj)) {
    const titles = groupTitles(measure.groups);
    const unit = str(measure.unitOfMeasure);
    paragraphs.push(
      [
        `${readable(str(measure.type))} outcome: ${str(measure.title)}.`,
        str(measure.description),
        str(measure.populationDescription),
        str(measure.timeFrame) && `Time frame: ${str(measure.timeFrame)}.`,
      ]
        .filter(Boolean)
        .join(" "),
    );
    for (const denom of list(measure.denoms).map(obj)) {
      const counts = list(denom.counts)
        .map(obj)
        .map((each) => `${titles.get(str(each.groupId)) ?? str(each.groupId)} ${str(each.value)}`)
        .join("; ");
      if (counts) paragraphs.push(`Participants analysed (${str(denom.units)}): ${counts}.`);
    }
    for (const each of list(measure.classes).map(obj)) {
      for (const category of list(each.categories).map(obj)) {
        const values = list(category.measurements)
          .map(obj)
          .map((value) => {
            const range =
              str(value.lowerLimit) || str(value.upperLimit)
                ? ` (${str(value.lowerLimit)} to ${str(value.upperLimit)})`
                : "";
            return `${titles.get(str(value.groupId)) ?? str(value.groupId)} ${str(value.value)}${range}`;
          })
          .join("; ");
        const label = [str(each.title), str(category.title)].filter(Boolean).join(", ");
        if (values)
          paragraphs.push(`${label ? `${label}: ` : ""}${values}${unit ? ` ${unit}` : ""}.`);
      }
    }
    for (const analysis of list(measure.analyses).map(obj)) {
      const test = [
        str(analysis.statisticalMethod),
        str(analysis.paramType) && `${str(analysis.paramType)} ${str(analysis.paramValue)}`,
        str(analysis.pValue) && `p ${str(analysis.pValue)}`,
      ]
        .filter(Boolean)
        .join(", ");
      if (test) paragraphs.push(`Analysis: ${test}.`);
    }
  }
  return { heading: "Results: outcome measures", level: 1, paragraphs };
}

function adverseEvents(results: Json): TextSection {
  const module = obj(results.adverseEventsModule);
  const groups = list(module.eventGroups).map(obj);
  const titles = groupTitles(groups);
  const paragraphs = [...blocks(module.timeFrame), ...blocks(module.description)];
  for (const group of groups) {
    paragraphs.push(
      `${str(group.title)}: serious ${str(group.seriousNumAffected)} of ${str(group.seriousNumAtRisk)}, other ${str(group.otherNumAffected)} of ${str(group.otherNumAtRisk)}, deaths ${str(group.deathsNumAffected)} of ${str(group.deathsNumAtRisk)}.`,
    );
  }
  for (const [key, label] of [
    ["seriousEvents", "Serious"],
    ["otherEvents", "Other"],
  ] as const) {
    for (const event of list(module[key]).map(obj)) {
      // Only the groups that had the event: the rest would repeat zeros for every term.
      const counts = list(event.stats)
        .map(obj)
        .filter((stat) => Number(stat.numAffected) > 0)
        .map(
          (stat) =>
            `${titles.get(str(stat.groupId)) ?? str(stat.groupId)} ${str(stat.numAffected)} of ${str(stat.numAtRisk)}`,
        )
        .join("; ");
      if (!counts) continue;
      const organ = str(event.organSystem);
      paragraphs.push(
        `${label} adverse event: ${str(event.term)}${organ ? ` (${organ})` : ""}: ${counts}.`,
      );
    }
  }
  return { heading: "Results: adverse events", level: 1, paragraphs };
}

/** Reads one study record; null when it has no NCT number or title. */
export function readTrial(record: Json): TrialRecord | null {
  const protocol = obj(record.protocolSection);
  const id = obj(protocol.identificationModule);
  const nct = str(id.nctId);
  const title = str(id.briefTitle) || str(id.officialTitle);
  if (!nct || !title) return null;
  const results = obj(record.resultsSection);
  const posted = isRecord(record.resultsSection);
  const parts = [
    overview(protocol, nct),
    design(protocol),
    arms(protocol),
    outcomes(protocol, "primaryOutcomes", "Primary outcome"),
    outcomes(protocol, "secondaryOutcomes", "Secondary outcomes"),
    outcomes(protocol, "otherOutcomes", "Other outcomes"),
    eligibility(protocol),
    ...(posted && isRecord(results.participantFlowModule) ? [participantFlow(results)] : []),
    ...(posted && isRecord(results.outcomeMeasuresModule) ? [outcomeMeasures(results)] : []),
    ...(posted && isRecord(results.adverseEventsModule) ? [adverseEvents(results)] : []),
  ];
  return {
    nct,
    title,
    sponsor: str(obj(obj(protocol.sponsorCollaboratorsModule).leadSponsor).name),
    updated: dateOf(obj(protocol.statusModule).lastUpdatePostDateStruct) || null,
    sections: parts.filter((part) => part.paragraphs.length > 0),
  };
}
