import { describe, expect, onTestFinished, test } from "vitest";
import type { Core, CoreEvents } from "../../src/core";
import { citedReference, differences, titleCoverage } from "../../src/core/referenceChecks/match";
import { createTempDataFolder, startCore } from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { heading, note, writeMind } from "../helpers/minds";
import { unusedLocalUrl } from "../helpers/ollama";
import { type CrossrefWork, startReferenceServices } from "../helpers/referenceServices";

/*
 * Checking a reference list (#711): each entry looked up in Crossref, by DOI
 * or as written, and marked found, doesn't match, not found or not checked.
 * A local stand-in answers for Crossref: no test reaches the real one.
 */

const LSTM: CrossrefWork = {
  DOI: "10.1162/neco.1997.9.8.1735",
  title: ["Long Short-Term Memory"],
  author: [
    { family: "Hochreiter", given: "Sepp" },
    { family: "Schmidhuber", given: "Jürgen" },
  ],
  issued: { "date-parts": [[1997, 11, 1]] },
  "container-title": ["Neural Computation"],
  type: "journal-article",
};
const DEEP: CrossrefWork = {
  DOI: "10.1038/nature14539",
  title: ["Deep learning"],
  author: [
    { family: "LeCun", given: "Yann" },
    { family: "Bengio", given: "Yoshua" },
    { family: "Hinton", given: "Geoffrey" },
  ],
  issued: { "date-parts": [[2015, 5, 27]] },
  "container-title": ["Nature"],
  type: "journal-article",
};

const REAL =
  "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.";
const BY_DOI =
  "LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521, 436–444. doi:10.1038/nature14539";
const INVENTED =
  "Marlowe, T. (2019). Quantum gradient folding in recurrent attention lattices. Journal of Imagined Results, 4, 1–9.";
const WRONG_YEAR =
  "Hochreiter, S., & Schmidhuber, J. (2003). Long short-term memory. Neural Computation.";
const WRONG_AUTHOR = "Smith, A. (2015). Deep learning. Nature. doi:10.1038/nature14539";
const DEAD_DOI = "Doe, J. (2020). A study of nothing much at all. doi:10.9999/nowhere.42";

/** A core whose Crossref is the stand-in, with a Mind whose reference list holds `entries`. */
async function setUp(entries: readonly string[], { localOnly = false } = {}) {
  const services = await startReferenceServices({ works: [LSTM, DEEP] });
  onTestFinished(() => services.close());
  const core: Core = startCore(await createTempDataFolder(), {
    referenceLookup: { crossref: services.url, arxiv: services.url },
  });
  if (localOnly) await core.setLocalOnly(true);
  const mind = await core.createMind({ title: "Draft" });
  const client = await connectToMind(core, mind.id);
  writeMind(client, [
    note("Recurrent networks remember (Hochreiter & Schmidhuber, 1997)."),
    heading(2, "References"),
    ...entries.map((entry) => note(entry)),
  ]);
  await client.settled();
  return { core, services, mindId: mind.id };
}

describe("Checking a reference list (#711)", { timeout: 30_000 }, () => {
  // protects=each mark: found by search and by DOI, not found (invented, or a DOI leading nowhere), doesn't match (year, author); only the reference is sent; fails_when=a mark is wrong or more than the reference goes; no_claim=Crossref's real matching (the evaluation, eval/references).
  test("every entry gets its mark, and only the reference is sent", async () => {
    const entries = [REAL, BY_DOI, INVENTED, WRONG_YEAR, WRONG_AUTHOR, DEAD_DOI];
    const { core, services, mindId } = await setUp(entries);
    const progress: CoreEvents["referenceCheck.progress"][] = [];
    core.on("referenceCheck.progress", (event) => progress.push(event));

    const result = await core.checkReferences(mindId);

    if (result.kind !== "checked") throw new Error(`Not checked: ${result.kind}`);
    expect(result.references.map((each) => each.mark)).toEqual([
      "found",
      "found",
      "not-found",
      "mismatch",
      "mismatch",
      "not-found",
    ]);
    const [real, byDoi, , wrongYear, wrongAuthor] = result.references;
    expect(real?.record).toEqual({
      title: "Long Short-Term Memory",
      authors: ["Hochreiter, Sepp", "Schmidhuber, Jürgen"],
      year: 1997,
      doi: "10.1162/neco.1997.9.8.1735",
    });
    expect(byDoi?.record?.doi).toBe("10.1038/nature14539");
    expect(wrongYear?.differences).toEqual([{ field: "year", found: "1997", cited: "2003" }]);
    expect(wrongAuthor?.differences).toEqual([
      { field: "authors", found: "LeCun, Bengio, Hinton", cited: null },
    ]);
    // The paragraph before the list isn't a reference; each request carries one reference.
    for (const request of services.requests) {
      expect(request.bodyLength).toBe(0);
      expect(decodeURIComponent(request.path)).not.toContain("Recurrent networks remember");
    }
    expect(progress.at(-1)).toEqual({ mindId, done: 6, total: 6 });
  });

  // protects=starting the check is the consent (ADR-0022): no dialog, and Settings › Privacy lists it as chosen; fails_when=a dialog opens or nothing is listed; no_claim=the line by the action, which the e2e covers.
  test("starting it is the consent: nothing asks, and Privacy lists it", async () => {
    const { core, mindId } = await setUp([REAL]);
    const asked: string[] = [];
    core.on("consent.requested", (request) => asked.push(request.flow.id));

    await core.checkReferences(mindId);

    expect(asked).toEqual([]);
    expect(await core.listDataFlows()).toContainEqual(
      expect.objectContaining({
        flow: expect.objectContaining({ id: "reference-check", sends: ["references"] }),
        consent: "accepted",
        chosen: true,
      }),
    );
  });

  // protects=local mode sends nothing and says so for each entry; fails_when=a request is made; no_claim=none beyond protects.
  test("in local mode nothing is sent, and every entry says why it wasn't checked", async () => {
    const { core, services, mindId } = await setUp([REAL, INVENTED], { localOnly: true });

    const result = await core.checkReferences(mindId);

    expect(result).toMatchObject({
      kind: "checked",
      references: [
        { mark: "not-checked", reason: "local-mode" },
        { mark: "not-checked", reason: "local-mode" },
      ],
    });
    expect(services.requests).toEqual([]);
  });

  // protects=a failing or unreachable Crossref leaves entries not checked with the reason, never "not found"; fails_when=a failure reads as not found; no_claim=none beyond protects.
  test("when Crossref fails or can't be reached, entries are not checked, never not found", async () => {
    const failing = await setUp([REAL, INVENTED]);
    failing.services.fail(500);
    expect(await failing.core.checkReferences(failing.mindId)).toMatchObject({
      kind: "checked",
      references: [
        { mark: "not-checked", reason: "service-error" },
        { mark: "not-checked", reason: "service-error" },
      ],
    });

    const offline = startCore(await createTempDataFolder(), {
      referenceLookup: { crossref: await unusedLocalUrl(), arxiv: await unusedLocalUrl() },
    });
    const mind = await offline.createMind({ title: "Offline" });
    const client = await connectToMind(offline, mind.id);
    writeMind(client, [heading(2, "参考文献"), note(REAL)]);
    await client.settled();
    expect(await offline.checkReferences(mind.id)).toMatchObject({
      kind: "checked",
      references: [{ mark: "not-checked", reason: "unreachable" }],
    });
  });

  // protects=a Mind with no reference list says so and sends nothing; a stopped check gives "cancelled"; fails_when=either sends or hangs; no_claim=none beyond protects.
  test("no reference list sends nothing; Stop cancels a check under way", async () => {
    const services = await startReferenceServices({ works: [LSTM] });
    onTestFinished(() => services.close());
    const core = startCore(await createTempDataFolder(), {
      referenceLookup: { crossref: services.url, arxiv: services.url },
    });
    const mind = await core.createMind({ title: "No list" });
    const client = await connectToMind(core, mind.id);
    writeMind(client, [note(REAL)]);
    await client.settled();
    expect(await core.checkReferences(mind.id)).toEqual({ kind: "no-references" });
    expect(services.requests).toEqual([]);

    const busy = await setUp([REAL, INVENTED, WRONG_YEAR, DEAD_DOI]);
    const checking = busy.core.checkReferences(busy.mindId);
    await busy.core.stopReferenceCheck(busy.mindId);
    expect(await checking).toEqual({ kind: "cancelled" });
  });
});

describe("Reading a reference as written", () => {
  // protects=the DOI and year are read from the text, the DOI's own digits not taken for a year; fails_when=a DOI's digits read as the year, or trailing punctuation stays in the DOI; no_claim=none beyond protects.
  test("its DOI and year", () => {
    expect(citedReference(`${BY_DOI}.`)).toMatchObject({ doi: "10.1038/nature14539", year: 2015 });
    expect(
      citedReference("Wu, L. (2021). 深度学习综述. 计算机学报. https://doi.org/10.1234/2019.5"),
    ).toMatchObject({ doi: "10.1234/2019.5", year: 2021 });
  });

  // protects=a title matches in any case, with accents and in Chinese; another title doesn't; fails_when=either is wrong; no_claim=none beyond protects.
  test("a title's words, in any case, with accents and in Chinese", () => {
    expect(titleCoverage(REAL, "Long Short-Term Memory")).toBe(1);
    expect(titleCoverage("Müller, K. (2020). Über Lernen.", "Uber lernen")).toBe(1);
    expect(titleCoverage("王五 (2020). 深度学习综述. 计算机学报.", "深度学习综述")).toBe(1);
    expect(titleCoverage(INVENTED, "Long Short-Term Memory")).toBe(0);
    expect(
      differences(citedReference(REAL), {
        fields: { title: "Deep learning", authors: [{ family: "LeCun", given: "Yann" }] },
        years: [2015],
      }).map((each) => each.field),
    ).toEqual(["title", "year", "authors"]);
  });
});
