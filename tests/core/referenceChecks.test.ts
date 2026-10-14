import { describe, expect, onTestFinished, test } from "vitest";
import type { Core, CoreEvents } from "../../src/core";
import {
  citedReference,
  differences,
  holdsTitle,
  titleCoverage,
  titleOf,
} from "../../src/core/referenceChecks/match";
import { checkableReferences } from "../../src/shared/referenceList";
import { createTempDataFolder, startCore } from "../helpers/core";
import { connectToMind } from "../helpers/mindClient";
import { heading, note, writeMind } from "../helpers/minds";
import { unusedLocalUrl } from "../helpers/ollama";
import {
  type CrossrefWork,
  type OpenAlexWork,
  recordedAnswer,
  startReferenceServices,
} from "../helpers/referenceServices";

/*
 * Checking a reference list (#711): each entry looked up in Crossref, by DOI
 * or as written, then in OpenAlex, and an arXiv number on arXiv, and marked
 * found, doesn't match, not found, not listed or not checked. A local
 * stand-in answers for all three: no test reaches the real ones.
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
const PREPRINT =
  "Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.";
const INVENTED_PREPRINT =
  "Ray, S. (2023). Recursive gradient lattices for sparse memory. arXiv preprint arXiv:2399.00001.";
const WEB_PAGE = "OpenAI. (2022). Introducing ChatGPT. https://openai.com/blog/chatgpt";
const CONFERENCE =
  "Brown, T., Mann, B., et al. (2020). Language models are few-shot learners. Advances in Neural Information Processing Systems, 33.";
const ON_ARXIV =
  "Vaswani, A., et al. (2017). Attention is all you need. arXiv preprint arXiv:1706.03762.";
const DATASET =
  "Doe, A. (2021). Measurements of widgets in coastal towns. Zenodo. doi:10.5281/zenodo.123456";

/** OpenAlex's records: a conference paper Crossref lacks, and an arXiv preprint by its DOI. */
const OPENALEX_WORKS: OpenAlexWork[] = [
  {
    id: "https://openalex.org/W3030163527",
    doi: "https://doi.org/10.48550/arxiv.2005.14165",
    display_name: "Language Models are Few-Shot Learners",
    publication_date: "2020-05-28",
    authorships: [{ author: { display_name: "Tom B. Brown" } }],
  },
  {
    id: "https://openalex.org/W1522301498",
    doi: "https://doi.org/10.48550/arxiv.1412.6980",
    display_name: "Adam: A Method for Stochastic Optimization",
    publication_date: "2014-12-22",
    authorships: [{ author: { display_name: "Diederik P. Kingma" } }],
  },
];

/** A core whose Crossref is the stand-in, with a Mind whose reference list holds `entries`. */
async function setUp(
  entries: readonly string[],
  { localOnly = false, works = [LSTM, DEEP] }: { localOnly?: boolean; works?: CrossrefWork[] } = {},
) {
  const services = await startReferenceServices({
    works,
    registered: {
      "10.48550/arXiv.1412.6980": "datacite",
      "10.48550/arXiv.1706.03762": "datacite",
      "10.5281/zenodo.123456": "datacite",
    },
    openalex: OPENALEX_WORKS,
    arxiv: { "1706.03762": recordedAnswer("arxiv-attention.xml") },
  });
  onTestFinished(() => services.close());
  const core: Core = startCore(await createTempDataFolder(), {
    referenceLookup: { crossref: services.url, arxiv: services.url, openalex: services.url },
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
  // protects=each mark and the service that gave it: found in Crossref by search and by DOI, in OpenAlex by title (a conference paper) and by DOI (an arXiv preprint), on arXiv; not listed (in neither Crossref nor OpenAlex); not found (a DOI or arXiv number leading nowhere); doesn't match (year, author); not checked when another agency holds a DOI neither has; a web page isn't looked up; only the reference is sent; checking again asks nothing twice; fails_when=a mark is wrong, a search miss reads as an invented reference, a preprint reads as not found, more than the reference goes, or a reference is asked about again; no_claim=Crossref's real matching (the evaluation, eval/references).
  test("every entry gets its mark, and only the reference is sent", async () => {
    const entries = [
      REAL,
      BY_DOI,
      INVENTED,
      WRONG_YEAR,
      WRONG_AUTHOR,
      DEAD_DOI,
      WEB_PAGE,
      PREPRINT,
      INVENTED_PREPRINT,
      CONFERENCE,
      ON_ARXIV,
      DATASET,
    ];
    const { core, services, mindId } = await setUp(entries);
    const progress: CoreEvents["referenceCheck.progress"][] = [];
    core.on("referenceCheck.progress", (event) => progress.push(event));

    const result = await core.checkReferences(mindId);

    if (result.kind !== "checked") throw new Error(`Not checked: ${result.kind}`);
    expect(result.references.map(({ mark, service, reason }) => [mark, service, reason])).toEqual([
      ["found", "crossref", null],
      ["found", "crossref", null],
      ["not-listed", null, null],
      ["mismatch", "crossref", null],
      ["mismatch", "crossref", null],
      ["not-found", "crossref", null],
      ["found", "openalex", null],
      ["not-found", "crossref", null],
      ["found", "openalex", null],
      ["found", "arxiv", null],
      ["not-checked", null, "registered-elsewhere"],
    ]);
    const [real, byDoi, , wrongYear, wrongAuthor] = result.references;
    expect(real?.record).toEqual({
      title: "Long Short-Term Memory",
      authors: [
        { family: "Hochreiter", given: "Sepp" },
        { family: "Schmidhuber", given: "Jürgen" },
      ],
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
    expect(services.requests.some(({ path }) => path.toLowerCase().includes("openai"))).toBe(false);
    expect(progress.at(-1)).toEqual({ mindId, done: 11, total: 11 });

    // Checked again, nothing is asked twice.
    const asked = services.requests.length;
    expect(await core.checkReferences(mindId)).toEqual(result);
    expect(services.requests).toHaveLength(asked);
  });

  // protects=a work with several records (a later reprint first) is found by the one it matches; fails_when=the reprint's year reads as a mismatch; no_claim=none beyond protects.
  test("of a work's several records, the one it matches is taken", async () => {
    const reprint: CrossrefWork = {
      ...LSTM,
      DOI: "10.5555/reprint.2020",
      issued: { "date-parts": [[2020]] },
    };
    const { core, mindId } = await setUp([REAL], { works: [reprint, LSTM] });

    expect(await core.checkReferences(mindId)).toMatchObject({
      kind: "checked",
      references: [{ mark: "found", record: { doi: "10.1162/neco.1997.9.8.1735", year: 1997 } }],
    });
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

  // protects=a failing or unreachable Crossref leaves entries not checked with the reason, never "not found"; offline, one request fails and the rest aren't sent; fails_when=a failure reads as not found, or every entry is tried offline; no_claim=none beyond protects.
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

    const dropped = await setUp([REAL, INVENTED, WRONG_YEAR, DEAD_DOI]);
    dropped.services.fail("unreachable");
    expect(await dropped.core.checkReferences(dropped.mindId)).toMatchObject({
      kind: "checked",
      references: Array.from({ length: 4 }, () => ({ mark: "not-checked", reason: "unreachable" })),
    });
    expect(dropped.services.requests).toHaveLength(1);
    // Back online, Try again checks them all.
    dropped.services.fail(null);
    expect(await dropped.core.checkReferences(dropped.mindId)).toMatchObject({
      kind: "checked",
      references: [
        { mark: "found" },
        { mark: "not-listed" },
        { mark: "mismatch" },
        { mark: "not-found" },
      ],
    });

    const offline = startCore(await createTempDataFolder(), {
      referenceLookup: {
        crossref: await unusedLocalUrl(),
        arxiv: await unusedLocalUrl(),
        openalex: await unusedLocalUrl(),
      },
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

  // protects=OpenAlex offline: Crossref's answers stand, the first miss isn't checked with OpenAlex named, and OpenAlex isn't asked again; fails_when=a miss reads as not listed when OpenAlex wasn't asked, or each miss tries OpenAlex; no_claim=none beyond protects.
  test("when OpenAlex can't be reached, Crossref's answers stand and OpenAlex is asked once", async () => {
    const { core, services, mindId } = await setUp([REAL, INVENTED, CONFERENCE]);
    services.fail("unreachable", "openalex");

    expect(await core.checkReferences(mindId)).toMatchObject({
      kind: "checked",
      references: [
        { mark: "found", service: "crossref" },
        { mark: "not-checked", service: "openalex", reason: "unreachable" },
        { mark: "not-checked", service: "openalex", reason: "unreachable" },
      ],
    });
    expect(services.requests.filter(({ path }) => path.includes("search="))).toHaveLength(1);
  });

  // protects=a Mind with no reference list says so and sends nothing; Stop gives "cancelled" and nothing more is sent; fails_when=either sends, hangs, or a stopped check goes on asking; no_claim=none beyond protects.
  test("no reference list sends nothing; Stop cancels a check under way", async () => {
    const services = await startReferenceServices({ works: [LSTM] });
    onTestFinished(() => services.close());
    const core = startCore(await createTempDataFolder(), {
      referenceLookup: { crossref: services.url, arxiv: services.url, openalex: services.url },
    });
    const mind = await core.createMind({ title: "No list" });
    const client = await connectToMind(core, mind.id);
    writeMind(client, [note(REAL)]);
    await client.settled();
    expect(await core.checkReferences(mind.id)).toEqual({ kind: "no-references" });
    expect(services.requests).toEqual([]);

    const busy = await setUp([REAL, INVENTED, WRONG_YEAR, DEAD_DOI]);
    const slow = busy.services.pause();
    const checking = busy.core.checkReferences(busy.mindId);
    await slow.reached;
    await busy.core.stopReferenceCheck(busy.mindId);
    expect(await checking).toEqual({ kind: "cancelled" });
    slow.release();
    expect(busy.services.requests).toHaveLength(1);
  });
});

describe("Reading a reference as written", () => {
  // protects=the DOI, arXiv number and year are read from the text, and neither identifier's digits taken for a year; fails_when=a DOI's or arXiv number's digits read as the year, or trailing punctuation stays in the DOI; no_claim=none beyond protects.
  test("its DOI and year", () => {
    expect(citedReference(`${BY_DOI}.`)).toMatchObject({ doi: "10.1038/nature14539", year: 2015 });
    expect(
      citedReference("Vaswani, A. Attention is all you need. arXiv:1706.03762v5 (2017)."),
    ).toMatchObject({ doi: null, arxiv: "1706.03762", year: 2017 });
    expect(
      citedReference("Wu, L. (2021). 深度学习综述. 计算机学报. https://doi.org/10.1234/2019.5"),
    ).toMatchObject({ doi: "10.1234/2019.5", year: 2021 });
  });

  // protects=the title is read as each common style sets it apart (APA, numbered with quotes, GB/T 7714, Vancouver), for OpenAlex's search by title; fails_when=an author list or venue is taken for the title; no_claim=styles beyond these.
  test("its title, in the common styles", () => {
    expect(titleOf(CONFERENCE)).toBe("Language models are few-shot learners");
    expect(
      titleOf('[3] A. Vaswani et al., "Attention is all you need," in Proc. NeurIPS, 2017.'),
    ).toBe("Attention is all you need");
    expect(titleOf("周志华. 机器学习[M]. 北京: 清华大学出版社, 2016.")).toBe("机器学习");
    expect(titleOf("Smith J, Doe K. Widgets in coastal towns. J Widgets. 2020;12:3-4.")).toBe(
      "Widgets in coastal towns",
    );
  });

  // protects=a list ends at the first paragraph that isn't a reference, so an appendix after it isn't looked up; fails_when=the appendix's text is taken for a reference; no_claim=none beyond protects.
  test("its list ends where the references do", () => {
    const blocks = [
      { id: "a", heading: false, text: "Recurrent networks remember (1997)." },
      { id: "b", heading: true, text: "参考文献" },
      { id: "c", heading: false, text: REAL },
      { id: "d", heading: false, text: `[2] ${BY_DOI}` },
      {
        id: "e",
        heading: false,
        text: "The survey asked forty people about their reading habits.",
      },
      { id: "f", heading: false, text: INVENTED },
    ];
    expect(checkableReferences(blocks).map((entry) => entry.blockId)).toEqual(["c", "d"]);
  });

  // protects=a short title must stand alone in the reference, so a record titled "Introduction" isn't a longer title's work; a short real title still is; fails_when=either is wrong; no_claim=none beyond protects.
  test("a short title stands alone", () => {
    const longer = "Smith, J. (2020). Introduction to quantum widgets. Widget Letters, 3.";
    expect(holdsTitle(longer, "Introduction")).toBe(false);
    expect(holdsTitle(longer, "Introduction to Quantum Widgets")).toBe(true);
    expect(holdsTitle(BY_DOI, "Deep learning")).toBe(true);
    expect(holdsTitle(REAL, "Deep learning")).toBe(false);
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
