/**
 * A Mind's own Search scope in the assistant (#1359): its Folder while the
 * Folder holds a Document, else every Document, so a Mind filed into an empty
 * Folder still finds and cites what the User has. A scope the User chose stays
 * as chosen; with nothing in it, the model is told why it can't search.
 */
import { describe, expect, test } from "vitest";
import type { AskInConversationInput, ConversationAnswer, SearchScope } from "../../src/core/api";
import { answerEnded, citingModel, setUpWithDocuments } from "../helpers/citations";
import { promptOf } from "../helpers/models";

const DPA = {
  name: "DPA - Copperleaf.txt",
  contents:
    "Data Processing Agreement with Copperleaf.\nClause 3: Copperleaf processes personal data only on documented instructions.\n",
};
const LEASE = {
  name: "Office lease.txt",
  contents: "Office lease.\nThe tenant pays rent quarterly in advance.\n",
};
const CLAUSE = "Copperleaf processes personal data only on documented instructions.";

const citing = () =>
  citingModel({
    query: "Copperleaf clause 3",
    records: (passages) => [
      {
        marker: 1,
        passage: passages.find((each) => each.document === "DPA - Copperleaf")?.id,
        quote: CLAUSE,
      },
    ],
    answer: "Copperleaf processes personal data only on documented instructions [^1].",
  });

const folderScope = (folderId: string): SearchScope => ({
  folderIds: [folderId],
  tagIds: [],
  documentIds: [],
});

async function ask(
  core: Awaited<ReturnType<typeof setUpWithDocuments>>["core"],
  input: AskInConversationInput,
) {
  const result = await core.askInConversation(input);
  if (!result.asked) throw new Error(`The Question wasn't asked: ${JSON.stringify(result)}`);
  expect((await answerEnded(core, result.answerId)).event).toBe("finished");
  const { messages } = await core.getConversation(result.conversation.id);
  const answer = messages.find((message) => message.id === result.answerId);
  const question = messages.find((message) => message.id === result.questionId);
  if (answer?.kind !== "answer" || question?.kind !== "question") throw new Error("Not asked.");
  return { answer: answer as ConversationAnswer, question };
}

const instructions = (model: ReturnType<typeof citing>) =>
  promptOf(model, 0).find((message) => message.role === "system")?.text ?? "";

describe("A Mind's own Search scope in the assistant", { timeout: 30_000 }, () => {
  // protects=a Mind filed into a Folder with no Documents searches every Document, so a request for cited text finds the Document Not in a Folder and cites it, and the Question says it searched everything (#1359); fails_when=the Answer is written from nothing, uncited, or the Question keeps the empty Folder as its scope; no_claim=what a real model writes, which the pull request's hand check covers.
  test("is every Document while the Mind's Folder holds none", async () => {
    const model = citing();
    const { core } = await setUpWithDocuments(model, [DPA, LEASE]);
    const contracts = await core.createLibraryGroup({
      name: "Contracts & agreements",
      description: "",
    });
    const mind = await core.createMind({ title: "Vendor notes", folderId: contracts.id });

    const { answer, question } = await ask(core, {
      text: "Add a two-sentence summary of clause 3 of the DPA, with Citations.",
      mindId: mind.id,
      // As a composer that shows the Mind's Folder sends it.
      scope: folderScope(contracts.id),
    });

    expect(model.doStreamCalls[0]?.tools?.map((each) => each.name)).toContain("search_documents");
    expect(instructions(model)).not.toContain("holds no Documents");
    expect(answer.citations).toHaveLength(1);
    expect(answer.citations[0]?.documentName).toBe("DPA - Copperleaf");
    expect(question.scope).toBeNull();
  });

  // protects=the Mind's Folder, once it holds a Document, is what its Questions search again: only its Documents; fails_when=it searches everything, or Documents outside the Folder; no_claim=none beyond protects.
  test("is the Mind's Folder once it holds a Document", async () => {
    const model = citing();
    const { core, documents } = await setUpWithDocuments(model, [DPA, LEASE]);
    const contracts = await core.createLibraryGroup({
      name: "Contracts & agreements",
      description: "",
    });
    const mind = await core.createMind({ title: "Vendor notes", folderId: contracts.id });
    const dpa = documents.find((each) => each.name === "DPA - Copperleaf");
    if (!dpa) throw new Error("No DPA.");
    await core.assignDocumentGroup(dpa.id, contracts.id);

    const { answer, question } = await ask(core, {
      text: "What does clause 3 of the DPA say?",
      mindId: mind.id,
      scope: folderScope(contracts.id),
    });

    expect(instructions(model)).toContain("The User limited this Question to 1 Document");
    expect(answer.citations[0]?.documentName).toBe("DPA - Copperleaf");
    expect(question.scope).toEqual(folderScope(contracts.id));
  });

  // protects=a scope the User chose that holds nothing stays as chosen, and the model is told it has no search because the scope is empty, not because a Tool failed, and not to write what a Document says; fails_when=it widens by itself, or the model is left to blame its Tools; no_claim=how a real model words its reply.
  test("a Folder the User chose that holds nothing stays, and the model is told why it can't search", async () => {
    const model = citing();
    const { core, mind } = await setUpWithDocuments(model, [DPA]);
    const empty = await core.createLibraryGroup({ name: "Board papers", description: "" });

    const { answer, question } = await ask(core, {
      text: "Summarise clause 3 of the DPA, with Citations.",
      mindId: mind.id,
      scope: folderScope(empty.id),
    });

    expect(question.scope).toEqual(folderScope(empty.id));
    expect(model.doStreamCalls[0]?.tools?.map((each) => each.name) ?? []).not.toContain(
      "search_documents",
    );
    const told = instructions(model);
    expect(told).toContain("holds no Documents, so there is nothing to search");
    expect(told).toContain("never say a Tool is unavailable");
    expect(told).toContain("never state what a Document says");
    expect(answer.citations).toHaveLength(0);
  });
});
