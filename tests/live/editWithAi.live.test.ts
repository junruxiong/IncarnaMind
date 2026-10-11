/**
 * Edit with AI (#93) against real models: Improve writing and Translate on a
 * cited paragraph, in English and in Chinese, through the core as the app
 * uses it. Each model runs only when its key (or, for Ollama, its name) is in
 * the environment, and is skipped otherwise, so `npx vitest run` makes no
 * request without one. It checks that each reply comes back as Proposals,
 * never as an edit, and that the paragraph's Citation is still there, and
 * still found, once they are accepted.
 *
 *   ANTHROPIC_API_KEY        Claude (LIVE_ANTHROPIC_MODEL, Haiku by default)
 *   OPENAI_API_KEY           OpenAI (LIVE_OPENAI_MODEL, its quick-tasks model by default)
 *   LIVE_OLLAMA_MODEL        a model in Ollama on this computer, e.g. qwen3.5:4b, in local mode
 *
 * Set LIVE_REPORT to a file to keep what each model gave back.
 */
import { appendFile } from "node:fs/promises";
import { describe, expect, test } from "vitest";
import type * as Y from "yjs";
import {
  type ChatProviderKind,
  CITATION_NODE,
  type Core,
  type Document,
  type EditAction,
  PROPOSAL_ATTRIBUTE,
  PROPOSAL_DELETE_MARK,
  PROPOSAL_INSERT_MARK,
} from "../../src/core";
import { catalogProvider } from "../../src/core/providers/catalog/providers";
import { createAiSdkChatModel } from "../../src/core/providers/models";
import { pendingProposals, resolveProposals } from "../../src/shared/proposals";
import { createTempDataFolder, startCore } from "../helpers/core";
import { addAndProcess, writeSourceFile } from "../helpers/documents";
import { connectToMind, type MindClient } from "../helpers/mindClient";
import { readMind, writeMind } from "../helpers/minds";

const TIDES = {
  name: "Tides.md",
  contents: [
    "# Tides",
    "",
    "Spring tides happen at new moon and at full moon, when the Sun and the Moon pull together.",
  ].join("\n"),
};
const CHAOXI = {
  name: "潮汐.md",
  contents: "# 潮汐\n\n大潮出现在新月和满月前后，这时太阳和月亮的引力叠加在一起。",
};

interface Live {
  name: string;
  kind: ChatProviderKind;
  apiKey: string | undefined;
  model: string;
  local: boolean;
}

const LIVE: Live[] = [
  {
    name: "Anthropic",
    kind: "anthropic",
    apiKey: process.env.ANTHROPIC_API_KEY,
    model:
      process.env.LIVE_ANTHROPIC_MODEL ??
      (catalogProvider("anthropic")?.roles.quickTasks as string),
    local: false,
  },
  {
    name: "OpenAI",
    kind: "openai",
    apiKey: process.env.OPENAI_API_KEY,
    model:
      process.env.LIVE_OPENAI_MODEL ?? (catalogProvider("openai")?.roles.quickTasks as string),
    local: false,
  },
  {
    name: "Ollama",
    kind: "ollama",
    apiKey: process.env.LIVE_OLLAMA_MODEL ? "" : undefined,
    model: process.env.LIVE_OLLAMA_MODEL ?? "",
    local: true,
  },
];

/** The paragraph's text, Proposals shown as [-removed-] and {+added+}, a Citation as ⟨its check⟩. */
function shown(client: MindClient, blockId: string): string {
  let line = "";
  readMind(client).descendants((node) => {
    if (node.attrs.id !== blockId) return true;
    node.forEach((child) => {
      const kind = (child.attrs[PROPOSAL_ATTRIBUTE] as { kind?: string } | null)?.kind;
      const inline = child.isText ? (child.text ?? "") : `⟨${String(child.attrs.check)}⟩`;
      const marks = child.marks.map((mark) => mark.type.name);
      if (marks.includes(PROPOSAL_DELETE_MARK) || kind === "delete") line += `[-${inline}-]`;
      else if (marks.includes(PROPOSAL_INSERT_MARK) || kind === "insert") line += `{+${inline}+}`;
      else line += inline;
    });
    return false;
  });
  return line.replace(/-\]\[-|\+\}\{\+/g, "");
}

function citationsOf(client: MindClient, blockId: string): Record<string, unknown>[] {
  const found: Record<string, unknown>[] = [];
  readMind(client).descendants((node) => {
    if (node.attrs.id !== blockId) return true;
    node.descendants((child) => {
      if (child.type.name === CITATION_NODE) found.push(child.attrs);
    });
    return false;
  });
  return found;
}

const citationOf = (document: Document, quote: string) => ({
  type: CITATION_NODE,
  attrs: {
    passageId: "passage",
    documentId: document.id,
    documentName: document.name,
    contentHash: document.contentHash,
    quote,
    check: "found",
  },
});

async function edit(
  core: Core,
  client: MindClient,
  mindId: string,
  blockId: string,
  action: EditAction,
  instruction?: string,
) {
  let plain = "";
  readMind(client).descendants((node) => {
    if (node.attrs.id !== blockId) return true;
    plain = node.textBetween(0, node.content.size, "", "￼");
    return false;
  });
  const started = Date.now();
  const result = await core.editWithAi({
    requestId: crypto.randomUUID(),
    mindId,
    from: { blockId, offset: 0 },
    to: { blockId, offset: plain.length },
    text: plain,
    action,
    ...(instruction && { instruction }),
  });
  await client.settled();
  return { result, seconds: (Date.now() - started) / 1000 };
}

describe.each(LIVE)("Edit with AI, live: $name", (live) => {
  test.skipIf(live.apiKey === undefined)(
    `${live.name}: Improve writing and Translate on cited paragraphs come back as changes, and their Citations survive`,
    { timeout: 600_000 },
    async () => {
      const dataDir = await createTempDataFolder();
      const sources = await createTempDataFolder();
      const core = startCore(dataDir, { createChatModel: createAiSdkChatModel });
      core.on("consent.requested", (asked) => void core.respondToConsent(asked.requestId, true));
      await core.saveChatProvider({
        kind: live.kind,
        ...(live.apiKey && { apiKey: live.apiKey }),
        modelId: live.model,
      });
      if (live.local) await core.setLocalOnly(true);
      const [tides, chaoxi] = await addAndProcess(core, [
        await writeSourceFile(sources, TIDES.name, TIDES.contents),
        await writeSourceFile(sources, CHAOXI.name, CHAOXI.contents),
      ]);
      const mind = await core.createMind({ title: "Tides" });
      const client = await connectToMind(core, mind.id);
      const english = "english";
      const chinese = "chinese";
      const cases: { blockId: string; action: EditAction; instruction?: string }[] = [
        { blockId: english, action: "improve" },
        { blockId: english, action: "translate", instruction: "Simplified Chinese (简体中文)" },
        { blockId: chinese, action: "improve" },
        { blockId: chinese, action: "translate", instruction: "English" },
      ];
      const report: string[] = [`## ${live.name} · ${live.model}`];
      for (const each of cases) {
        writeMind(client, [
          {
            type: "paragraph",
            attrs: { id: english },
            content: [
              {
                type: "text",
                text: "the spring tides they happens at a new moon and also when the moon is full ",
              },
              citationOf(tides as Document, "Spring tides happen at new moon and at full moon"),
              { type: "text", text: ", because sun and moon is pulling together." },
            ],
          },
          {
            type: "paragraph",
            attrs: { id: chinese },
            content: [
              { type: "text", text: "大潮一般是在新月和满月的前后出现的的，" },
              citationOf(chaoxi as Document, "大潮出现在新月和满月前后"),
              { type: "text", text: "因为太阳和月亮的引力它们叠加在一起。" },
            ],
          },
        ]);
        await client.settled();
        const { result, seconds } = await edit(
          core,
          client,
          mind.id,
          each.blockId,
          each.action,
          each.instruction,
        );
        const proposed = shown(client, each.blockId);
        report.push(
          `- ${each.blockId} ${each.action}${each.instruction ? ` (${each.instruction})` : ""}: ${result.kind}${
            result.kind === "proposed"
              ? `, ${result.proposals} changes, ${result.droppedCitations} Citations dropped`
              : result.kind === "failed"
                ? `: ${result.error.kind} ${result.error.message}`
                : ""
          }, ${seconds.toFixed(1)} s\n  ${proposed}`,
        );
        expect(result.kind, JSON.stringify(result)).toBe("proposed");
        // Nothing is applied before it is accepted: the text as it was is all there.
        expect(pendingProposals(client.blocks).length).toBeGreaterThan(0);
        client.doc.transact(() => {
          resolveProposals(client.blocks as Y.XmlFragment, () => true, "accept");
        }, "editor");
        await client.settled();
        const kept = citationsOf(client, each.blockId);
        report.push(`  accepted: ${shown(client, each.blockId)}`);
        if (result.kind === "proposed" && result.droppedCitations === 0) {
          expect(kept).toHaveLength(1);
          expect(kept[0]?.check).toBe("found");
        }
      }
      if (process.env.LIVE_REPORT) {
        await appendFile(process.env.LIVE_REPORT, `${report.join("\n")}\n\n`);
      }
    },
  );
});
