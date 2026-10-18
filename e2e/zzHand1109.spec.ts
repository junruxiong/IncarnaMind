import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import { test } from "@playwright/test";
import type { CoreBridge } from "../src/core/api";
import { acceptPlainTextKeys, addDocuments, askInAssistant, createDataFolder, dismissChatSetup, enterContent, launchApp, newMind, setWindowSize } from "./app";

const CITIES = ["Lyon", "Lille", "Nantes", "Bordeaux", "Toulouse", "Nice", "Rennes"];
const DOCS = CITIES.map((city, index) => ({
  name: `Lease ${city}.md`,
  text: `# Lease ${city}\n\nThe office lease in ${city} runs for ${3 + index} years. The rent is ${1000 + index * 150} euros a month, paid quarterly. The tenant may leave after ${index + 1} years with six months' notice.\n`,
}));
const MODELS = [
  ["haiku", "anthropic", process.env.INCARNAMIND_EVAL_CHAT_KEY ?? "", "claude-haiku-5-5"],
  ["sol", "openai", process.env.OPENAI_API_KEY ?? "", "gpt-6.1-sol"],
] as const;
const RUNS = [
  ["write-zh", "zh-CN", "", "根据我现有的文件帮我写一份summary"],
  ["write-en", "en", "", "Based on my files, write a summary"],
  ["doc-zh", "zh-CN", "这些文件里的租约有什么不同？", "你不给我建一份文档吗"],
  ["doc-en", "en", "How do the leases in my files differ?", "Make it a document"],
] as const;
const only = process.env.HAND_ONLY ?? "";

for (const [label, kind, key, model] of MODELS) {
  for (const [run, language, before, ask] of RUNS) {
    const name = `${label}-${run}`;
    if (only && !only.split(",").includes(name)) continue;
    test(`hand 1109 ${name}`, async () => {
      test.setTimeout(600_000);
      const dataDir = await createDataFolder();
      const sources = await createDataFolder();
      for (const doc of DOCS) await writeFile(join(sources, doc.name), doc.text);
      const { app, window } = await launchApp(dataDir);
      await dismissChatSetup(window);
      await acceptPlainTextKeys(window);
      await window.evaluate(async ({ kind, key, model, language }) => {
        const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
        const provider = await core.saveChatProvider({ kind, apiKey: key, modelId: model } as Parameters<CoreBridge["saveChatProvider"]>[0]);
        if (provider.service) for (const flow of ["chat", "tagging"] as const) await core.allowDataFlow(flow, provider.service.id);
        await core.updateSettings({ user: { language } });
      }, { kind, key, model, language });
      await setWindowSize(app, window, 1440, 900);
      await addDocuments(window, DOCS.map((doc) => join(sources, doc.name)));
      await newMind(window);
      await window.keyboard.type(language === "en" ? "Lease notes" : "租约笔记");
      await window.keyboard.press("Enter");
      await window.keyboard.type(language === "en" ? "My notes on the office leases." : "关于办公室租约的笔记。");
      if (before) {
        await askInAssistant(window, before, { timeout: 400_000 });
        await window.waitForTimeout(2000);
      }
      await askInAssistant(window, ask, { timeout: 400_000 });
      await window.waitForTimeout(3000);
      const report = await window.evaluate(async () => {
        const core = (globalThis as unknown as { incarnamind: CoreBridge }).incarnamind;
        const [conversation] = await core.listConversations();
        if (!conversation) return "no conversation";
        const { messages } = await core.getConversation(conversation.id);
        const answers = messages.filter((m) => m.kind === "answer");
        const lines: string[] = [];
        for (const a of answers) {
          if (a.kind !== "answer") continue;
          lines.push(`ANSWER citations=${a.citations.length} text=${a.text.slice(0, 160).replace(/\n/g, " ")}`);
          for (const call of a.toolCalls) lines.push(`  CALL ${call.tool} ${JSON.stringify(call.input).slice(0, 120)} ${JSON.stringify(call.mindChange ?? null)}`);
        }
        return lines.join("\n");
      });
      console.log(`=== ${name}\n${report}`);
      await window.screenshot({ path: `${process.env.HAND_OUT ?? "/tmp"}/h1109-${name}.png` });
    });
  }
}
