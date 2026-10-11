---
status: accepted (2026-10-11)
---

# A few providers and agents get first-class support; the rest go through generic routes

The catalog (ADR-0005) holds 13 chat providers plus the ChatGPT plan and Jev, and Settings → Agents (#351) lists 23 outside agents, about a dozen of which IncarnaMind sets up by writing into their own configuration files. Each route we promise carries upkeep that grows with every release. Model ids, prices and quirks change. Errors need plain words. A key guide has to be kept in English and Chinese, the vendor's terms have to be watched, and every AI feature is tried by hand on real providers before it is called done (`AGENTS.md`). An agent's configuration format can change under us at any time. At the same time, a User who already pays for a plan should be able to start in one click, and a User in mainland China needs a provider that works there.

We support two tiers:

- **First-class providers:** the ChatGPT plan (Sign in with ChatGPT, #122), Anthropic, OpenAI, Google, OpenRouter and Ollama (local models and Ollama's cloud models), and for mainland China DeepSeek, Qwen (Alibaba Cloud Model Studio) and SiliconFlow. They appear in the first-run model setup (#103), each has a plain-words key guide in English and Chinese where a key is needed, and each is tried by hand before a release.
- **Best-effort providers:** Mistral, xAI, Kimi and GLM stay in the catalog and keep working for Users who have their keys, listed under "More providers" in Settings → Models, with no guide of their own and no release check. Any other service is reached as an OpenAI-compatible endpoint (MiniMax, Volcengine, Tencent and the rest), or through OpenRouter and SiliconFlow, which reach most models with one account.
- **First-class agents:** the Claude desktop app (every plan, Free included), Claude Code, the ChatGPT desktop app and Codex (one configuration; the ChatGPT app reaches local servers in its Work and Codex modes), Cursor, and Antigravity (the route for Google AI Pro and Ultra subscribers, since Gemini CLI no longer serves those plans' personal accounts); in China, Tencent's WorkBuddy, and the 豆包 desktop app once it has been tried by hand and its data terms read. Each is set up in one click or by IncarnaMind and tried before a release. Chat apps that only reach servers on the internet (ChatGPT and Claude on the web, Gemini, Le Chat, Grok and others) can't reach IncarnaMind, which serves only this computer.
- **Other agents** connect as "Other agent": IncarnaMind shows a configuration to copy, and no longer writes into their files. Pairings made before this keep working.
- **Plans: only routes the vendor allows.** Sign in with ChatGPT, Ollama's and OpenRouter's own sign-in, the Claude desktop app through the agent connection (Anthropic doesn't let other apps sign in with a Claude plan), and the API credits that Claude Max and Team include, used as an ordinary key, with a short note that links to Anthropic's own pages instead of copying its steps. Google AI Pro and Ultra subscribers are pointed to Antigravity, where their plan pays, with only a link to Google's credit page, whose steps (three consoles, a card, prepaid billing) are too many to guide. No presets for coding or token plans, and never a borrowed login.
- **One experiment where the vendor's word is mixed:** the User's own Claude Code, unmodified and signed in through Anthropic's own flow, run by IncarnaMind on the User's Claude plan (#385). Anthropic's help page says the plan's limits can be used "with the Claude Agent SDK, claude -p, and third-party apps", while the Agent SDK's overview says third-party products need Anthropic's approval to offer them. It ships as an opt-in experiment while we ask Anthropic to confirm, and goes in the next release if Anthropic objects.

A provider or agent moves between tiers with a one-line change: a field in the catalog or the agent list. Promote one when Users ask for it; demote one that breaks often or loses its Users.

## Considered options

- **Every provider and agent first-class.** Rejected: the upkeep grows with each one, and most would be tested rarely, so a promise we keep for a few becomes one we break for many.
- **Only aggregators (OpenRouter, SiliconFlow) and the ChatGPT plan.** Rejected: Users with an Anthropic, OpenAI or Google key, often a company key, expect to paste it, and each provider's data terms differ from the aggregator's.
- **Removing the best-effort providers.** Rejected: a catalog row costs little to keep, and Users who set one up would lose it.
- **Presets for the coding and token plans of Chinese vendors.** Rejected: most limit use to coding tools or listed tools, and at the usage of a notebook pay-as-you-go costs less than any of them.

## Consequences

- The catalog gains a support field per provider, and the agent list one per agent. The first-run setup lists only first-class providers. Settings → Models lists first-class ones first and the rest under "More providers". A provider a User already set up shows under their providers whatever its tier.
- `docs/agents/testing.md` names the first-class providers and agents for release checks. The multi-provider rule for AI features stays as it is.
- Settings → Agents shows the first-class agents and "Other agent". The code that writes other agents' configuration files goes, except what Disconnect needs to remove an entry IncarnaMind wrote before.
- A best-effort provider can break for a while before anyone notices. Its errors still reach the User in plain words, and a report moves it up the queue.
