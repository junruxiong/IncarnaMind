# AI providers are configurable, and we don't use LangChain

**Amended 2026-10-10 (#176): what the app knows about providers and models is one catalog in code** (`src/core/providers/catalog`), read instead of guessing from model names.
- **Providers are written by hand:** names in English and Chinese, endpoints per region, the AI SDK adapter, the "get a key" link, request quirks (OpenAI's `store: false`), a line on what the provider does with API data, and the models for Answers, quick tasks and reading images. A provider's Answers default is its cheaper capable model; its strongest is one click away.
- **Model facts are generated** from models.dev (MIT), cross-checked against LiteLLM's model list (MIT), by `node scripts/catalogModels.ts`, and corrected in a short hand-written overrides file. Their licences ship with the data. Prices are in the provider's own currency.
- **No runtime fetch.** The catalog changes only with releases. A provider's own model list, fetched after consent as before, brings new model ids on the day they launch; until the catalog knows them their capabilities are unknown, and they still work.
- **Precedence:** what the User set, then what the server reports (Ollama's model details, Anthropic's model list, a refusal an Answer meets), then the catalog, then unknown, which keeps the step-down of ADR-0007. Reading images, temperature, the way of citing, structured output and a cloud model's context window come from this; the name patterns that guessed them, the not-a-chat-model filter and the prefilled model ids are gone.
- **Stored providers** carry their catalog id and endpoint (migration 29); keys stay where they were.

The User picks one chat provider and one embedding provider. The options are OpenAI, Anthropic (chat only), Google, or any OpenAI-compatible server, which includes Ollama for fully local models and Chinese models such as DeepSeek and Qwen. Reranking is on by default, with a small multilingual reranking model built into the app: mmarco-mMiniLMv2-L12-H384, a 136 MB download once there are Documents to search, run on the CPU, sending nothing. In the retrieval evaluation (#31, 2026-10-09) it took hybrid search from 13 to 16 of 20 English Questions and from 19 to 20 of 20 Chinese ones, so the 80% bar holds for what Users get, for about 0.5 to 0.9 s a search. The User can rerank with a Cohere or Voyage key instead, or turn reranking off. ChatGPT Plus and Pro users can also sign in with ChatGPT and use their plan for chat, if OpenAI accepts IncarnaMind into that program. This does not block a release.

The default embedding provider is a small multilingual model built into the app. It is downloaded on first use and runs on the User's CPU. Document search therefore works without any API key, and Document text never leaves the machine during indexing. Users who want better quality can switch to an API or Ollama. Since 2026-10-10 embeddings are off by default, and search is keyword search, reranked; the built-in model is one of the choices the User turns on (ADR-0009).

This replaces the old backend, which needed Azure OpenAI, Google Vertex, Cohere, Voyage and a private Hugging Face endpoint all at once.

Tagging a Document automatically uses the chat model by default. If a TypeSafe Jev key is configured, Jev is used instead: it is faster and cheaper, and it returns calibrated probabilities. Its endpoint can be changed, so Jev-compatible models can be used too.

We don't use LangChain. The only parts the old code used were document loaders and a text splitter, which are small enough to own, and the user wants as few dependencies as possible.

## Considered options

**Using a Gemini subscription instead of an API key**: rejected. Google's terms forbid third-party apps from using Gemini subscription credentials, and accounts that do so can be suspended. Gemini is available with an API key only.

## Consequences

- Changing the embedding provider means re-processing every Document, because Passages embedded by different models can't be compared.
  - Each Document records the model its vectors come from (provider, server and model) and their size. Search compares a query only with vectors of the current model.
  - During a switch, search uses the new model at once and is marked as rebuilding. Keyword search covers every Document throughout; vector search covers the Documents already embedded again. We chose this over searching the old vectors until the switch completes. That would keep sending queries to the old provider after the User left it, which would defeat local mode. It would also need the old key and two sets of vectors.
  - A Document keeps its old vectors until the rebuild reaches it. Switching back before then costs nothing for that Document.
- Local mode ("keep everything on this computer") is a per-device setting. One-click Ollama also turns it on. It switches a cloud embedding provider back to the built-in model, which means re-processing every Document; an embedding provider on this computer, such as Ollama, is kept. It also pauses a reranking service; the built-in reranking model, the default, carries on. It doesn't block cloud chat or tagging. Settings tells the User when the chat model still sends Questions to a cloud provider.
- Jev is a closed, hosted model in early access, so nothing may depend on it being configured.
