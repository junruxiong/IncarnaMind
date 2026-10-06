# AI providers are configurable, and we don't use LangChain

The User picks one chat provider and one embedding provider. The options are OpenAI, Anthropic (chat only), Google, or any OpenAI-compatible server, which includes Ollama for fully local models and Chinese models such as DeepSeek and Qwen. Reranking is used only when a Cohere or Voyage key is configured. ChatGPT Plus and Pro users can also sign in with ChatGPT and use their plan for chat, if OpenAI accepts IncarnaMind into that program. This does not block a release.

The default embedding provider is a small multilingual model built into the app. It is downloaded on first use and runs on the User's CPU. Document search therefore works without any API key, and Document text never leaves the machine during indexing. Users who want better quality can switch to an API or Ollama.

This replaces the old backend, which needed Azure OpenAI, Google Vertex, Cohere, Voyage and a private Hugging Face endpoint all at once.

Tagging a Document automatically uses the chat model by default. If a TypeSafe Jev key is configured, Jev is used instead: it is faster and cheaper, and it returns calibrated probabilities. Its endpoint can be changed, so Jev-compatible models can be used too.

We don't use LangChain. The only parts the old code used were document loaders and a text splitter, which are small enough to own, and the user wants as few dependencies as possible.

## Considered options

**Using a Gemini subscription instead of an API key**: rejected. Google's terms forbid third-party apps from using Gemini subscription credentials, and accounts that do so can be suspended. Gemini is available with an API key only.

## Consequences

- Changing the embedding provider means re-processing every Document, because Passages embedded by different models can't be compared.
- Jev is a closed, hosted model in early access, so nothing may depend on it being configured.
