# AI providers are configurable, and we don't use LangChain

The User picks one chat provider and one embedding provider. The options are OpenAI, Anthropic (chat only), Google, or any OpenAI-compatible server, which includes Ollama for fully local models. Reranking is used only when a Cohere or Voyage key is configured. This replaces the old backend, which needed Azure OpenAI, Google Vertex, Cohere, Voyage and a private Hugging Face endpoint all at once.

We don't use LangChain. The only parts the old code used were document loaders and a text splitter, which are small enough to own, and the user wants as few dependencies as possible.

## Consequences

Changing the embedding provider means re-processing every Document, because Passages embedded by different models can't be compared.
