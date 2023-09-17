from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# ================================================================================

refine_qa_template = """Given the chat history and document names in the database, break down the follow up input into fewer than 3 heterogeneous ONE-HOP queries for retrieval engine input, if it is a multi-hop or comparative query. The output must be only queries and should keep the same stylpe as the original follow up input.

Available Document Names to decompose into standalone queries in the database:
```
{database}
```

Chat History:
```
{chat_history}

```

Begin:

Follow Up Input: {question}

Standalone Query(s):
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["database", "chat_history", "question"],
    template=refine_qa_template,
)


# ================================================================================

docs_selection_prompt = """Below are some verified sources and a human input. If you think any of them are relevant or contain any keywords related to the human input, then list all possible context numbers.

```
{snippets}
```

The output format must be like the following, nothing else. If not, you will output []:
[0, ..., n]
Human Input: {query}
"""

DOCS_SELECTION_PROMPT = PromptTemplate(
    input_variables=["snippets", "query"],
    template=docs_selection_prompt,
)

# ================================================================================


prompt_template = """You are a helpful assistant designed by IncarnaMind.
You have access to the names of files in the database.

File Names:
```
{database}
```

Chat History:
```
{chat_history}
```

If you think the verified sources from the database below are relevant to the human input, please respond to the human based on the relevant retrieved sources; otherwise, respond in your own words.
----------------

Contexts:
```
{context}
```

Human Input: {question}
Helpful Answer:"""

QA_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["database", "chat_history", "context", "question"],
)

system_template = """You are a helpful assistant designed by IncarnaMind.
You have the access to name of files in the database.

File Names:
```
{database}
```

Chat History:
```
{chat_history}
```

If you think the verified sources from the database below are relevant to the human input, please respond to the human based on the relevant retrieved sources; otherwise, respond in your own words.
----------------

Contexts:
```
{context}
```
"""


messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
QA_CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)
