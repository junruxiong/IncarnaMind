from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# ================================================================================

refine_qa_template = """Given the Chat History and Doc Names in the database, break up the Follow Up Input into less than 3 ONE-HOP query(s) for a retrieval engine input if it is a multi-hop or comparative query.
The output must be only query(s) and nothing else and keep the same format as the original Follow Up Input.

Aviliable Doc Names to decompose into standalone Query(s) in the database:
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

docs_selection_prompt = """Below are some verified sources and a human Input. If you think any of them are relevant or contain any keywords to Human Input, then list all possible Context numbers. 

```
{snippets}
```

The output format must be like the following, nothing else. If not, you will get [] instead:
[0, ..., n]
Human Input: {query}
"""

DOCS_SELECTION_PROMPT = PromptTemplate(
    input_variables=["snippets", "query"],
    template=docs_selection_prompt,
)

# ================================================================================


prompt_template = """You are a helpful assistant designed by IncarnaMind.
You have the access to name of files in the database.

File Names:
```
{database}
```

Chat History:
```
{chat_history}
```

If you think the below verified sources from the database are relevant to human input, please respond to the human based on the relevant retrieved sources; otherwise, respond to it in your own words.
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

If you think the below verified sources from the database are relevant to human input, please respond to the human based on the relevant retrieved sources; otherwise, respond to it in your own words.
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
