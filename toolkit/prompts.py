from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# ================================================================================

refine_qa_template = """Given the Chat History and Doc Names, break up the Follow Up Input into less than 3 ONE-HOP query(s) for a retrieval engine input if it is a multi-hop or comparative query.


Aviliable Doc Names to decompose into standalone Query(s):
```
{database}
```

Chat History:
```
{chat_history}

```

Begin:

Follow Up Input: {question}

Standalone Query(s) :
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
If you think the below verified sources are relevant to human input, please respond to the human based on the relevant retrieved sources; otherwise, respond to it in your own words.

{context}

Human Input: {question}
Helpful Answer:"""

QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

system_template = """You are a helpful assistant designed by IncarnaMind.
If you think the below verified sources are relevant to human input, please respond to the human based on the relevant retrieved sources; otherwise, respond to it in your own words.

----------------
{context}"""


messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
QA_CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)
