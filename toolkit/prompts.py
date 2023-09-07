from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# ================================================================================

refine_qa_template = """Given the Chat History and File Names.
If the Follow Up Input is relevant to Chat History and File Names or is complex (such as about comparison, multi-hop questions and multiple documents etc.,) then decompose into less than 3 standalone question(s) and MUST have clear mentions. 
Otherwise, keep the Follow Up Input as it is.

File Names:
```
{database}
```

Chat History:
```
{chat_history}
```

The number of standalone question(s) MUST be less than 3 and keep as MINIMAL as possbile!

Begin:

Follow Up Input: {question}
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["database", "chat_history", "question"],
    template=refine_qa_template,
)


# ================================================================================

docs_selection_prompt = """Below are some verified sources and a human Input. If you think they are relevant or contain any keywords to Human Input, please list all possible snippet numbers. 

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
If you think the verified source below is relevant to human input, please respond to the human based on the only relevant retrieved sources; otherwise, respond to it in your own words.

{context}

Human Input: {question}
Helpful Answer:"""

QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

system_template = """You are a helpful assistant designed by IncarnaMind.
If you think the verified source below is relevant to human input, please respond to the human based on the only relevant retrieved sources; otherwise, respond to it in your own words.

----------------
{context}"""


messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
QA_CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)
