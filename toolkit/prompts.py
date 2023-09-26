from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model

# ================================================================================

REFINE_QA_TEMPLATE = """Break down or rephrase the follow up input into fewer than heterogeneous one-hop queries to be the input of a retrieval tool, if the follow up inout is multi-hop, multi-step, complex or comparative queries and relevant to Chat History and Document Names. Otherwise keep the follow up input as it is.


The output format should strictly follow the following, and each query can only conatain 1 document name:
```
1. One-hop standalone query
...
3. One-hop standalone query
...
```


Document Names in the database:
```
{database}
```


Chat History:
```
{chat_history}
```


Begin:

Follow Up Input: {question}

One-hop standalone queries(s):
"""


# ================================================================================

DOCS_SELECTION_TEMPLATE = """Below are some verified sources and a human input. If you think any of them are relevant or contain any keywords related to the human input, then list all possible context numbers.

```
{snippets}
```

The output format must be like the following, nothing else. If not, you will output []:
[0, ..., n]

Human Input: {query}
"""


# ================================================================================

RETRIEVAL_QA_SYS = """You are a helpful assistant designed by IncarnaMind.
If you think the below below information are relevant to the human input, please respond to the human based on the relevant retrieved sources; otherwise, respond in your own words only about the human input."""


RETRIEVAL_QA_TEMPLATE = """
File Names in the database:
```
{database}
```


Chat History:
```
{chat_history}
```


Verified Sources:
```
{context}
```


User: {question}
"""


RETRIEVAL_QA_CHAT_TEMPLATE = """
File Names in the database:
```
{database}
```


Chat History:
```
{chat_history}
```


Verified Sources:
```
{context}
```
"""


class PromptTemplates:
    """_summary_"""

    def __init__(self):
        self.refine_qa_prompt = REFINE_QA_TEMPLATE
        self.docs_selection_prompt = DOCS_SELECTION_TEMPLATE
        self.retrieval_qa_sys = RETRIEVAL_QA_SYS
        self.retrieval_qa_prompt = RETRIEVAL_QA_TEMPLATE
        self.retrieval_qa_chat_prompt = RETRIEVAL_QA_CHAT_TEMPLATE

    def get_refine_qa_template(self, llm: str):
        """get the refine qa prompt template"""
        if "llama" in llm.lower():
            temp = f"[INST] {self.refine_qa_prompt} [/INST]"
        else:
            temp = self.refine_qa_prompt

        return PromptTemplate(
            input_variables=["database", "chat_history", "question"],
            template=temp,
        )

    def get_docs_selection_template(self, llm: str):
        """get the docs selection prompt template"""
        if "llama" in llm.lower():
            temp = f"[INST] {self.docs_selection_prompt} [/INST]"
        else:
            temp = self.docs_selection_prompt

        return PromptTemplate(
            input_variables=["snippets", "query"],
            template=temp,
        )

    def get_retrieval_qa_template_selector(self, llm: str):
        """get the retrieval qa prompt template"""
        if "llama" in llm.lower():
            temp = f"[INST] <<SYS>>\n{self.retrieval_qa_sys}\n<</SYS>>\n\n{self.retrieval_qa_prompt} [/INST]"
            messages = [
                SystemMessagePromptTemplate.from_template(
                    f"[INST] <<SYS>>\n{self.retrieval_qa_sys}\n<</SYS>>\n\n{self.retrieval_qa_chat_prompt} [/INST]"
                ),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        else:
            temp = f"{self.retrieval_qa_sys}\n{self.retrieval_qa_prompt}"
            messages = [
                SystemMessagePromptTemplate.from_template(
                    f"{self.retrieval_qa_sys}\n{self.retrieval_qa_chat_prompt}"
                ),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]

        prompt_temp = PromptTemplate(
            template=temp,
            input_variables=["database", "chat_history", "context", "question"],
        )
        prompt_temp_chat = ChatPromptTemplate.from_messages(messages)

        return ConditionalPromptSelector(
            default_prompt=prompt_temp,
            conditionals=[(is_chat_model, prompt_temp_chat)],
        )
