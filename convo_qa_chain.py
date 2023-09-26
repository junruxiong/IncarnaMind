"""Conversational QA Chain"""
from __future__ import annotations
import inspect
import logging
from typing import Any, Dict, List, Optional
from pydantic import Field

from langchain.schema import BasePromptTemplate, BaseRetriever, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)

from toolkit.utils import (
    Config,
    _get_chat_history,
    _get_standalone_questions_list,
)
from toolkit.retrivers import MyRetriever
from toolkit.prompts import PromptTemplates

configs = Config("configparser.ini")
logger = logging.getLogger(__name__)

prompt_templates = PromptTemplates()


class ConvoRetrievalChain(BaseConversationalRetrievalChain):
    """Chain for having a conversation based on retrieved documents.

    This chain takes in chat history (a list of messages) and new questions,
    and then returns an answer to that question.
    The algorithm for this chain consists of three parts:

    1. Use the chat history and the new question to create a "standalone question".
    This is done so that this question can be passed into the retrieval step to fetch
    relevant documents. If only the new question was passed in, then relevant context
    may be lacking. If the whole conversation was passed into retrieval, there may
    be unnecessary information there that would distract from retrieval.

    2. This new question is passed to the retriever and relevant documents are
    returned.

    3. The retrieved documents are passed to an LLM along with either the new question
    (default behavior) or the original question and chat history to generate a final
    response.

    Example:
        .. code-block:: python

            from langchain.chains import (
                StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
            )
            from langchain.prompts import PromptTemplate
            from langchain.llms import OpenAI

            combine_docs_chain = StuffDocumentsChain(...)
            vectorstore = ...
            retriever = vectorstore.as_retriever()

            # This controls how the standalone question is generated.
            # Should take `chat_history` and `question` as input variables.
            template = (
                "Combine the chat history and follow up question into "
                "a standalone question. Chat History: {chat_history}"
                "Follow up question: {question}"
            )
            prompt = PromptTemplate.from_template(template)
            llm = OpenAI()
            question_generator_chain = LLMChain(llm=llm, prompt=prompt)
            chain = ConversationalRetrievalChain(
                combine_docs_chain=combine_docs_chain,
                retriever=retriever,
                question_generator=question_generator_chain,
            )
    """

    retriever: MyRetriever = Field(exclude=True)
    """Retriever to use to fetch documents."""
    file_names: List = Field(exclude=True)
    """file_names (List): List of file names used for retrieval."""

    def _get_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        num_query: int,
        *,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> List[Document]:
        """Get docs."""
        try:
            docs = self.retriever.get_relevant_documents(
                question, num_query=num_query, run_manager=run_manager
            )
            return docs
        except (IOError, FileNotFoundError) as error:
            logger.error("An error occurred in _get_docs: %s", error)
            return []

    def _retrieve(
        self,
        question_list: List[str],
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> List[str]:
        num_query = len(question_list)
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )

        total_results = {}
        for question in question_list:
            docs_dict = (
                self._get_docs(
                    question, inputs, num_query=num_query, run_manager=run_manager
                )
                if accepts_run_manager
                else self._get_docs(question, inputs, num_query=num_query)
            )

            for file_name, docs in docs_dict.items():
                if file_name not in total_results:
                    total_results[file_name] = docs
                else:
                    total_results[file_name].extend(docs)

            logger.info(
                "-----step_done--------------------------------------------------",
            )

        snippets = ""
        redundancy = set()
        for file_name, docs in total_results.items():
            sorted_docs = sorted(docs, key=lambda x: x.metadata["medium_chunk_idx"])
            temp = "\n".join(
                doc.page_content
                for doc in sorted_docs
                if doc.metadata["page_content_md5"] not in redundancy
            )
            redundancy.update(doc.metadata["page_content_md5"] for doc in sorted_docs)
            snippets += f"\nContext about {file_name}:\n{{{temp}}}\n"

        return snippets, docs_dict

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])

        callbacks = _run_manager.get_child()
        new_questions = self.question_generator.run(
            question=question,
            chat_history=chat_history_str,
            database=self.file_names,
            callbacks=callbacks,
        )
        logger.info("new_questions: %s", new_questions)
        new_question_list = _get_standalone_questions_list(new_questions, question)[:3]
        # print("new_question_list:", new_question_list)
        logger.info("user_input: %s", question)
        logger.info("new_question_list: %s", new_question_list)

        snippets, source_docs = self._retrieve(
            new_question_list, inputs, run_manager=_run_manager
        )

        docs = [
            Document(
                page_content=snippets,
                metadata={},
            )
        ]

        new_inputs = inputs.copy()
        new_inputs["chat_history"] = chat_history_str
        answer = self.combine_docs_chain.run(
            input_documents=docs,
            database=self.file_names,
            callbacks=_run_manager.get_child(),
            **new_inputs,
        )
        output: Dict[str, Any] = {self.output_key: answer}
        if self.return_source_documents:
            output["source_documents"] = source_docs
        if self.return_generated_question:
            output["generated_question"] = new_questions

        logger.info("*****response*****: %s", output["answer"])
        logger.info(
            "=====epoch_done============================================================",
        )
        return output

    async def _aget_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        num_query: int,
        *,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> List[Document]:
        """Get docs."""
        try:
            docs = await self.retriever.aget_relevant_documents(
                question, num_query=num_query, run_manager=run_manager
            )
            return docs
        except (IOError, FileNotFoundError) as error:
            logger.error("An error occurred in _get_docs: %s", error)
            return []

    async def _aretrieve(
        self,
        question_list: List[str],
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        num_query = len(question_list)
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )

        total_results = {}
        for question in question_list:
            docs_dict = (
                await self._aget_docs(
                    question, inputs, num_query=num_query, run_manager=run_manager
                )
                if accepts_run_manager
                else await self._aget_docs(question, inputs, num_query=num_query)
            )

            for file_name, docs in docs_dict.items():
                if file_name not in total_results:
                    total_results[file_name] = docs
                else:
                    total_results[file_name].extend(docs)

            logger.info(
                "-----step_done--------------------------------------------------",
            )

        snippets = ""
        redundancy = set()
        for file_name, docs in total_results.items():
            sorted_docs = sorted(docs, key=lambda x: x.metadata["medium_chunk_idx"])
            temp = "\n".join(
                doc.page_content
                for doc in sorted_docs
                if doc.metadata["page_content_md5"] not in redundancy
            )
            redundancy.update(doc.metadata["page_content_md5"] for doc in sorted_docs)
            snippets += f"\nContext about {file_name}:\n{{{temp}}}\n"

        return snippets, docs_dict

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])

        callbacks = _run_manager.get_child()
        new_questions = await self.question_generator.arun(
            question=question,
            chat_history=chat_history_str,
            database=self.file_names,
            callbacks=callbacks,
        )
        new_question_list = _get_standalone_questions_list(new_questions, question)[:3]
        logger.info("new_questions: %s", new_questions)
        logger.info("new_question_list: %s", new_question_list)

        snippets, source_docs = await self._aretrieve(
            new_question_list, inputs, run_manager=_run_manager
        )

        docs = [
            Document(
                page_content=snippets,
                metadata={},
            )
        ]

        new_inputs = inputs.copy()
        new_inputs["chat_history"] = chat_history_str
        answer = await self.combine_docs_chain.arun(
            input_documents=docs,
            database=self.file_names,
            callbacks=_run_manager.get_child(),
            **new_inputs,
        )
        output: Dict[str, Any] = {self.output_key: answer}
        if self.return_source_documents:
            output["source_documents"] = source_docs
        if self.return_generated_question:
            output["generated_question"] = new_questions

        logger.info("*****response*****: %s", output["answer"])
        logger.info(
            "=====epoch_done============================================================",
        )

        return output

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        condense_question_prompt: BasePromptTemplate = prompt_templates.get_refine_qa_template(
            configs.model_name
        ),
        chain_type: str = "stuff",  # only support stuff chain now
        verbose: bool = False,
        condense_question_llm: Optional[BaseLanguageModel] = None,
        combine_docs_chain_kwargs: Optional[Dict] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseConversationalRetrievalChain:
        """Convenience method to load chain from LLM and retriever.

        This provides some logic to create the `question_generator` chain
        as well as the combine_docs_chain.

        Args:
            llm: The default language model to use at every part of this chain
                (eg in both the question generation and the answering)
            retriever: The retriever to use to fetch relevant documents from.
            condense_question_prompt: The prompt to use to condense the chat history
                and new question into standalone question(s).
            chain_type: The chain type to use to create the combine_docs_chain, will
                be sent to `load_qa_chain`.
            verbose: Verbosity flag for logging to stdout.
            condense_question_llm: The language model to use for condensing the chat
                history and new question into standalone question(s). If none is
                provided, will default to `llm`.
            combine_docs_chain_kwargs: Parameters to pass as kwargs to `load_qa_chain`
                when constructing the combine_docs_chain.
            callbacks: Callbacks to pass to all subchains.
            **kwargs: Additional parameters to pass when initializing
                ConversationalRetrievalChain
        """
        combine_docs_chain_kwargs = combine_docs_chain_kwargs or {
            "prompt": prompt_templates.get_retrieval_qa_template_selector(
                configs.model_name
            ).get_prompt(llm)
        }
        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            verbose=verbose,
            callbacks=callbacks,
            **combine_docs_chain_kwargs,
        )

        _llm = condense_question_llm or llm
        condense_question_chain = LLMChain(
            llm=_llm,
            prompt=condense_question_prompt,
            verbose=verbose,
            callbacks=callbacks,
        )
        return cls(
            retriever=retriever,
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
            callbacks=callbacks,
            **kwargs,
        )
