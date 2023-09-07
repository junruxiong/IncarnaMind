"""Conversational QA Chain"""
from __future__ import annotations
import os
import re
import configparser
import time
import logging

from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.memory import ConversationTokenBufferMemory
from convo_qa_chain import ConvoRetrievalChain

from toolkit.utils import (
    Config,
    choose_embeddings,
    load_embedding,
    load_pickle,
)
from toolkit.retrivers import MyRetriever


# Load the config file
configs = Config("configparser.ini")
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
config.read("configparser.ini")
os.environ["OPENAI_API_KEY"] = configs.openai_api_key
os.environ["ANTHROPIC_API_KEY"] = configs.anthropic_api_key

embedding = choose_embeddings(configs.embedding_name)
embedding_store_path = config.get("directory", "EMB_DIR")


# set models
models = {
    "llm_chat_gpt3": ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo", max_tokens=configs.max_llm_generation
    ),
    "llm_chat_gpt4": ChatOpenAI(
        temperature=0, model="gpt-4", max_tokens=configs.max_llm_generation
    ),
    "llm_chat_anthropic": ChatAnthropic(
        temperature=0,
        model="claude-2.0",
        max_tokens_to_sample=configs.max_llm_generation,
    ),
}


# load retrieval database
db_embedding_chunks_small = load_embedding(
    store_name=configs.embedding_name,
    embedding=embedding,
    suffix="chunks_small",
    path=embedding_store_path,
)
db_embedding_chunks_medium = load_embedding(
    store_name=configs.embedding_name,
    embedding=embedding,
    suffix="chunks_medium",
    path=embedding_store_path,
)

db_docs_chunks_small = load_pickle(
    prefix="docs_pickle", suffix="chunks_small", path=embedding_store_path
)
db_docs_chunks_medium = load_pickle(
    prefix="docs_pickle", suffix="chunks_medium", path=embedding_store_path
)
file_names = load_pickle(prefix="file", suffix="names", path=embedding_store_path)


# Initialize the retriever
my_retriever = MyRetriever(
    llm=models[configs.llm_model],
    embedding_chunks_small=db_embedding_chunks_small,
    embedding_chunks_medium=db_embedding_chunks_medium,
    docs_chunks_small=db_docs_chunks_small,
    docs_chunks_medium=db_docs_chunks_medium,
    first_retrieval_k=configs.first_retrieval_k,
    second_retrieval_k=configs.second_retrieval_k,
    num_windows=configs.num_windows,
    retriever_weights=configs.retriever_weights,
)


# Initialize the memory
memory = ConversationTokenBufferMemory(
    llm=models[configs.llm_model],
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True,
    max_token_limit=configs.max_chat_history,
)


# Initialize the QA chain
qa = ConvoRetrievalChain.from_llm(
    models[configs.llm_model],
    my_retriever,
    file_names=file_names,
    memory=memory,
    return_source_documents=False,
    return_generated_question=False,
)


if __name__ == "__main__":
    while True:
        user_input = input("Human: ")
        start_time = time.time()
        user_input_ = re.sub(r"^Human: ", "", user_input)
        print("*" * 6)
        resp = qa({"question": user_input_})
        print()
        print(f"AI:{resp['answer']}")
        print(f"Time used: {time.time() - start_time}")
        print("-" * 60)

        # async def async_generate(input_str: str):
        #     resp = await qa.arun(input_str)
        #     print()
        #     print(f"AI:{resp}")

        # async def main():
        #     while True:
        #         start_time = time.time()
        #         user_input = input("Human: ")
        #         print()
        #         user_input_ = re.sub(r"^Human: ", "", user_input)
        #         await asyncio.gather(*[async_generate(user_input_)])
        #         print(f"Time used: {time.time() - start_time}")
        #         print("-" * 50)

        # asyncio.run(main())
