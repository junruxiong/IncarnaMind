"""Conversational QA Chain"""
from __future__ import annotations
import os
import re
import time
import logging

from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.memory import ConversationTokenBufferMemory
from convo_qa_chain import ConvoRetrievalChain

from toolkit.together_api_llm import TogetherLLM
from toolkit.retrivers import MyRetriever
from toolkit.local_llm import load_local_llm
from toolkit.utils import (
    Config,
    choose_embeddings,
    load_embedding,
    load_pickle,
    check_device,
)


# Load the config file
configs = Config("configparser.ini")
logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = configs.openai_api_key
os.environ["ANTHROPIC_API_KEY"] = configs.anthropic_api_key

embedding = choose_embeddings(configs.embedding_name)
db_store_path = configs.db_dir


# get models
def get_llm(llm_name: str, temperature: float, max_tokens: int):
    """Get the LLM model from the model name."""

    if not os.path.exists(configs.local_model_dir):
        os.makedirs(configs.local_model_dir)

    splits = llm_name.split("|")  # [provider, model_name, model_file]

    if "openai" in splits[0].lower():
        llm_model = ChatOpenAI(
            model=splits[1],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif "anthropic" in splits[0].lower():
        llm_model = ChatAnthropic(
            model=splits[1],
            temperature=temperature,
            max_tokens_to_sample=max_tokens,
        )

    elif "together" in splits[0].lower():
        llm_model = TogetherLLM(
            model=splits[1],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif "huggingface" in splits[0].lower():
        llm_model = load_local_llm(
            model_id=splits[1],
            model_basename=splits[-1],
            temperature=temperature,
            max_tokens=max_tokens,
            device_type=check_device(),
        )
    else:
        raise ValueError("Invalid Model Name")

    return llm_model


llm = get_llm(configs.model_name, configs.temperature, configs.max_llm_generation)


# load retrieval database
db_embedding_chunks_small = load_embedding(
    store_name=configs.embedding_name,
    embedding=embedding,
    suffix="chunks_small",
    path=db_store_path,
)
db_embedding_chunks_medium = load_embedding(
    store_name=configs.embedding_name,
    embedding=embedding,
    suffix="chunks_medium",
    path=db_store_path,
)

db_docs_chunks_small = load_pickle(
    prefix="docs_pickle", suffix="chunks_small", path=db_store_path
)
db_docs_chunks_medium = load_pickle(
    prefix="docs_pickle", suffix="chunks_medium", path=db_store_path
)
file_names = load_pickle(prefix="file", suffix="names", path=db_store_path)


# Initialize the retriever
my_retriever = MyRetriever(
    llm=llm,
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
    llm=llm,
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True,
    max_token_limit=configs.max_chat_history,
)


# Initialize the QA chain
qa = ConvoRetrievalChain.from_llm(
    llm,
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
