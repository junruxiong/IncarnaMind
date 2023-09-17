"""
The widgets defines utility functions for loading data, text cleaning,
and indexing documents, as well as classes for handling document queries
and formatting chat history.
"""
import re
import pickle
import string
import logging
import configparser
from enum import Enum
from typing import List, Tuple, Union
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
import tiktoken
from langchain.vectorstores import Chroma

from langchain.schema import Document, BaseMessage
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings


tokenizer_name = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokenizer = tiktoken.get_encoding(tokenizer_name.name)

# if nltk stopwords not downloaded, download it
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

ChatTurnType = Union[Tuple[str, str], BaseMessage]
_ROLE_MAP = {"human": "Human: ", "ai": "Assistant: "}


class Config:
    """Initializes configs."""

    def __init__(self, config_file):
        self.config = configparser.ConfigParser(interpolation=None)
        self.config.read(config_file)

        # Tokens
        self.openai_api_key = self.config.get("tokens", "OPENAI_API_KEY")
        self.anthropic_api_key = self.config.get("tokens", "ANTHROPIC_API_KEY")
        self.version = self.config.get("tokens", "VERSION")

        # Directory
        self.docs_dir = self.config.get("directory", "DOCS_DIR")
        self.emb_dir = self.config.get("directory", "EMB_DIR")

        # Parameters
        self.llm_model = self.config.get("parameters", "LLM_MODEL")
        self.max_chat_history = self.config.getint("parameters", "MAX_CHAT_HISTORY")
        self.max_llm_context = self.config.getint("parameters", "MAX_LLM_CONTEXT")
        self.max_llm_generation = self.config.getint("parameters", "MAX_LLM_GENERATION")
        self.embedding_name = self.config.get("parameters", "EMBEDDING_NAME")

        self.base_chunk_size = self.config.getint("parameters", "BASE_CHUNK_SIZE")
        self.chunk_overlap = self.config.getint("parameters", "CHUNK_OVERLAP")
        self.chunk_scale = self.config.getint("parameters", "CHUNK_SCALE")
        self.window_steps = self.config.getint("parameters", "WINDOW_STEPS")
        self.window_scale = self.config.getint("parameters", "WINDOW_SCALE")

        self.retriever_weights = [
            float(x.strip())
            for x in self.config.get("parameters", "RETRIEVER_WEIGHTS").split(",")
        ]
        self.first_retrieval_k = self.config.getint("parameters", "FIRST_RETRIEVAL_K")
        self.second_retrieval_k = self.config.getint("parameters", "SECOND_RETRIEVAL_K")
        self.num_windows = self.config.getint("parameters", "NUM_WINDOWS")

        # Logging
        self.logging_enabled = self.config.getboolean("logging", "enabled")
        self.logging_level = self.config.get("logging", "level")
        self.logging_filename = self.config.get("logging", "filename")
        self.logging_format = self.config.get("logging", "format")

        self.configure_logging()

    def configure_logging(self):
        """
        Configure the logger for each .py files.
        """

        if not self.logging_enabled:
            logging.disable(logging.CRITICAL + 1)
            return

        log_level = self.config.get("logging", "level")
        log_filename = self.config.get("logging", "filename")
        log_format = self.config.get("logging", "format")

        logging.basicConfig(level=log_level, filename=log_filename, format=log_format)


def configure_logger():
    """
    Configure the logger for each .py files.
    """
    config = configparser.ConfigParser(interpolation=None)
    config.read("configparser.ini")

    enabled = config.getboolean("logging", "enabled")

    if not enabled:
        logging.disable(logging.CRITICAL + 1)
        return

    log_level = config.get("logging", "level")
    log_filename = config.get("logging", "filename")
    log_format = config.get("logging", "format")

    logging.basicConfig(level=log_level, filename=log_filename, format=log_format)


def tiktoken_len(text):
    """token length function"""
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def check_device():
    """Check if cuda or MPS is available, else fallback to CPU"""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def choose_embeddings(embedding_name):
    """Choose embeddings for a given model's name"""
    try:
        if embedding_name == "openAIEmbeddings":
            return OpenAIEmbeddings()
        elif embedding_name == "hkunlpInstructorLarge":
            device = check_device()
            return HuggingFaceInstructEmbeddings(
                model_name="hkunlp/instructor-large", model_kwargs={"device": device}
            )
        else:
            device = check_device()
            return HuggingFaceEmbeddings(model_name=embedding_name, device=device)
    except Exception as error:
        raise ValueError(f"Embedding {embedding_name} not supported") from error


def load_embedding(store_name, embedding, suffix, path):
    """Load chroma embeddings"""
    vector_store = Chroma(
        persist_directory=f"{path}/chroma_{store_name}_{suffix}",
        embedding_function=embedding,
    )
    return vector_store


def load_pickle(prefix, suffix, path):
    """Load langchain documents from a pickle file.

    Args:
        store_name (str): The name of the store where data is saved.
        suffix (str): Suffix to append to the store name.
        path (str): The path where the pickle file is stored.

    Returns:
        Document: documents from the pickle file
    """
    with open(f"{path}/{prefix}_{suffix}.pkl", "rb") as file:
        return pickle.load(file)


def clean_text(text):
    """
    Converts text to lowercase, removes punctuation, stopwords, and lemmatizes it
    for BM25 retriever.

    Parameters:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned and lemmatized text.
    """
    # remove [SEP] in the text
    text = text.replace("[SEP]", "")
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing
    tokens = [w.lower() for w in tokens]
    # Remove punctuation
    table = str.maketrans("", "", string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # Keep tokens that are alphabetic, numeric, or contain both.
    words = [
        word
        for word in stripped
        if word.isalpha()
        or word.isdigit()
        or (re.search("\d", word) and re.search("[a-zA-Z]", word))
    ]
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    # Lemmatization (or you could use stemming instead)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    # Convert list of words to a string
    lemmatized_ = " ".join(lemmatized)

    return lemmatized_


class IndexerOperator(Enum):
    """
    Enumeration for different query operators used in indexing.
    """

    EQ = "=="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="


class DocIndexer:
    """
    A class to handle indexing and searching of documents.

    Attributes:
        documents (List[Document]): List of documents to be indexed.
    """

    def __init__(self, documents):
        self.documents = documents
        self.index = self.build_index(documents)

    def build_index(self, documents):
        """
        Build an index for the given list of documents.

        Parameters:
            documents (List[Document]): The list of documents to be indexed.

        Returns:
            dict: The built index.
        """
        index = {}
        for doc in documents:
            for key, value in doc.metadata.items():
                if key not in index:
                    index[key] = {}
                if value not in index[key]:
                    index[key][value] = []
                index[key][value].append(doc)
        return index

    def retrieve_metadata(self, search_dict):
        """
        Retrieve documents based on the search criteria provided in search_dict.

        Parameters:
            search_dict (dict): Dictionary specifying the search criteria.
                                It can contain "AND" or "OR" operators for
                                complex queries.

        Returns:
            List[Document]: List of documents that match the search criteria.
        """
        if "AND" in search_dict:
            return self._handle_and(search_dict["AND"])
        elif "OR" in search_dict:
            return self._handle_or(search_dict["OR"])
        else:
            return self._handle_single(search_dict)

    def _handle_and(self, search_dicts):
        results = [self.retrieve_metadata(sd) for sd in search_dicts]
        if results:
            intersection = set.intersection(
                *[set(map(self._hash_doc, r)) for r in results]
            )
            return [self._unhash_doc(h) for h in intersection]
        else:
            return []

    def _handle_or(self, search_dicts):
        results = [self.retrieve_metadata(sd) for sd in search_dicts]
        union = set.union(*[set(map(self._hash_doc, r)) for r in results])
        return [self._unhash_doc(h) for h in union]

    def _handle_single(self, search_dict):
        unions = []
        for key, query in search_dict.items():
            operator, value = query
            union = set()
            if operator == IndexerOperator.EQ:
                if key in self.index and value in self.index[key]:
                    union.update(map(self._hash_doc, self.index[key][value]))
            else:
                if key in self.index:
                    for k, v in self.index[key].items():
                        if (
                            (operator == IndexerOperator.GT and k > value)
                            or (operator == IndexerOperator.GTE and k >= value)
                            or (operator == IndexerOperator.LT and k < value)
                            or (operator == IndexerOperator.LTE and k <= value)
                        ):
                            union.update(map(self._hash_doc, v))
            if union:
                unions.append(union)

        if unions:
            intersection = set.intersection(*unions)
            return [self._unhash_doc(h) for h in intersection]
        else:
            return []

    def _hash_doc(self, doc):
        return (doc.page_content, frozenset(doc.metadata.items()))

    def _unhash_doc(self, hashed_doc):
        page_content, metadata = hashed_doc
        return Document(page_content=page_content, metadata=dict(metadata))


def _get_chat_history(chat_history: List[ChatTurnType]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, BaseMessage):
            role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
            buffer += f"\n{role_prefix}{dialogue_turn.content}"
        elif isinstance(dialogue_turn, tuple):
            human = "Human: " + dialogue_turn[0]
            ai = "Assistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        else:
            raise ValueError(
                f"Unsupported chat history format: {type(dialogue_turn)}."
                f" Full chat history: {chat_history} "
            )
    return buffer


# TODO revise
def _get_standalone_questions_list(
    standalone_questions_str: str, original_question: str
) -> List[str]:
    match = re.search(
        r"(?i)standalone[^\n]*:\n(.*)", standalone_questions_str, re.DOTALL
    )

    sentence_source = match.group(1).strip() if match else standalone_questions_str
    sentences = sentence_source.split("\n")

    return [
        re.sub(r"^(?:\d+\.\s?|\(\d+\)\s?)", "", sentence.strip())
        for sentence in sentences
        if sentence.strip()
    ]
