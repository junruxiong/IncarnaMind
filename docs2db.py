"""
This module save documents to embeddings and langchain Documents.
"""
import os
import glob
import pickle
from typing import List
from multiprocessing import Pool
from collections import deque
import hashlib
import tiktoken

from tqdm import tqdm

from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

from toolkit.utils import Config, choose_embeddings, clean_text


# Load the config file
configs = Config("configparser.ini")

os.environ["OPENAI_API_KEY"] = configs.openai_api_key
os.environ["ANTHROPIC_API_KEY"] = configs.anthropic_api_key

embedding_store_path = configs.emb_dir
files_path = glob.glob(configs.docs_dir + "/*")

tokenizer_name = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokenizer = tiktoken.get_encoding(tokenizer_name.name)

loaders = {
    "pdf": (PyPDFLoader, {}),
    "txt": (TextLoader, {}),
}


def tiktoken_len(text: str):
    """Calculate the token length of a given text string using TikToken.

    Args:
        text (str): The text to be tokenized.

    Returns:
        int: The length of the tokenized text.
    """
    tokens = tokenizer.encode(text, disallowed_special=())

    return len(tokens)


def string2md5(text: str):
    """Convert a string to its MD5 hash.

    Args:
        text (str): The text to be hashed.

    Returns:
        str: The MD5 hash of the input string.
    """
    hash_md5 = hashlib.md5()
    hash_md5.update(text.encode("utf-8"))

    return hash_md5.hexdigest()


def load_file(file_path):
    """Load a file and return its content as a Document object.

    Args:
        file_path (str): The path to the file.

    Returns:
        Document: The loaded document.
    """
    ext = file_path.split(".")[-1]

    if ext in loaders:
        loader_type, args = loaders[ext]
        loader = loader_type(file_path, **args)
        doc = loader.load()

        return doc

    raise ValueError(f"Extension {ext} not supported")


def docs2vectorstore(docs: List[Document], embedding_name: str, suffix: str = ""):
    """Convert a list of Documents into a Chroma vector store.

    Args:
        docs (Document): The list of Documents.
        suffix (str, optional): Suffix for the embedding. Defaults to "".
    """
    embedding = choose_embeddings(embedding_name)
    name = f"{embedding_name}_{suffix}"
    # if embedding_store_path is not existing, create it
    if not os.path.exists(embedding_store_path):
        os.makedirs(embedding_store_path)
    Chroma.from_documents(
        docs,
        embedding,
        persist_directory=f"{embedding_store_path}/chroma_{name}",
    )


def file_names2pickle(file_names: list, save_name: str = ""):
    """Save the list of file names to a pickle file.

    Args:
        file_names (list): The list of file names.
        save_name (str, optional): The name for the saved pickle file. Defaults to "".
    """
    name = f"{save_name}"
    with open(f"{embedding_store_path}/{name}.pkl", "wb") as file:
        pickle.dump(file_names, file)


def docs2pickle(docs: List[Document], suffix: str = ""):
    """Serializes a list of Document objects to a pickle file.

    Args:
        docs (Document): List of Document objects.
        suffix (str, optional): Suffix for the pickle file. Defaults to "".
    """
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    name = f"pickle_{suffix}"

    with open(f"{embedding_store_path}/docs_{name}.pkl", "wb") as file:
        pickle.dump(docs, file)


def split_doc(
    doc: List[Document], chunk_size: int, chunk_overlap: int, chunk_idx_name: str
):
    """Splits a document into smaller chunks based on the provided size and overlap.

    Args:
        doc (Document): Document to be split.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between adjacent chunks.
        chunk_idx_name (str): Metadata key for storing chunk indices.

    Returns:
        list: List of Document objects representing the chunks.
    """
    data_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len,
    )
    doc_split = data_splitter.split_documents(doc)
    chunk_idx = 0

    for d_split in doc_split:
        d_split.metadata[chunk_idx_name] = chunk_idx
        chunk_idx += 1

    return doc_split


def process_metadata(doc: List[Document]):
    """Processes and updates the metadata for a list of Document objects.

    Args:
        doc (list): List of Document objects.
    """
    # get file name and remove extension
    file_name_with_extension = os.path.basename(doc[0].metadata["source"])
    file_name, _ = os.path.splitext(file_name_with_extension)

    for _, item in enumerate(doc):
        for key, value in item.metadata.items():
            if isinstance(value, list):
                item.metadata[key] = str(value)
        item.metadata["page_content"] = item.page_content
        item.metadata["page_content_md5"] = string2md5(item.page_content)
        item.metadata["source_md5"] = string2md5(item.metadata["source"])
        item.page_content = f"{file_name}\n{item.page_content}"


def add_window(
    doc: Document, window_steps: int, window_size: int, window_idx_name: str
):
    """Adds windowing information to the metadata of each document in the list.

    Args:
        doc (Document): List of Document objects.
        window_steps (int): Step size for windowing.
        window_size (int): Size of each window.
        window_idx_name (str): Metadata key for storing window indices.
    """
    window_id = 0
    window_deque = deque()

    for idx, item in enumerate(doc):
        if idx % window_steps == 0 and idx != 0 and idx < len(doc) - window_size:
            window_id += 1
        window_deque.append(window_id)

        if len(window_deque) > window_size:
            for _ in range(window_steps):
                window_deque.popleft()

        window = set(window_deque)
        item.metadata[f"{window_idx_name}_lower_bound"] = min(window)
        item.metadata[f"{window_idx_name}_upper_bound"] = max(window)


def merge_metadata(dicts_list: dict):
    """Merges a list of metadata dictionaries into a single dictionary.

    Args:
        dicts_list (list): List of metadata dictionaries.

    Returns:
        dict: Merged metadata dictionary.
    """
    merged_dict = {}
    bounds_dict = {}
    keys_to_remove = set()

    for dic in dicts_list:
        for key, value in dic.items():
            if key in merged_dict:
                if value not in merged_dict[key]:
                    merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    for key, values in merged_dict.items():
        if len(values) > 1 and all(isinstance(x, (int, float)) for x in values):
            bounds_dict[f"{key}_lower_bound"] = min(values)
            bounds_dict[f"{key}_upper_bound"] = max(values)
            keys_to_remove.add(key)

    merged_dict.update(bounds_dict)

    for key in keys_to_remove:
        del merged_dict[key]

    return {
        k: v[0] if isinstance(v, list) and len(v) == 1 else v
        for k, v in merged_dict.items()
    }


def merge_chunks(doc: Document, scale_factor: int, chunk_idx_name: str):
    """Merges adjacent chunks into larger chunks based on a scaling factor.

    Args:
        doc (Document): List of Document objects.
        scale_factor (int): The number of small chunks to merge into a larger chunk.
        chunk_idx_name (str): Metadata key for storing chunk indices.

    Returns:
        list: List of Document objects representing the merged chunks.
    """
    merged_doc = []
    page_content = ""
    metadata_list = []

    chunk_idx = 0
    for idx, item in enumerate(doc):
        page_content += item.page_content
        metadata_list.append(item.metadata)

        if ((idx + 1) % scale_factor == 0 and idx != 0) or (idx == len(doc) - 1):
            metadata = merge_metadata(metadata_list)
            metadata[chunk_idx_name] = chunk_idx
            merged_doc.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )
            chunk_idx += 1
            page_content = ""
            metadata_list = []

    return merged_doc


def process_files():
    """Main function for processing files. Loads, tokenizes, and saves document data."""
    with Pool() as pool:
        chunks_small = []
        chunks_medium = []
        file_names = []

        with tqdm(total=len(files_path), desc="Processing files", ncols=80) as pbar:
            for doc in pool.imap_unordered(load_file, files_path):
                file_name_with_extension = os.path.basename(doc[0].metadata["source"])

                chunk_split_small = split_doc(
                    doc=doc,
                    chunk_size=configs.base_chunk_size,
                    chunk_overlap=configs.chunk_overlap,
                    chunk_idx_name="small_chunk_idx",
                )
                add_window(
                    doc=chunk_split_small,
                    window_steps=configs.window_steps,
                    window_size=configs.window_scale,
                    window_idx_name="large_chunks_idx",
                )

                chunk_split_medium = merge_chunks(
                    doc=chunk_split_small,
                    scale_factor=configs.chunk_scale,
                    chunk_idx_name="medium_chunk_idx",
                )

                process_metadata(chunk_split_small)
                process_metadata(chunk_split_medium)

                file_names.append(file_name_with_extension)
                chunks_small.extend(chunk_split_small)
                chunks_medium.extend(chunk_split_medium)

                pbar.update()

    file_names2pickle(file_names, save_name="file_names")

    docs2vectorstore(chunks_small, configs.embedding_name, suffix="chunks_small")
    docs2vectorstore(chunks_medium, configs.embedding_name, suffix="chunks_medium")

    docs2pickle(chunks_small, suffix="chunks_small")
    docs2pickle(chunks_medium, suffix="chunks_medium")


if __name__ == "__main__":
    process_files()
