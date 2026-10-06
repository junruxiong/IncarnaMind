import os
import tempfile
import uuid
from collections import deque
from multiprocessing import Pool
from typing import List

import tiktoken
import weaviate.classes as wvc
from azure.storage.blob import BlobClient

# from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

# from incarna_ai.incarna_text_splitters import RecursiveCharacterTextSplitter
from vectordb import connect_vectordb_client, connect_voyage_client

# from langchain_voyageai import VoyageAIEmbeddings


loaders = {
    "pdf": (PyPDFLoader, {"ext_name": ".pdf"}),
    "application/pdf": (PyPDFLoader, {"ext_name": ".pdf"}),
}

tokenizer_name = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokenizer = tiktoken.get_encoding(tokenizer_name.name)


def tiktoken_len(text: str):
    """Calculate the token length of a given text string using TikToken.

    Args:
        text (str): The text to be tokenized.

    Returns:
        int: The length of the tokenized text.
    """
    tokens = tokenizer.encode(text, disallowed_special=())

    return len(tokens)


def load_file(file_path, file_type):

    if file_type in loaders:
        loader_type, loader_args = loaders[file_type]
        loader_init_args = {k: v for k, v in loader_args.items() if k != "ext_name"}
        loader = loader_type(file_path, **loader_init_args)
        doc = loader.load()

        return doc

    raise ValueError(f"Extension {file_type} not supported.")


def load_blob(file_dir, file_type, user_uuid):

    container = "media"
    blob_name = f"files/{user_uuid}/{file_dir}"
    client = BlobClient.from_connection_string(
        conn_str=os.environ.get("AZURE_ACCOUNT_CONN_STRING"),
        blob_name=blob_name,
        container_name=container,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = f"{temp_dir}/{container}/{blob_name}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(f"{file_path}", "wb") as file:
            blob_data = client.download_blob()
            blob_data.readinto(file)
        print("Downloaded file to: ", file_path)

        if file_type in loaders:
            loader_type, loader_args = loaders[file_type]
            loader_init_args = {k: v for k, v in loader_args.items() if k != "ext_name"}
            loader = loader_type(file_path, **loader_init_args)
            doc = loader.load()

            return doc

    return ValueError(f"Extension {file_type} not supported.")


def process_metadata(
    doc: List[Document], file_name: str, file_md5: str, file_id: str, user_uuid: str
):
    """Processes and updates the metadata for a list of Document objects.

    Args:
        doc (list): List of Document objects.
    """
    text = list()
    payload = list()

    # remove file_name extension
    # file_name_ = os.path.splitext(file_name)[0]

    for idx, d in enumerate(doc):
        page_from = d.metadata.get("page_from", d.metadata.get("page", None))
        page_to = d.metadata.get("page_to", d.metadata.get("page", None))

        file_name_prefix = f"File name: {file_name}:\nSnippet: "
        t = f"{file_name_prefix}{d.page_content}"
        text.append(t)
        # text.append(f"{d.page_content}")

        file_name_len = len(file_name_prefix)

        if idx > 0:
            overlap_len = max_overlap(doc[idx - 1].page_content, doc[idx].page_content)
        else:
            overlap_len = 0

        payload.append(
            {
                "uuid": str(uuid.uuid4()),
                "tenant_id": user_uuid,
                "file_name": file_name,
                "page_content": t,
                "file_name_len": file_name_len,
                "prev_overlap_len": overlap_len,
                "source": "/".join(d.metadata["source"].split("/", 2)[2:]),
                "idx": d.metadata["idx"],
                # "s_idx": d.metadata.get("s_idx", None),
                # "l_chunk_idx": d.metadata["l_chunk_idx"],
                "l_chunk_idx_from": d.metadata["l_chunk_idx_from"],
                "l_chunk_idx_to": d.metadata["l_chunk_idx_to"],
                "page_from": page_from,
                "page_to": page_to,
                "file_md5": file_md5,
                "file_id": file_id,
                "type": "doc",
            }
        )

    return text, payload


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

    # get the total words count
    total_tokens = 0
    for d in doc:
        total_tokens += tiktoken_len(d.page_content)

    doc_split = data_splitter.split_documents(doc)

    chunk_idx = 0
    for d_split in doc_split:
        d_split.metadata[chunk_idx_name] = chunk_idx

        chunk_idx += 1

    return doc_split, total_tokens


def count_tokens(doc: List[Document]):
    """Calculates the total number of tokens in a list of Document objects.

    Args:
        doc (list): List of Document objects.

    Returns:
        int: Total number of tokens.
    """
    total_tokens = 0
    for d in doc:
        total_tokens += tiktoken_len(d.page_content)
    return total_tokens


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
        # add to the right of the window
        window_deque.append(idx)

        if len(window_deque) > window_size:
            for _ in range(window_steps):
                window_deque.popleft()

        window = set(window_deque)
        item.metadata[f"{window_idx_name}"] = list(window)
        item.metadata[f"{window_idx_name}_from"] = min(window)
        item.metadata[f"{window_idx_name}_to"] = max(window)


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
            bounds_dict[f"{key}_from"] = min(values)
            bounds_dict[f"{key}_to"] = max(values)
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

        if (idx + 1) % scale_factor == 0 or idx == len(doc) - 1:
            metadata = merge_metadata(metadata_list)
            metadata[chunk_idx_name] = chunk_idx
            metadata["s_idx"] = idx
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


def uploadfile(file_item, file_type):
    if file_type not in loaders:
        raise ValueError("Unsupported file type")

    with tempfile.NamedTemporaryFile(
        delete=False, prefix=loaders[file_type][1]["ext_name"]
    ) as tmp:
        for chunk in file_item.chunks():
            tmp.write(chunk)
        tmp.seek(0)
        try:
            if file_type in loaders:
                loader_type, loader_args = loaders[file_type]
                loader_init_args = {
                    k: v for k, v in loader_args.items() if k != "ext_name"
                }
                loader = loader_type(tmp.name, **loader_init_args)
                doc = loader.load()
        except Exception as e:
            raise ValueError(f"Error loading document: {e}, 109")
        finally:
            os.unlink(tmp.name)

    return doc


def delete_file_vectordb(file_md5, user_uuid):
    vectordb_client = connect_vectordb_client()
    vectordb_client.delete(
        collection_name="incarnamind_s",
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_md5",
                        match=models.MatchValue(value=str(file_md5)),
                    ),
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=str(user_uuid)),
                    ),
                ],
            )
        ),
    )
    vectordb_client.close()


# def rename_file_vectordb(file_md5, user_uuid):
#     vectordb_client = connect_vectordb_client()
#     tenant_s = get_tenant(vectordb_client, "incarnamind_s", user_uuid)
#     tenant_m = get_tenant(vectordb_client, "incarnamind_m", user_uuid)

#     tenant_s.data.delete_many(
#         where=wvc.query.Filter.by_property("file_md5").equal(file_md5)
#     )
#     tenant_m.data.delete_many(
#         where=wvc.query.Filter.by_property("file_md5").equal(file_md5)
#     )

#     vectordb_client.close()


def compute_pi_table(pattern: str) -> list:
    """
    Compute the partial match table (pi-table) used by the KMP algorithm.
    """
    pi_table = [0] * len(pattern)
    j = 0

    for i in range(1, len(pattern)):
        while j > 0 and pattern[i] != pattern[j]:
            j = pi_table[j - 1]

        if pattern[i] == pattern[j]:
            j += 1
            pi_table[i] = j

    return pi_table


def kmp_search(text: str, pattern: str) -> int:
    """
    Use the KMP algorithm to find the maximum overlap between text and pattern.
    """
    if not pattern:
        return 0

    pi_table = compute_pi_table(pattern)
    j = 0

    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = pi_table[j - 1]

        if text[i] == pattern[j]:
            j += 1

            if j == len(
                pattern
            ):  # Match found (not used here but part of KMP structure)
                j = pi_table[j - 1]

    return j  # Length of the overlap


def max_overlap(s1: str, s2: str) -> int:
    """
    Compute the maximum overlap using the KMP search algorithm.
    """
    return kmp_search(s1, s2)
