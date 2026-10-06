import asyncio
import json
import os

# import re
# import string
import re
import uuid
from typing import Iterable, List, Literal, Optional

# import nltk
import requests

# import torch
from anthropic import Anthropic
from django.http import JsonResponse

# from langdetect import detect
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer, WordNetLemmatizer
# from nltk.tokenize import word_tokenize
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, Field, ValidationInfo, model_validator
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.models import Batch

from vectordb import (
    connect_cohere_aclient,
    connect_cohere_client,
    connect_vectordb_aclient,
    connect_vectordb_client,
    connect_voyage_client,
)


def process_sparse_batch(texts, batch_size=1000):
    # API_URL = os.environ.get("SPLADE_DOC_URL")
    API_URL = "https://gwlhfaq5d11i2fgm.us-east-1.aws.endpoints.huggingface.cloud"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {os.environ.get('SPLADE_DOC_KEY')}",
        "Content-Type": "application/json",
    }

    sparse_vectors = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        payload = {"inputs": batch_texts}
        response = requests.post(API_URL, headers=headers, json=payload)
        outputs = response.json()[0]
        response.close()

        for indices, values in zip(outputs["indices"], outputs["values"]):
            sparse_vectors.append(models.SparseVector(indices=indices, values=values))

    return sparse_vectors


# try multi label classifier
LABELS = Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


class FilenamesSelect(BaseModel):
    labels: List[LABELS] = Field(
        ...,
        description="Only select the file name ids that are relavent to the user input.",
    )


def filenames_filter(
    client: Anthropic | OpenAI | AzureOpenAI,
    provider: str,
    model_name: str,
    filenames: str,
    input: str,
) -> FilenamesSelect:
    messages = [
        # {
        #     "role": "system",
        #     "content": """Only select file name ids that are relavent to any one of user input(s). The file name and user's input(s) might be in different language.
        #     """,
        # },
        {
            "role": "system",
            "content": f"You are a multilingual file names' id filter.",
        },
        {
            "role": "user",
            "content": f"Only select file name ids that are relavent to any one of user input(s). If no file name is relevant user's input(s), please return None.\n\nFile names:\n{filenames}\n\nUser input(s):```{input}\n```",
        },
    ]

    if provider == "openai" or provider == "azure":
        return client.chat.completions.create(
            model=model_name,  # gpt-3.5-turbo fails
            response_model=FilenamesSelect,
            temperature=0.1,
            max_tokens=500,
            max_retries=3,
            messages=messages,
        )  # type: ignore

    if provider == "anthropic":
        return client.messages.create(
            model=model_name,
            response_model=FilenamesSelect,
            temperature=0,
            max_tokens=500,
            max_retries=3,
            messages=messages,
        )  # type: ignore

    # get docs uuid

    return None


class Search(BaseModel):
    query: str = Field(
        description="The direct dependency search query of the original user input."
    )


def segment_query(
    client,
    provider: str,
    model_name: str,
    chat_history: str,
    file_names: str,
    query: str,
) -> Search:

    # if chat_history is not empty string

    if chat_history:
        messages = [
            {
                "role": "system",
                "content": f"You are a multilingual query segmenter. Based on the chat history and file names, segment them into one or multiple the most direct dependency search queries in its original language. Each query must has clear full file name (no pronouns or abbreviations), if the user input might be relevant to file names. Otherwise, return None. These queries should be designed to gather information that can inform the user input  or chat history.",
            },
            {
                "role": "user",
                "content": f"Chat History:\n```{chat_history}```\n\Available file names:\n```{file_names}```\n\nUser input:\n```\n{query}\n```",
            },
        ]

    else:
        messages = [
            {
                "role": "system",
                "content": f"You are a multilingual query segmenter. Based on the file names, segment them into one or multiple the most direct dependency search queries in its original language. Each query must has clear full file name (no pronouns or abbreviations), if the user input might be relevant to file names. Otherwise, return None. These queries should be designed to gather information that can inform the user input or chat history.",
            },
            {
                "role": "user",
                "content": f"Available file names:\n```{file_names}```\n\nUser input:\n```\n{query}\n```",
            },
        ]

    if provider == "openai" or provider == "azure":
        return client.chat.completions.create(
            model=model_name,  # gpt-3.5-turbo fails
            response_model=Iterable[Search],
            temperature=0.1,
            max_tokens=1024,
            max_retries=3,
            messages=messages,
        )  # type: ignore

    if provider == "anthropic" or provider == "gcp_anthropic":
        return client.messages.create(
            model=model_name,
            response_model=Iterable[Search],
            temperature=0.1,
            max_tokens=1024,
            max_retries=3,
            messages=messages,
        )  # type: ignore

    if provider == "gcp_gemini" or provider == "gemini":
        return client.messages.create(
            model=model_name,
            response_model=Iterable[Search],
            temperature=0.1,
            max_tokens=1024,
            max_retries=3,
            messages=messages,
        )  # type: ignore

    return None


def execute_segment_query(searches: Iterable[Search]) -> List[str]:
    results = []
    for search in searches:
        results.append(search.query)
    return results


# def compute_sparse_vector(text, tokenizer_type: str):
#     """
#     Computes a vector from logits and attention mask using ReLU, log, and max operations.

#     Args:
#     logits (torch.Tensor): The logits output from a model.
#     attention_mask (torch.Tensor): The attention mask corresponding to the input tokens.

#     Returns:
#     torch.Tensor: Computed vector.
#     """

#     if tokenizer_type == "query":
#         query_model_id = "naver/efficient-splade-VI-BT-large-query"
#         tokenizer = AutoTokenizer.from_pretrained(query_model_id)
#         model = AutoModelForMaskedLM.from_pretrained(query_model_id)
#     else:  # doc
#         doc_model_id = "naver/efficient-splade-VI-BT-large-doc"
#         tokenizer = AutoTokenizer.from_pretrained(doc_model_id)
#         model = AutoModelForMaskedLM.from_pretrained(doc_model_id)

#     tokens = tokenizer(text, return_tensors="pt")
#     output = model(**tokens)
#     logits, attention_mask = output.logits, tokens.attention_mask
#     relu_log = torch.log(1 + torch.relu(logits))
#     weighted_log = relu_log * attention_mask.unsqueeze(-1)
#     max_val, _ = torch.max(weighted_log, dim=1)
#     vec = max_val.squeeze()

#     return vec, tokens


def rerank(query, page_contents, top_k=10):
    contents_reranked = connect_voyage_client().rerank(
        query,
        page_contents,
        model="rerank-1",
        top_k=top_k,
    )
    return contents_reranked


def generate_snippets(query_results):

    # file_names = ""
    # file_name_set = set()
    # for d_name_md5 in docs_name_md5:
    #     for d in d_name_md5:
    #         if d[1] not in file_name_set:
    #             file_names += f"{d[1]}, "
    #             file_name_set.add(d[1])

    metadata = {"citations": []}
    merged_docs = {}
    citation_ids_set = set()
    processed_uuids = set()
    for query_result in query_results:
        if query_result is None:
            continue
        for res in query_result:
            uuid = res["uuid"]
            if uuid in processed_uuids:
                continue

            file_name = res["file_name"]
            file_md5 = res["file_md5"]
            file_id = res["file_id"]

            if file_id not in citation_ids_set:
                # add to metadata
                metadata["citations"].append(
                    {
                        "file_id": file_id,
                        "file_name": file_name,
                        "file_md5": file_md5,
                    }
                )
                citation_ids_set.add(file_id)

            if file_md5 not in merged_docs:
                merged_docs[file_md5] = {"docs": [], "file_name": file_name}

            merged_docs[file_md5]["docs"].append(res)
            processed_uuids.add(uuid)

    for key in merged_docs:
        merged_docs[key]["docs"] = sorted(
            merged_docs[key]["docs"], key=lambda x: x["idx"]
        )

    snippets = ""
    for k, v in merged_docs.items():
        snippets += f"{v['file_name']}: "
        prev_idx = -99999
        for i, doc in enumerate(v["docs"]):
            current_idx = doc["idx"]
            if prev_idx + 1 == doc["idx"]:
                prefix_len = doc["file_name_len"] + doc["prev_overlap_len"]
                snippets += f"{doc['page_content'][prefix_len:]}\n"
            else:
                prefix_len = doc["file_name_len"]
                snippets += f"\n{doc['page_content'][prefix_len:]}\n"
            prev_idx = current_idx

    print("snippets:", snippets)

    return snippets, metadata


def merge_strings_rabin_karp(s1, s2):
    # Constants for the hashing algorithm
    p1, m1 = 31, 1e9 + 7
    p2, m2 = 37, 1e9 + 9

    # Helper function to compute dual hashes of a string
    def compute_dual_hashes(s):
        hash1, hash2 = 0, 0
        p1_pow, p2_pow = 1, 1
        for char in s:
            hash1 = (hash1 + (ord(char) - ord("a") + 1) * p1_pow) % m1
            p1_pow = (p1_pow * p1) % m1
            hash2 = (hash2 + (ord(char) - ord("a") + 1) * p2_pow) % m2
            p2_pow = (p2_pow * p2) % m2
        return (hash1, hash2)

    max_overlap_len = 0

    # Compute hashes for all suffixes of s1
    suffix_hashes = {}
    for i in range(1, min(len(s1), len(s2)) + 1):
        hash_pair = compute_dual_hashes(s1[-i:])
        suffix_hashes[hash_pair] = i

    # Check for matches with prefixes of s2
    for j in range(1, min(len(s1), len(s2)) + 1):
        prefix_hash_pair = compute_dual_hashes(s2[:j])
        if prefix_hash_pair in suffix_hashes:
            if s1[-suffix_hashes[prefix_hash_pair] :] == s2[:j]:
                if j > max_overlap_len:
                    max_overlap_len = j

    # Merge s1 with the non-overlapping part of s2
    merged_string = s1 + s2[max_overlap_len:]
    return merged_string


async def upload2vectordb(
    collection_name, payloads, sparse_vectors, texts, include_dense=True, batch_size=30
):
    # vectordb_client = connect_vectordb_client()
    cohere_aclient = connect_cohere_aclient()
    vectordb_aclient = connect_vectordb_aclient()

    tasks = []
    for i in range(0, len(texts), batch_size):
        chunk_payloads = payloads[i : i + batch_size]
        chunk_sparse_vectors = sparse_vectors[i : i + batch_size]
        chunk_texts = texts[i : i + batch_size]

        # Schedule the upsert operations to run concurrently
        task = upsert(
            vectordb_aclient,
            chunk_texts,
            collection_name,
            chunk_payloads,
            chunk_sparse_vectors,
            cohere_aclient,
            include_dense,
        )
        tasks.append(task)

    # Wait for all scheduled tasks to complete
    await asyncio.gather(*tasks)

    await vectordb_aclient.close()


async def upsert(
    vectordb_aclient,
    chunk_texts,
    collection_name,
    chunk_payloads,
    chunk_sparse_vectors,
    cohere_client,
    include_dense,
):
    # Generate 2 uuid.uuid4() list with different suffix (sparse,dense) for sparse and dense vectors
    sparse_ids = [str(uuid.uuid4()) for _ in range(len(chunk_texts))]
    dense_ids = [str(uuid.uuid4()) for _ in range(len(chunk_texts))]

    # First upsert for sparse vectors
    await vectordb_aclient.upsert(
        collection_name=collection_name,
        points=Batch(
            ids=sparse_ids,
            payloads=chunk_payloads,
            vectors={"text-sparse": chunk_sparse_vectors},
        ),
    )

    if include_dense == True:
        # Embeddings for dense vectors - assuming this is an async operation
        dense_vectors = await cohere_client.embed(
            model="embed-multilingual-v3.0",
            input_type="search_document",
            texts=chunk_texts,
        )

        # Second upsert for dense vectors
        await vectordb_aclient.upsert(
            collection_name=collection_name,
            points=Batch(
                ids=dense_ids,
                payloads=chunk_payloads,
                vectors={"text-dense": dense_vectors.embeddings},
            ),
        )


def hybrid_search(
    query_text,
    sparse_k=15,
    dense_k=10,
    top_k=10,
    filter=None,
    collection_name: str = "incarnamind_s",
    rerank: bool = True,
    with_payload=True,
):
    vectordb_client = connect_vectordb_client()
    cohere_client = connect_cohere_client()

    # query_vec, query_tokens = compute_sparse_vector(query_text, "query")
    # query_indices = query_vec.nonzero().numpy().flatten()
    # query_values = query_vec.detach().numpy()[query_indices]

    sparse_vectors = process_sparse_batch([query_text], batch_size=1)

    try:
        results = vectordb_client.search_batch(
            collection_name=collection_name,
            requests=[
                models.SearchRequest(
                    vector=models.NamedVector(
                        name="text-dense",
                        vector=cohere_client.embed(
                            model="embed-multilingual-v3.0",
                            input_type="search_query",
                            texts=[query_text],
                        ).embeddings[0],
                    ),
                    # with_payload=["page_content"],
                    with_payload=with_payload,
                    filter=filter,
                    limit=dense_k,
                ),
                models.SearchRequest(
                    vector=models.NamedSparseVector(
                        name="text-sparse",
                        vector=sparse_vectors[0],
                        # vector=models.SparseVector(
                        #     indices=query_indices.tolist(),
                        #     values=query_values.tolist(),
                        # ),
                    ),
                    with_payload=with_payload,
                    filter=filter,
                    limit=sparse_k,
                ),
            ],
        )

        vectordb_client.close()

        if not rerank:
            return results

        reranked_results = apply_rerank(query_text, results, top_k)

        return reranked_results

    except Exception as e:
        return None


async def ahybrid_search(
    query_text,
    sparse_k=15,
    dense_k=10,
    top_k=10,
    filter=None,
    collection_name: str = "incarnamind_s",
):
    vectordb_aclient = connect_vectordb_aclient()
    cohere_aclient = connect_cohere_aclient()

    # query_vec, query_tokens = compute_sparse_vector(query_text, "query")
    # query_indices = query_vec.nonzero().numpy().flatten()
    # query_values = query_vec.detach().numpy()[query_indices]

    sparse_vectors = process_sparse_batch([query_text], batch_size=1)

    try:
        results = await vectordb_aclient.search_batch(
            collection_name=collection_name,
            requests=[
                models.SearchRequest(
                    vector=models.NamedVector(
                        name="text-dense",
                        vector=cohere_aclient.embed(
                            model="embed-multilingual-v3.0",
                            input_type="search_query",
                            texts=[query_text],
                        ).embeddings[0],
                    ),
                    # with_payload=["page_content"],
                    with_payload=True,
                    filter=filter,
                    limit=dense_k,
                ),
                models.SearchRequest(
                    vector=models.NamedSparseVector(
                        name="text-sparse",
                        vector=sparse_vectors[0],
                        # vector=models.SparseVector(
                        #     indices=query_indices.tolist(),
                        #     values=query_values.tolist(),
                        # ),
                    ),
                    with_payload=True,
                    filter=filter,
                    limit=sparse_k,
                ),
            ],
        )

        vectordb_aclient.close()

        reranked_results = apply_rerank(query_text, results, top_k)

        return reranked_results

    except Exception as e:
        return None


def apply_rerank(query_text, results, top_k):
    unique_id_set, unique_results, page_contents = set(), list(), list()

    try:
        for res in results:
            for r in res:
                if r.payload["uuid"] not in unique_id_set:
                    unique_id_set.add(r.payload["uuid"])
                    unique_results.append(r.payload)
                    page_contents.append(r.payload["page_content"])

        page_contents_rerank = rerank(query_text, page_contents, top_k)
        reranked_results = [
            unique_results[p.index] for p in page_contents_rerank.results
        ]

        return reranked_results

    except Exception as e:
        print(e)
        return None


def extract_list_string(list_like_string):

    match = re.search(r"\[(.*)\]", list_like_string)
    if not match:
        return []

    list_content = match.group(1)

    # Removing the leading 'find documents in' and trailing 'about machine learning'
    list_content = list_content.replace("find documents in ", "", 1).replace(
        " about machine learning", "", 1
    )

    # Using regular expression to split on commas outside of double quotes
    items = re.findall(r'"(.*?)"|\'(.*?)\'|([^,]+)', list_content)
    items = [item[0] or item[1] or item[2].strip() for item in items]

    return items
