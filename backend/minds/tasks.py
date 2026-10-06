import asyncio
import copy
import uuid
from collections import Counter

import numpy as np
from celery import chain, chord, group, shared_task
from langchain_text_splitters import CharacterTextSplitter
from markdownify import markdownify as md
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from incarna_ai.clients import Clients, InstructorParams
from incarna_ai.utils import (
    filenames_filter,
    hybrid_search,
    process_sparse_batch,
    upload2vectordb,
)

# from minds.ai_retriever import Clients, InstructorParams
from minds.filters import first_filter, last_filter, second_filter
from vectordb import connect_vectordb_client


def first_search(query, user_uuid, num_queries):
    print("Query+++++++++++$$$$$", query)

    filter_1 = first_filter(user_uuid)
    res = hybrid_search(
        query_text=query,
        sparse_k=50,
        dense_k=50,
        top_k=50,
        filter=filter_1,
    )

    if not res or not res[0]:
        return None

    # print("Search Results:------")
    # for idx, r1 in enumerate(res[:40]):
    #     print("r1---:", idx, r1)
    #     print("================================================================")

    res_k = res[: min(20, len(res))]
    md5s = set()
    for r in res_k:
        md5s.add(r["file_md5"])

    if num_queries > 2:
        top_k = min(len(md5s) * 10, 50)
    else:
        top_k = min(len(md5s) * 24, 50)

    # snippets = get_snippets(res[: min(top_k, len(res))])
    file_names, idx2name_md5 = get_filenames(res[: min(top_k, len(res))])
    print("file_names++++++:", file_names)
    doc_md5s = filter_doc_md5s(res, file_names, idx2name_md5, query)
    print("filter_doc_md5s++++++:", doc_md5s)

    if not doc_md5s:
        return None
    # res_labels = get_relavent_snippets(snippets, query)

    return {"res": res, "doc_md5s": doc_md5s}


@shared_task()
def incarna_ai(input_query, query, user_uuid, num_queries):

    # 1st stage retrieval
    if input_query != query or len(input_query) > 2000:
        res_1 = first_search(
            # f"query 1:\n{input_query}\n\nquery 2:\n{query}",
            # user_uuid,
            # num_queries,
            f"```{query}```\n\n```{input_query}```",
            user_uuid,
            num_queries,
        )
        if res_1 is None:
            res_1 = first_search(f"{query}", user_uuid, num_queries)
            if not res_1:
                return None
    else:
        res_1 = first_search(f"{query}", user_uuid, num_queries)
        if not res_1:
            return None

    res_1_relevant = []
    for r_1 in res_1["res"]:
        if r_1["file_md5"] in res_1["doc_md5s"]:
            res_1_relevant.append(r_1)

    if num_queries > 2:
        top_k = min(len(res_1["doc_md5s"]) * 7, 20)
    else:
        top_k = min(len(res_1["doc_md5s"]) * 20, 30)

    res_indexed_1 = indexing_res_by_id(
        res_1_relevant[: min(top_k, len(res_1_relevant))]
    )
    # print("res_indexed_1:", res_indexed_1)
    res_1_grouped = find_group_l_chunks(res_indexed_1, query)
    print("res_1_grouped:", res_1_grouped)

    if not res_1_grouped:
        return None

    # last step

    k = 25 // max(num_queries, 1)
    res = []

    for r in res_1["res"]:
        idx_range = range(r["l_chunk_idx_from"], r["l_chunk_idx_to"] + 1)
        file_md5 = r["file_md5"]

        if file_md5 in res_1_grouped:
            if any(x in res_1_grouped[file_md5]["interval_pts"] for x in idx_range):
                res.append(r)
                k -= 1
            if k == 0:
                break
    print("res++++", res)

    return res


@shared_task
def update_point(
    new_text: str, block_id: str, mind_id: str, mind_name: str, tenant_id: str
):

    vectordb_client = connect_vectordb_client()

    res = vectordb_client.scroll(
        collection_name="incarnamind_s",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="uuid",
                    match=models.MatchText(text=block_id),
                ),
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchText(text=tenant_id),
                ),
            ]
        ),
    )
    # print("update_point++++", res, block_id)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400, chunk_overlap=100
    )
    texts = text_splitter.split_text(new_text)

    if res[0]:
        vectordb_client.delete(
            collection_name="incarnamind_s",
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="uuid",
                            match=models.MatchText(text=block_id),
                        ),
                        models.FieldCondition(
                            key="tenant_id",
                            match=models.MatchText(text=tenant_id),
                        ),
                    ],
                )
            ),
        )

        payloads, text_chunks, sparse_vectors = crete_points(
            texts, block_id, mind_id, mind_name, tenant_id
        )
        asyncio.run(
            upload2vectordb(
                collection_name="incarnamind_s",
                payloads=payloads,
                sparse_vectors=sparse_vectors,
                texts=text_chunks,
                # include_dense=False,
            )
        )
    else:
        payloads, text_chunks, sparse_vectors = crete_points(
            texts, block_id, mind_id, mind_name, tenant_id
        )
        asyncio.run(
            upload2vectordb(
                collection_name="incarnamind_s",
                payloads=payloads,
                sparse_vectors=sparse_vectors,
                texts=text_chunks,
                # include_dense=False,
            )
        )
    return f"successfully updated point {block_id}"


@shared_task
def rename_points(mind_id: str, new_mind_name: str, tenant_id: str):
    print("rename_points++++", mind_id, new_mind_name, tenant_id)
    # mind id must be converted to string
    vectordb_client = connect_vectordb_client()
    res = vectordb_client.scroll(
        collection_name="incarnamind_s",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="file_id",
                    match=models.MatchText(text=mind_id),
                ),
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchText(text=tenant_id),
                ),
            ]
        ),
    )

    if res[0]:
        vectordb_client.delete(
            collection_name="incarnamind_s",
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_id",
                            match=models.MatchText(text=mind_id),
                        ),
                        models.FieldCondition(
                            key="tenant_id",
                            match=models.MatchText(text=tenant_id),
                        ),
                    ],
                )
            ),
        )

        payloads, text_chunks, sparse_vectors = create_renamed_points(
            res, new_mind_name
        )

        # print("+++++", payloads, text_chunks)

        asyncio.run(
            upload2vectordb(
                collection_name="incarnamind_s",
                payloads=payloads,
                sparse_vectors=sparse_vectors,
                texts=text_chunks,
            )
        )
        return f"successfully renamed mind {new_mind_name}"


def create_renamed_points(results: any, new_mind_name: str):
    payloads = []
    page_contents = []
    seen_uuids = set()  # Set to track uuids and avoid duplicates

    for point in results[0]:
        if point.payload["uuid"] in seen_uuids:
            continue

        seen_uuids.add(point.payload["uuid"])

        new_payload = copy.deepcopy(point.payload)

        new_payload["file_name"] = new_mind_name
        new_file_name_prefix = f"File name: {new_mind_name}:\nContext: "
        new_file_name_len = len(new_file_name_prefix)

        new_payload["page_content"] = (
            f"{new_file_name_prefix}{point.payload['page_content'][new_file_name_len:]}"
        )
        new_payload["file_name_len"] = new_file_name_len

        # Add the modified payload to the list
        payloads.append(new_payload)
        page_contents.append(new_payload["page_content"])

    # Assuming process_sparse_batch is a function to process page_contents
    sparse_vectors = process_sparse_batch(page_contents)

    return payloads, page_contents, sparse_vectors


def crete_points(
    texts: str, block_id: str, mind_id: str, mind_name: str, tenant_id: str
):
    payloads = []
    page_contents = []
    text_chunks = copy.deepcopy(texts)
    file_name_prefix = f"File name: {mind_name}:\nContext: "
    file_name_len = len(file_name_prefix)

    for idx, text in enumerate(texts):
        payloads.append(
            {
                "uuid": f"{block_id}-{idx}",
                "tenant_id": tenant_id,
                "file_id": mind_id,
                "file_name": mind_name,
                "file_name_len": file_name_len,
                "page_content": f"{file_name_prefix}{text}",
                "type": "block",
            },
        )
        page_contents.append(f"{mind_name}\n{text}")

    sparse_vectors = process_sparse_batch(page_contents)

    return payloads, text_chunks, sparse_vectors


def get_snippets(reranked_results):
    snippets = ""
    for idx, p in enumerate(reranked_results):
        snippets += f"\n{idx}:\n{p['page_content']}\n"
    return snippets


def get_filenames(reranked_results):
    filenames = ""
    idx2name_md5 = {}
    filenames_set = set()
    idx = 0
    for _, p in enumerate(reranked_results):
        # if p["file_name"] not in filenames_set and len(filenames_set) <= 10:
        if p["file_name"] not in filenames_set and len(filenames_set) < 10:
            filenames += f"\nId: {idx}:\n```\n{p['file_name']}\n```"
            filenames_set.add(p["file_name"])
            idx2name_md5[idx] = {"file_name": p["file_name"], "file_md5": p["file_md5"]}
            idx += 1
        if len(filenames_set) == 10:
            break
    #     snippets += f"\n{idx}:\n{p['page_content']}\n"
    # return snippets
    print("idx2name_md5+++", idx2name_md5)
    return filenames, idx2name_md5


#
def get_relavent_snippets(snippets, query):
    client = Clients()
    instructor_params = InstructorParams()
    # try:
    prediction = filenames_filter(
        client.instrutor_client_azure_4o,
        instructor_params.azure_4o["provider"],
        instructor_params.azure_4o["model_name"],
        snippets,
        query,
    )

    print("prediction+++++++", prediction)

    # except Exception as e:
    #     print("get_relavent_snippets:", e)
    #     return None

    if prediction is None or not prediction.labels:
        return None

    return prediction.labels


def filter_doc_md5s(rerank_results, snippets, idx2name_md5, query):

    labels = get_relavent_snippets(snippets, query)
    if labels is None:
        print("get_relavent_snippets returned None")
        return None

    print("get_relavent_snippets:", labels)

    doc_names = set()
    doc_md5s = set()
    for p in labels:
        try:
            print("get_relavent_snippets+++++:", p, query)
            # doc_names.add(idx2name_md5[int(p)]["file_name"])
            doc_md5s.add(idx2name_md5[int(p)]["file_md5"])
        except Exception as e:
            print("filter_doc_md5s:", e)
            continue

    print("doc_md5s:", doc_md5s)

    return doc_md5s


def indexing_res_by_id(rerank_results):
    rerank_results_split = {}
    for doc in rerank_results:
        md5 = doc["file_md5"]
        if md5 not in rerank_results_split:
            rerank_results_split[md5] = []
        rerank_results_split[md5].append(doc)
    return rerank_results_split


def interval_intersection(intervals):
    intersection = intervals[0]
    min_idx = intervals[0][2]

    for i in range(1, len(intervals)):
        start, end, idx = intervals[i]

        intersection_start = max(intersection[0], start)
        intersection_end = min(intersection[1], end)

        if intersection_start > intersection_end:
            return None

        if idx < min_idx:
            min_idx = idx

        intersection = (intersection_start, intersection_end, min_idx)

    return intersection


def find_clusters(intervals):
    intervals.sort(key=lambda x: x[0])

    result_sets = []
    current_set = []
    min_right = float("inf")
    for interval in intervals:
        if interval[0] <= min_right:
            current_set.append(interval)
            if min_right > interval[1]:
                min_right = interval[1]
        else:
            result_sets.append(current_set)
            last_set_overlapping = [
                x for x in current_set if x[0] <= interval[1] and x[1] >= interval[0]
            ]
            current_set = last_set_overlapping + [interval]
            min_right = min([x[1] for x in last_set_overlapping] + [interval[1]])

    if current_set:
        result_sets.append(current_set)

    return result_sets


def get_centroids(intervals):
    sets = find_clusters(intervals)
    # print("sets+++++++", sets)

    centroids = []
    interval_pts = set()
    for s in sets:
        if len(s) > 1:
            overlap = interval_intersection(s)
            centroid = (overlap[0] + overlap[1]) // 2
            centroids.append((centroid, overlap[2]))

            # add integers between (centroid, overlap[2]) to intervals
            interval_pts.update(range(centroid, overlap[2] + 1))

        else:

            centroids.append((s[0][2], s[0][2]))
            interval_pts.add(s[0][2])
            # centroids.append((None, s[0][2]))

    print("intervals++++++++++++", interval_pts)
    return centroids, interval_pts


def find_group_l_chunks(res_indexed_2, query):
    res = {}

    for k, v in res_indexed_2.items():
        intervals = [
            (item["l_chunk_idx_from"], item["l_chunk_idx_to"], item["idx"])
            for item in v
        ]
        print("intervals:", intervals)
        centrodis, interval_pts = get_centroids(intervals)
        print("centroids:", centrodis)
        res[k] = {"centroids": centrodis, "query": query, "interval_pts": interval_pts}

    return res
