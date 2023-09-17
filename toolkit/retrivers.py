"""
This module provides custom implementation of a document retriever, designed for multi-stage retrieval.
The system uses ensemble methods combining BM25 and Chroma Embeddings to retrieve relevant documents for a given query.
It also utilizes various optimizations like rank fusion and weighted reciprocal rank by Langchain.

Classes:
--------
- MyEnsembleRetriever: Custom retriever for BM25 and Chroma Embeddings.
- MyRetriever: Handles multi-stage retrieval.

"""
import re
import ast
import copy
import math
import logging
from typing import Dict, List, Optional
from langchain.chains import LLMChain
from langchain.schema import BaseRetriever, Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from toolkit.utils import Config, clean_text, DocIndexer, IndexerOperator
from toolkit.prompts import DOCS_SELECTION_PROMPT


configs = Config("configparser.ini")
logger = logging.getLogger(__name__)


class MyEnsembleRetriever(EnsembleRetriever):
    """
    Custom retriever for BM24 and Chroma Embeddings
    """

    retrievers: Dict[str, BaseRetriever]

    def rank_fusion(
        self, query: str, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve the results of the retrievers and use rank_fusion_func to get
        the final result.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """
        # Get the results of all retrievers.
        retriever_docs = []
        for key, retriever in self.retrievers.items():
            if key == "bm25":
                res = retriever.get_relevant_documents(
                    clean_text(query),
                    callbacks=run_manager.get_child(tag=f"retriever_{key}"),
                )
                retriever_docs.append(res)
            else:
                res = retriever.get_relevant_documents(
                    query, callbacks=run_manager.get_child(tag=f"retriever_{key}")
                )
                retriever_docs.append(res)

        # apply rank fusion
        fused_documents = self.weighted_reciprocal_rank(retriever_docs)

        return fused_documents

    async def arank_fusion(
        self, query: str, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Asynchronously retrieve the results of the retrievers
        and use rank_fusion_func to get the final result.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get the results of all retrievers.
        retriever_docs = []
        for key, retriever in self.retrievers.items():
            if key == "bm25":
                res = retriever.get_relevant_documents(
                    clean_text(query),
                    callbacks=run_manager.get_child(tag=f"retriever_{key}"),
                )
                retriever_docs.append(res)
                # print("retriever_docs 1:", res)
            else:
                res = await retriever.aget_relevant_documents(
                    query, callbacks=run_manager.get_child(tag=f"retriever_{key}")
                )
                retriever_docs.append(res)

        # apply rank fusion
        fused_documents = self.weighted_reciprocal_rank(retriever_docs)

        return fused_documents

    def weighted_reciprocal_rank(
        self, doc_lists: List[List[Document]]
    ) -> List[Document]:
        """
        Perform weighted Reciprocal Rank Fusion on multiple rank lists.
        You can find more details about RRF here:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.

        Returns:
            list: The final aggregated list of items sorted by their weighted RRF
                    scores in descending order.
        """
        if len(doc_lists) != len(self.weights):
            raise ValueError(
                "Number of rank lists must be equal to the number of weights."
            )

        # replace the page_content with the original uncleaned page_content
        doc_lists_ = copy.copy(doc_lists)
        for doc_list in doc_lists_:
            for doc in doc_list:
                doc.page_content = doc.metadata["page_content"]
                # doc.metadata["page_content"] = None

        # Create a union of all unique documents in the input doc_lists
        all_documents = set()
        for doc_list in doc_lists_:
            for doc in doc_list:
                all_documents.add(doc.page_content)

        # Initialize the RRF score dictionary for each document
        rrf_score_dic = {doc: 0.0 for doc in all_documents}

        # Calculate RRF scores for each document
        for doc_list, weight in zip(doc_lists_, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score = weight * (1 / (rank + self.c))
                rrf_score_dic[doc.page_content] += rrf_score

        # Sort documents by their RRF scores in descending order
        sorted_documents = sorted(
            rrf_score_dic.keys(), key=lambda x: rrf_score_dic[x], reverse=True
        )

        # Map the sorted page_content back to the original document objects
        page_content_to_doc_map = {
            doc.page_content: doc for doc_list in doc_lists_ for doc in doc_list
        }
        sorted_docs = [
            page_content_to_doc_map[page_content] for page_content in sorted_documents
        ]

        return sorted_docs


class MyRetriever:
    """
    Retriever class to handle multi-stage retrieval.
    """

    def __init__(
        self,
        llm,
        embedding_chunks_small: List[Document],
        embedding_chunks_medium: List[Document],
        docs_chunks_small: DocIndexer,
        docs_chunks_medium: DocIndexer,
        first_retrieval_k: int,
        second_retrieval_k: int,
        num_windows: int,
        retriever_weights: List[float],
    ):
        """
        Initialize the MyRetriever class.

        Args:
            llm: Language model for retrieval.
            embedding_chunks_small (List[Document]): List of small embedding chunks.
            embedding_chunks_medium (List[Document]): List of medium embedding chunks.
            docs_chunks_small (DocIndexer): Document indexer for small chunks.
            docs_chunks_medium (DocIndexer): Document indexer for medium chunks.
            first_retrieval_k (int): Number of top documents to retrieve in first retrieval.
            second_retrieval_k (int): Number of top documents to retrieve in second retrieval.
            num_windows (int): Number of overlapping windows to consider.
            retriever_weights (List[float]): Weights for ensemble retrieval.
        """
        self.llm = llm
        self.embedding_chunks_small = embedding_chunks_small
        self.embedding_chunks_medium = embedding_chunks_medium
        self.docs_index_small = DocIndexer(docs_chunks_small)
        self.docs_index_medium = DocIndexer(docs_chunks_medium)

        self.first_retrieval_k = first_retrieval_k
        self.second_retrieval_k = second_retrieval_k
        self.num_windows = num_windows
        self.retriever_weights = retriever_weights

    def get_retriever(
        self,
        docs_chunks,
        emb_chunks,
        emb_filter=None,
        k=2,
        weights=(0.5, 0.5),
    ):
        """
        Initialize and return a retriever instance with specified parameters.

        Args:
            docs_chunks: The document chunks for the BM25 retriever.
            emb_chunks: The document chunks for the Embedding retriever.
            emb_filter: A filter for embedding retriever.
            k (int): The number of top documents to return.
            weights (list): Weights for ensemble retrieval.

        Returns:
            MyEnsembleRetriever: An instance of MyEnsembleRetriever.
        """
        bm25_retriever = BM25Retriever.from_documents(docs_chunks)
        bm25_retriever.k = k

        emb_retriever = emb_chunks.as_retriever(
            search_kwargs={
                "filter": emb_filter,
                "k": k,
                "search_type": "mmr",
            }
        )
        return MyEnsembleRetriever(
            retrievers={"bm25": bm25_retriever, "chroma": emb_retriever},
            weights=weights,
        )

    def find_overlaps(self, doc: List[Document]):
        """
        Find overlapping intervals of windows.

        Args:
            doc (Document): A document object to find overlaps in.

        Returns:
            list: A list of overlapping intervals.
        """
        intervals = []
        for item in doc:
            intervals.append(
                (
                    item.metadata["large_chunks_idx_lower_bound"],
                    item.metadata["large_chunks_idx_upper_bound"],
                )
            )
        remaining_intervals, grouped_intervals, centroids = intervals.copy(), [], []

        while remaining_intervals:
            curr_interval = remaining_intervals.pop(0)
            curr_group = [curr_interval]
            subset_interval = None

            for start, end in remaining_intervals.copy():
                for s, e in curr_group:
                    overlap = set(range(s, e + 1)) & set(range(start, end + 1))
                    if overlap:
                        curr_group.append((start, end))
                        remaining_intervals.remove((start, end))
                        if set(range(start, end + 1)).issubset(set(range(s, e + 1))):
                            subset_interval = (start, end)
                        break

            if subset_interval:
                centroid = [math.ceil((subset_interval[0] + subset_interval[1]) / 2)]
            elif len(curr_group) > 2:
                first_overlap = max(
                    set(range(curr_group[0][0], curr_group[0][1] + 1))
                    & set(range(curr_group[1][0], curr_group[1][1] + 1))
                )
                last_overlap_set = set(
                    range(curr_group[-1][0], curr_group[-1][1] + 1)
                ) & set(range(curr_group[-2][0], curr_group[-2][1] + 1))

                if not last_overlap_set:
                    last_overlap = first_overlap  # Fallback if no overlap
                else:
                    last_overlap = min(last_overlap_set)

                step = 1 if first_overlap <= last_overlap else -1
                centroid = list(range(first_overlap, last_overlap + step, step))
            else:
                centroid = [
                    round(
                        sum([math.ceil((s + e) / 2) for s, e in curr_group])
                        / len(curr_group)
                    )
                ]

            grouped_intervals.append(
                curr_group if len(curr_group) > 1 else curr_group[0]
            )
            centroids.extend(centroid)

        return centroids

    def get_filter(self, top_k: int, file_md5: str, doc: List[Document]):
        """
        Create a filter for retrievers based on overlapping intervals.

        Args:
            top_k (int): Number of top intervals to consider.
            file_md5 (str): MD5 hash of the file to filter.
            doc (List[Document]): List of document objects.

        Returns:
            tuple: A tuple of containing dictionary filters for DocIndexer and Chroma retrievers.
        """
        overlaps = self.find_overlaps(doc)
        if len(overlaps) < 1:
            raise ValueError("No overlapping intervals found.")

        overlaps_k = overlaps[:top_k]
        logger.info("windows_at_2nd_retrieval: %s", overlaps_k)
        search_dict_docindexer = {"OR": []}
        search_dict_chroma = {"$or": []}

        for chunk_idx in overlaps_k:
            search_dict_docindexer["OR"].append(
                {
                    "large_chunks_idx_lower_bound": (
                        IndexerOperator.LTE,
                        chunk_idx,
                    ),
                    "large_chunks_idx_upper_bound": (
                        IndexerOperator.GTE,
                        chunk_idx,
                    ),
                    "source_md5": (IndexerOperator.EQ, file_md5),
                }
            )

            if len(overlaps_k) == 1:
                search_dict_chroma = {
                    "$and": [
                        {"large_chunks_idx_lower_bound": {"$lte": overlaps_k[0]}},
                        {"large_chunks_idx_upper_bound": {"$gte": overlaps_k[0]}},
                        {"source_md5": {"$eq": file_md5}},
                    ]
                }
            else:
                search_dict_chroma["$or"].append(
                    {
                        "$and": [
                            {"large_chunks_idx_lower_bound": {"$lte": chunk_idx}},
                            {"large_chunks_idx_upper_bound": {"$gte": chunk_idx}},
                            {"source_md5": {"$eq": file_md5}},
                        ]
                    }
                )

        return search_dict_docindexer, search_dict_chroma

    def get_relevant_doc_ids(self, docs: List[Document], query: str):
        """
        Get relevant document IDs given a query using an LLM.

        Args:
            docs (List[Document]): List of document objects to find relevant IDs in.
            query (str): The query string.

        Returns:
            list: A list of relevant document IDs.
        """
        snippets = "\n\n\n".join(
            [f"Context {idx}:\n{{{doc.page_content}}}" for idx, doc in enumerate(docs)]
        )
        id_chain = LLMChain(
            llm=self.llm,
            prompt=DOCS_SELECTION_PROMPT,
            output_key="IDs",
        )
        ids = id_chain.run({"query": query, "snippets": snippets})
        pattern = r"\[\s*\d+\s*(?:,\s*\d+\s*)*\]"
        match = re.search(pattern, ids)
        if match:
            return ast.literal_eval(match.group(0))
        else:
            return []

    def get_relevant_documents(
        self,
        query: str,
        num_query: int,
        *,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> List[Document]:
        """
        Perform multi-stage retrieval to get relevant documents.

        Args:
            query (str): The query string.
            num_query (int): Number of queries.
            run_manager (Optional[CallbackManagerForChainRun], optional): Callback manager for chain run.

        Returns:
            List[Document]: A list of relevant documents.
        """
        # ! First retrieval
        first_retriever = self.get_retriever(
            docs_chunks=self.docs_index_small.documents,
            emb_chunks=self.embedding_chunks_small,
            emb_filter=None,
            k=self.first_retrieval_k,
            weights=self.retriever_weights,
        )
        first = first_retriever.get_relevant_documents(
            query, callbacks=run_manager.get_child()
        )
        for doc in first:
            logger.info("----1st retrieval----: %s", doc)
        ids_clean = self.get_relevant_doc_ids(first, query)
        logger.info("relevant doc ids: %s", ids_clean)
        qa_chunks = {}  # key is file name, value is a list of relevant documents
        # res_chunks = []
        if ids_clean and isinstance(ids_clean, list):
            source_md5_dict = {}
            for ids_c in ids_clean:
                if ids_c < len(first):
                    if ids_c not in source_md5_dict:
                        source_md5_dict[first[ids_c].metadata["source_md5"]] = [
                            first[ids_c]
                        ]
                    # else:
                    #     source_md5_dict[first[ids_c].metadata["source_md5"]].append(
                    #         ids_clean[ids_c]
                    #     )
            if len(source_md5_dict) == 0:
                source_md5_dict[first[0].metadata["source_md5"]] = [first[0]]
            num_docs = len(source_md5_dict.keys())
            third_num_k = max(
                1,
                (
                    int(
                        (
                            configs.max_llm_context
                            / (configs.base_chunk_size * configs.chunk_scale)
                        )
                        // num_docs
                        * num_query
                    )
                ),
            )

            for source_md5, docs in source_md5_dict.items():
                logger.info(
                    "selected_docs_at_1st_retrieval: %s", docs[0].metadata["source"]
                )
                second_docs_chunks = self.docs_index_small.retrieve_metadata(
                    {
                        "source_md5": (IndexerOperator.EQ, source_md5),
                    }
                )
                second_retriever = self.get_retriever(
                    docs_chunks=second_docs_chunks,
                    emb_chunks=self.embedding_chunks_small,
                    emb_filter={"source_md5": source_md5},
                    k=self.second_retrieval_k,
                    weights=self.retriever_weights,
                )
                # ! Second retrieval
                second = second_retriever.get_relevant_documents(
                    query, callbacks=run_manager.get_child()
                )
                for doc in second:
                    logger.info("----2nd retrieval----: %s", doc)
                docs.extend(second)
                docindexer_filter, chroma_filter = self.get_filter(
                    self.num_windows, source_md5, docs
                )
                third_docs_chunks = self.docs_index_medium.retrieve_metadata(
                    docindexer_filter
                )
                third_retriever = self.get_retriever(
                    docs_chunks=third_docs_chunks,
                    emb_chunks=self.embedding_chunks_medium,
                    emb_filter=chroma_filter,
                    k=third_num_k,
                    weights=self.retriever_weights,
                )
                # ! Third retrieval
                third_temp = third_retriever.get_relevant_documents(
                    query, callbacks=run_manager.get_child()
                )
                third = third_temp[:third_num_k]
                # chunks = sorted(third, key=lambda x: x.metadata["medium_chunk_idx"])
                for doc in third:
                    logger.info(
                        "----3rd retrieval----page_content: %s", [doc.page_content]
                    )
                    mtdata = doc.metadata
                    mtdata["page_content"] = None
                    logger.info("----3rd retrieval----metadata: %s", mtdata)
                file_name = third[0].metadata["source"].split("/")[-1]
                if file_name not in qa_chunks:
                    qa_chunks[file_name] = third
                else:
                    qa_chunks[file_name].extend(third)

        return qa_chunks

    async def aget_relevant_documents(
        self,
        query: str,
        num_query: int,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """
        Asynchronous version of get_relevant_documents method.

        Args:
            query (str): The query string.
            num_query (int): Number of queries.
            run_manager (AsyncCallbackManagerForChainRun): Callback manager for asynchronous chain run.

        Returns:
            List[Document]: A list of relevant documents.
        """
        # ! First retrieval
        first_retriever = self.get_retriever(
            docs_chunks=self.docs_index_small.documents,
            emb_chunks=self.embedding_chunks_small,
            emb_filter=None,
            k=self.first_retrieval_k,
            weights=self.retriever_weights,
        )
        first = await first_retriever.aget_relevant_documents(
            query, callbacks=run_manager.get_child()
        )
        for doc in first:
            logger.info("----1st retrieval----: %s", doc)
        ids_clean = self.get_relevant_doc_ids(first, query)
        logger.info("relevant doc ids: %s", ids_clean)
        qa_chunks = {}  # key is file name, value is a list of relevant documents
        # res_chunks = []
        if ids_clean and isinstance(ids_clean, list):
            source_md5_dict = {}
            for ids_c in ids_clean:
                if ids_c < len(first):
                    if ids_c not in source_md5_dict:
                        source_md5_dict[first[ids_c].metadata["source_md5"]] = [
                            first[ids_c]
                        ]
                    # else:
                    #     source_md5_dict[first[ids_c].metadata["source_md5"]].append(
                    #         ids_clean[ids_c]
                    #     )
            if len(source_md5_dict) == 0:
                source_md5_dict[first[0].metadata["source_md5"]] = [first[0]]
            num_docs = len(source_md5_dict.keys())
            third_num_k = max(
                1,
                (
                    int(
                        (
                            configs.max_llm_context
                            / (configs.base_chunk_size * configs.chunk_scale)
                        )
                        // (num_docs * num_query)
                    )
                ),
            )

            for source_md5, docs in source_md5_dict.items():
                logger.info(
                    "selected_docs_at_1st_retrieval: %s", docs[0].metadata["source"]
                )
                second_docs_chunks = self.docs_index_small.retrieve_metadata(
                    {
                        "source_md5": (IndexerOperator.EQ, source_md5),
                    }
                )
                second_retriever = self.get_retriever(
                    docs_chunks=second_docs_chunks,
                    emb_chunks=self.embedding_chunks_small,
                    emb_filter={"source_md5": source_md5},
                    k=self.second_retrieval_k,
                    weights=self.retriever_weights,
                )
                # ! Second retrieval
                second = await second_retriever.aget_relevant_documents(
                    query, callbacks=run_manager.get_child()
                )
                for doc in second:
                    logger.info("----2nd retrieval----: %s", doc)
                docs.extend(second)
                docindexer_filter, chroma_filter = self.get_filter(
                    self.num_windows, source_md5, docs
                )
                third_docs_chunks = self.docs_index_medium.retrieve_metadata(
                    docindexer_filter
                )
                third_retriever = self.get_retriever(
                    docs_chunks=third_docs_chunks,
                    emb_chunks=self.embedding_chunks_medium,
                    emb_filter=chroma_filter,
                    k=third_num_k,
                    weights=self.retriever_weights,
                )
                # ! Third retrieval
                third_temp = await third_retriever.aget_relevant_documents(
                    query, callbacks=run_manager.get_child()
                )
                third = third_temp[:third_num_k]
                # chunks = sorted(third, key=lambda x: x.metadata["medium_chunk_idx"])
                for doc in third:
                    logger.info(
                        "----3rd retrieval----page_content: %s", [doc.page_content]
                    )
                    mtdata = doc.metadata
                    mtdata["page_content"] = None
                    logger.info("----3rd retrieval----metadata: %s", mtdata)
                file_name = third[0].metadata["source"].split("/")[-1]
                if file_name not in qa_chunks:
                    qa_chunks[file_name] = third
                else:
                    qa_chunks[file_name].extend(third)

        return qa_chunks
