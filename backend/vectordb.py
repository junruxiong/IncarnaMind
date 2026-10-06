import os

import cohere
import requests
import voyageai
from django.conf import settings
from qdrant_client import AsyncQdrantClient, QdrantClient, models


def connect_vectordb_client():
    vectordb_client = QdrantClient(
        url=os.environ.get("VECTOR_DB_URL"),
        api_key=os.environ.get("VECTOR_DB_API_KEY"),
    )
    return vectordb_client


def connect_vectordb_aclient():
    vectordb_client = AsyncQdrantClient(
        url=os.environ.get("VECTOR_DB_URL"),
        api_key=os.environ.get("VECTOR_DB_API_KEY"),
    )
    return vectordb_client


def connect_voyage_client():
    voyage_client = voyageai.Client(
        api_key=os.environ.get("VOYAGE_API_KEY"),
    )
    return voyage_client


def connect_cohere_client():
    return cohere.Client(
        base_url=os.environ.get("CO_API_URL"),
        api_key=os.environ.get("CO_API_KEY"),
    )


def connect_cohere_aclient():
    return cohere.AsyncClient(
        base_url=os.environ.get("CO_API_URL"),
        api_key=os.environ.get("CO_API_KEY"),
    )


def create_collection(client: QdrantClient, collection_name: str):
    try:
        # client.delete_collection(collection_name=collection_name)
        client.create_collection(
            collection_name=collection_name,
            hnsw_config=models.HnswConfigDiff(
                payload_m=16,
                m=0,
                on_disk=True,
            ),
            optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
            vectors_config={
                "text-dense": models.VectorParams(
                    size=1024, distance=models.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(),
            },
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                ),
            ),
        )
        # indexing for multi-tenancy
        client.create_payload_index(
            collection_name=collection_name,
            field_name="tenant_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        # indexing for file_md5
        client.create_payload_index(
            collection_name=collection_name,
            field_name="file_md5",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        # indexing for idx
        client.create_payload_index(
            collection_name=collection_name,
            field_name="idx",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        # indexing for l idx
        client.create_payload_index(
            collection_name=collection_name,
            field_name="l_chunk_idx_from",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        # indexing for l idx
        client.create_payload_index(
            collection_name=collection_name,
            field_name="l_chunk_idx_to",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        # indexing for full-text search
        client.create_payload_index(
            collection_name=collection_name,
            field_name="page_content",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.MULTILINGUAL,
                min_token_len=2,
                max_token_len=15,
                lowercase=True,
            ),
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    pass
    # create_collection(connect_vectordb_client(), "incarnamind_s")
    # client = QdrantClient(
    #     url=os.environ.get("VECTOR_DB_URL"),
    #     api_key=os.environ.get("VECTOR_DB_API_KEY"),
    # )

    # snapshot_info = client.create_snapshot(collection_name="incarnamind_s")
    # url = os.environ.get("VECTOR_DB_URL")
    # snapshot_url = f"{url}/collections/incarnamind_s/snapshots/{snapshot_info.name}"

    # print(snapshot_url)

    # # snapshot_url = "https://adf4e6da-8b7e-48ca-b57c-c873bdae3146.us-east4-0.gcp.cloud.qdrant.io:6333/collections/test_collection/snapshots/incarnamind_s-3443213118157764-2024-06-17-22-03-00.snapshot"

    # snapshot_name = os.path.basename(snapshot_url)
    # local_snapshot_path = os.path.join("snapshots", snapshot_name)
    # response = requests.get(
    #     snapshot_url, headers={"api-key": os.environ.get("VECTOR_DB_API_KEY")}
    # )

    # with open(local_snapshot_path, "wb") as f:
    #     response.raise_for_status()
    #     f.write(response.content)

    # print(local_snapshot_path)

    # snapshot_path = (
    #     "snapshots/incarnamind_s-3443213118157764-2024-06-17-22-06-20.snapshot"
    # )

    # snapshot_name = os.path.basename(snapshot_path)
    # url = "https://f43e1060-253a-4988-940d-bdca04f7ff41.us-east4-0.gcp.cloud.qdrant.io:6333"
    # requests.post(
    #     f"{url}/collections/incarnamind_s/snapshots/upload?priority=snapshot",
    #     headers={
    #         "api-key": "iLPazDc_ZyamA3VkMAuu97dMxALjSthtpi1jiLivQQmJR569dh1-RQ",
    #     },
    #     files={"snapshot": (snapshot_name, open(snapshot_path, "rb"))},
    # )
