import asyncio
import os
import tempfile
import uuid
from typing import Iterable, List, Literal, Optional

import requests
from azure.storage.blob import BlobClient
from celery import shared_task
from django.db import transaction
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.models import Batch

from backend import settings
from incarna_ai.utils import process_sparse_batch, upload2vectordb
from rag.models import File
from rag.process_docs import (
    add_window,
    delete_file_vectordb,
    load_blob,
    load_file,
    merge_chunks,
    process_metadata,
    split_doc,
    uploadfile,
)
from users.models import UserAccount


@shared_task()
def file2vectordb(file_dir, file_name, file_md5, file_id, file_type, user_uuid):
    try:
        # with transaction.atomic():

        user = UserAccount.objects.get(id=user_uuid)

        # get the base media directory
        if "WEBSITE_HOSTNAME" in os.environ:
            doc = load_blob(file_dir, file_type, user_uuid)
        else:
            media_dir = os.path.join(
                settings.MEDIA_ROOT, f"files/{user_uuid}/{file_dir}"
            )
            doc = load_file(media_dir, file_type)

        chunk_split_small, total_tokens = split_doc(
            doc=doc,
            chunk_size=400,
            chunk_overlap=200,
            chunk_idx_name="idx",
        )

        add_window(
            doc=chunk_split_small,
            window_steps=1,
            window_size=3,
            window_idx_name="l_chunk_idx",
        )

        s_texts, s_payload = process_metadata(
            chunk_split_small,
            file_name,
            file_md5,
            file_id,
            str(user_uuid),
        )

        # upload to qdrant
        s_sparse_vector = process_sparse_batch(s_texts)
        asyncio.run(
            upload2vectordb(
                collection_name="incarnamind_s",
                payloads=s_payload,
                sparse_vectors=s_sparse_vector,
                texts=s_texts,
            )
        )

        return {
            "success": True,
            "file_name": file_name,
            "total_tokens": total_tokens,
            "file_md5": file_md5,
        }
    # f"successfully uploaded: {file_name}-{file_md5}"

    except Exception as e:
        # with transaction.atomic():
        file_instance = File.objects.get(md5=file_md5, user=user)
        #     total_tokens = file_instance.token_count
        #     user.token_count -= total_tokens
        #     file_instance.dir.delete(save=False)
        #     file_instance.delete()
        # return str(e)

        return {
            "success": False,
            "file_name": file_name,
            "total_tokens": file_instance.token_count,
            "file_md5": file_md5,
            # "error": str(e),
        }
