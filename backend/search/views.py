import uuid

from django.shortcuts import render
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from rest_framework import permissions, status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from incarna_ai.utils import apply_rerank, hybrid_search
from minds.models import Block
from minds.serializers import BlockSerializer
from rag.models import File
from rag.serializers import FileSerializer
from vectordb import connect_cohere_client, connect_vectordb_client

# Create your views here.


class SearchView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        query = request.query_params.get("search", "")

        user_uuid = str(request.user.id)
        # print("SearchView", user_uuid)

        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(
                        value=user_uuid,
                    ),
                ),
            ]
        )

        res = hybrid_search(
            query_text=query,
            sparse_k=20,
            dense_k=30,
            top_k=30,
            filter=filter,
        )

        if res:
            response = []
            for r in res:
                response.append(
                    {
                        "uuid": standardize_uuid(r["uuid"]),
                        "page_content": r["page_content"][r["file_name_len"] :],
                        "type": r["type"],
                        "file_id": r["file_id"],
                        "file_name": r["file_name"],
                        "page_from": r.get("page_from", None),
                        "page_to": r.get("page_to", None),
                    }
                )

            return Response({"res": response}, status=status.HTTP_200_OK)
        else:
            return Response({"res": []}, status=status.HTTP_200_OK)


def standardize_uuid(s):
    # Split the string from the right on the hyphen, and allow up to 1 split
    parts = s.rsplit("-", 1)

    # Attempt to parse the result to check if it is a valid UUID
    try:
        uuid.UUID(parts[0])
        return parts[0]
    except ValueError:
        return s
