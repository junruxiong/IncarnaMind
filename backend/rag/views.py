import os
import pickle
import tempfile
from io import BytesIO

from celery import group
from django.core.exceptions import PermissionDenied, ValidationError
from django.core.files.base import ContentFile
from django.db import transaction
from django.db.models import Q
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, render
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.pagination import PageNumberPagination
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from backend import settings

from .models import File
from .process_docs import count_tokens, delete_file_vectordb, load_blob, load_file
from .serializers import FileSerializer
from .tasks import file2vectordb

# from rag.process_docs import load_file


# Create your views here.


class FilesPagination(PageNumberPagination):
    page_size = 30
    page_size_query_param = "page_size"


class FileViewSet(viewsets.ModelViewSet):
    # upload, retrieve, update, destroy files
    serializer_class = FileSerializer
    pagination_class = FilesPagination
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def get_queryset(self):
        user = self.request.user
        if user.is_authenticated:
            # return File.objects.filter(user=user).order_by("updated_at")
            return File.objects.filter(user=user).order_by("filename")
        else:
            return File.objects.none()

    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        pk = self.kwargs.get("pk")
        obj = get_object_or_404(queryset, pk=pk)
        self.check_object_permissions(self.request, obj)

        return obj

    # get a file
    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance.user == request.user:
            serializer = self.get_serializer(instance)
            return Response(serializer.data)
        else:
            raise PermissionDenied()

    @action(detail=False, methods=["post"])
    def retrieve_by_md5(self, request, *args, **kwargs):
        md5 = request.data.get("md5")
        if not md5:
            return Response(
                {"message": "MD5 hash is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        file_instance = File.objects.filter(user=request.user, md5=md5).first()
        if not file_instance:
            return Response(
                {"message": "File not found"}, status=status.HTTP_404_NOT_FOUND
            )

        serializer = self.get_serializer(file_instance)
        return Response(serializer.data)

    # open a file
    @action(detail=True, methods=["get"])
    def open(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance.user == request.user:
            try:
                return FileResponse(instance.dir, as_attachment=True)
            except FileNotFoundError:
                raise Http404()
        else:
            raise PermissionDenied()

    # upload files
    def create(self, request, *args, **kwargs):
        files = request.FILES.getlist("files")
        md5_hashes = request.data.getlist("md5s")  # Extract MD5 hashes from the request
        file_names = request.data.getlist("filenames")
        file_types = request.data.getlist("types")
        if not files or not md5_hashes or not file_names:
            return Response(
                {"message": "No files or MD5 hashes provided"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        duplicated_md5 = []
        duplicated_filename = []
        uploaded_files = []

        overloaded_files = []
        other_failures = []
        return_metadata = []
        task_signatures = []

        for file_item, file_md5, file_name, file_type in zip(
            files, md5_hashes, file_names, file_types
        ):
            if File.objects.filter(
                Q(user=request.user, md5=file_md5)
                | Q(user=request.user, filename=file_name)
            ).exists():
                duplicated_md5.append(file_md5)
                duplicated_filename.append(file_name)

            else:
                file_item.name = file_md5 + os.path.splitext(file_item.name)[1]
                file_instance = File(
                    dir=file_item,
                    user=request.user,
                    md5=file_md5,
                    filename=file_name,
                    type=file_type,
                )

                file_instance.save()

                file_dir = file_item.name
                file_id = file_instance.id
                user_uuid = str(request.user.id)

                if "WEBSITE_HOSTNAME" in os.environ:
                    doc = load_blob(file_dir, file_type, user_uuid)
                else:
                    media_dir = os.path.join(
                        settings.MEDIA_ROOT, f"files/{user_uuid}/{file_dir}"
                    )
                    doc = load_file(media_dir, file_type)

                total_tokens = count_tokens(doc)

                user = request.user

                token_count = sum(
                    File.objects.filter(user=user).values_list("token_count", flat=True)
                )
                # token_count = user.token_count

                max_tokens = user.max_tokens

                if token_count + total_tokens <= max_tokens:
                    # print(
                    #     "Token count+++: ",
                    #     file_name,
                    #     token_count,
                    #     total_tokens,
                    #     max_tokens,
                    # )
                    task = file2vectordb.s(
                        file_dir, file_name, file_md5, file_id, file_type, user_uuid
                    )
                    task_signatures.append(task)

                    serializer = self.get_serializer(file_instance)
                    return_metadata.append(serializer.data)
                    uploaded_files.append(file_name)

                    file_instance.token_count = total_tokens
                    file_instance.save(update_fields=["token_count"])

                    user.token_count = token_count + total_tokens
                    user.save(update_fields=["token_count"])
                else:
                    overloaded_files.append(file_name)
                    file_instance.delete()

        job = group(task_signatures)
        result = job.apply_async()
        results = result.get(timeout=300)
        # consider a separate endpoint to poll for the result

        for res in results:
            if res["success"] == False:
                # remove the file from the response_data
                return_metadata = [
                    item for item in return_metadata if item["md5"] != res["file_md5"]
                ]
                user.token_count -= res["total_tokens"]
                user.save(update_fields=["token_count"])
                file_instance = File.objects.get(md5=res["file_md5"], user=request.user)
                file_instance.dir.delete(save=False)
                file_instance.delete()
                other_failures.append(res["file_name"])

        response_data = {
            "uploaded": return_metadata,
            "duplicated_md5": duplicated_md5,
            "duplicated_filename": duplicated_filename,
            "overloaded_files": overloaded_files,
            "other_failures": other_failures,
        }

        return Response(response_data, status=status.HTTP_201_CREATED)

    # ! also sync with weaviate db
    def partial_update(self, request, pk=None, *args, **kwargs):
        file_instance = get_object_or_404(File, pk=pk, user=request.user)
        serializer = self.serializer_class(
            file_instance, data=request.data, partial=True
        )

        if serializer.is_valid():
            filename = serializer.validated_data.get("filename")

            # Check for duplicate filename in the database, excluding the current instance
            if (
                filename
                and File.objects.filter(user=request.user)
                .exclude(pk=pk)
                .filter(filename=filename)
                .exists()
            ):
                return Response(
                    {"error": "Duplicate filename"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # ! update filename to vectordb
            serializer.save()

            # Add your logic here to sync with weaviate db if needed

            return Response(serializer.data, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @transaction.atomic
    def destroy(self, request, *args, **kwargs):
        file_instance = self.get_object()
        # Check if the file belongs to the user
        if file_instance.user != request.user:
            raise PermissionDenied("You do not have permission to delete this file.")
        # get the file_md5
        file_md5 = str(file_instance.md5)
        # get the user id
        user_uuid = request.user.id
        # delete the data from the vector db
        delete_file_vectordb(file_md5, user_uuid)
        print(f"successfully deleted {user_uuid} from vectordb")

        # delete the actual file from the server's file system
        file_instance.dir.delete(save=False)
        # Delete the file record from the database
        file_instance.delete()

        token_count = file_instance.token_count
        request.user.token_count -= token_count
        request.user.save(update_fields=["token_count"])

        # return response to the client with string "id + delete successful"
        return Response(
            {"message": "File deleted successfully"}, status=status.HTTP_200_OK
        )
