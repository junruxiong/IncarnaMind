import asyncio
import os
import re
import time
import traceback
import uuid

import litellm
import markdown
import mistune
from anthropic import AnthropicVertex
from celery import group
from django.core.exceptions import ObjectDoesNotExist
from django.db import transaction
from django.db.models import F, Max, Q
from django.http import StreamingHttpResponse
from django.shortcuts import get_object_or_404, render
from langchain_text_splitters import CharacterTextSplitter
from litellm import completion
from markdown.extensions import fenced_code
from markdownify import markdownify as md
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action, permission_classes, renderer_classes
from rest_framework.exceptions import PermissionDenied
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

# from minds.ai_retriever import Clients, InstructorParams, RetrievalPipeline
from incarna_ai.clients import Clients, InstructorParams
from incarna_ai.utils import (
    execute_segment_query,
    extract_list_string,
    generate_snippets,
    hybrid_search,
    segment_query,
)
from minds.filters import last_filter
from minds.tasks import incarna_ai, rename_points, update_point
from rag.process_docs import tiktoken_len
from vectordb import (
    connect_cohere_aclient,
    connect_vectordb_aclient,
    connect_vectordb_client,
)

from .models import Block, Session
from .serializers import BlockSerializer, SessionSerializer

LOCATION = "us-central1"  # or "europe-west4"


def delete_point(block_id: str) -> None:
    vectordb_client = connect_vectordb_client()

    res = vectordb_client.scroll(
        collection_name="incarnamind_s",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="uuid",
                    match=models.MatchText(text=block_id),
                )
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
                            key="uuid",
                            match=models.MatchText(text=block_id),
                        )
                    ],
                )
            ),
        )

    print("delete_point++++", block_id)


def delete_points_mind(mind_id: str) -> None:
    vectordb_client = connect_vectordb_client()

    res = vectordb_client.scroll(
        collection_name="incarnamind_s",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="file_id",
                    match=models.MatchValue(value=mind_id),
                ),
            ],
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
                            match=models.MatchValue(value=mind_id),
                        ),
                    ],
                )
            ),
        )

    print("delete_point_mind++++", mind_id)


class IsOwnerOrReadOnly(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        # Read permissions are allowed to any request,
        # so we'll always allow GET, HEAD or OPTIONS requests.
        if request.method in permissions.SAFE_METHODS:
            return True

        # Write permissions are only allowed to the owner of the session.
        return obj.user == request.user


# Create your views here.
class SessionPagination(PageNumberPagination):
    page_size = 30
    page_size_query_param = "page_size"


# Create your views here.
class BlockPagination(PageNumberPagination):
    page_size = 999
    page_size_query_param = "page_size"


# You can only manipulate when you are authenticated
class SessionViewSet(viewsets.ModelViewSet):
    serializer_class = SessionSerializer
    pagination_class = SessionPagination
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrReadOnly]

    def get_queryset(self):
        user = self.request.user
        if user.is_authenticated:
            return Session.objects.filter(user=user).order_by("-updated_at")
        else:
            return Session.objects.none()

    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        pk = self.kwargs.get("pk")
        obj = get_object_or_404(queryset, pk=pk)
        self.check_object_permissions(self.request, obj)

        return obj

    def perform_create(self, serializer):
        # Assign the session to the logged-in user
        serializer.save(user=self.request.user)

    def update(self, request, *args, **kwargs):
        try:
            session = self.get_object()
            if request.user == session.user or request.user.is_superuser:
                # if the request data contain name, use rename_points task
                if "name" in request.data:
                    rename_points.delay(
                        str(session.id), request.data["name"], str(request.user.id)
                    )

                response = super().update(request, *args, **kwargs)
                return Response(
                    {"message": "Session updated successfully", "data": response.data},
                    status=status.HTTP_200_OK,
                )
            else:
                raise PermissionDenied(
                    "You do not have permission to edit this session."
                )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, *args, **kwargs):
        try:
            session = self.get_object()
            if request.user == session.user or request.user.is_superuser:
                delete_points_mind(str(session.id))
                super().destroy(request, *args, **kwargs)
                return Response(
                    {"message": "Session deleted successfully"},
                    status=status.HTTP_200_OK,
                )
            else:
                raise PermissionDenied(
                    "You do not have permission to delete this session."
                )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class BlockViewSet(viewsets.ModelViewSet):
    """
    A viewset for viewing, creating, and editing blocks within a session.
    """

    serializer_class = BlockSerializer
    pagination_class = BlockPagination
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        if user.is_authenticated:
            mind_id = self.kwargs.get("mind_id")
            return Block.objects.filter(mind_id=mind_id).order_by("order", "created_at")
        else:
            return Block.objects.none()

    def get_object(self):
        queryset = self.filter_queryset(self.get_queryset())
        pk = self.kwargs.get("pk")
        obj = get_object_or_404(queryset, pk=pk)
        self.check_object_permissions(self.request, obj)

        return obj

    def partial_update(self, request, *args, **kwargs):
        with transaction.atomic():
            try:
                block = self.get_object()

                if request.user == block.user or request.user.is_superuser:
                    if request.data.get("text", None):
                        new_text = md(request.data.get("text", None)).rstrip()
                        old_text = (
                            md(block.text).rstrip() if hasattr(block, "text") else None
                        )
                        # get the block id
                        block_id = str(block.id)
                        mind_id = str(self.kwargs.get("mind_id"))
                        mind_name = str(Session.objects.get(id=mind_id).name)
                        tenant_id = str(request.user.id)

                        if new_text != old_text:
                            update_point.delay(
                                new_text, block_id, mind_id, mind_name, tenant_id
                            )
                            # update the Session updated_at field to now
                            session = Session.objects.get(id=mind_id)
                            session.save(update_fields=["updated_at"])

                    response = super().partial_update(request, *args, **kwargs)
                    return Response(
                        {
                            "message": "Block updated successfully",
                            "data": response.data,
                        },
                        status=status.HTTP_200_OK,
                    )
                else:
                    raise PermissionDenied(
                        "You do not have permission to edit this block."
                    )
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=["post"], url_path="initial")
    def initialize_blocks(self, request, mind_id=None):
        """
        Custom action to create a text and a query block under a given session.
        """
        # Ensure the session exists
        mind_id = get_object_or_404(Session, pk=mind_id)
        text_uuid = uuid.uuid4()
        query_uuid = uuid.uuid4()

        text_block = Block.objects.create(
            client_id=text_uuid,
            id=text_uuid,
            metadata=None,
            order=0,
            query_to=None,
            reply_to=None,
            text="<p></p>",
            type=Block.MindType.TEXT,
            mind_id=mind_id,
            user=request.user,
        )

        query_block = Block.objects.create(
            client_id=query_uuid,
            id=query_uuid,
            metadata=None,
            order=1,
            query_to=None,
            reply_to=None,
            text="<p></p>",
            type=Block.MindType.QUERY,
            mind_id=mind_id,
            user=request.user,
        )

        # Construct response data or use serializers to represent the created blocks
        response_data = {
            "text_block_id": text_block.id,
            "query_block_id": query_block.id,
        }

        return Response(response_data, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=["post"], url_path="modify")
    def modify(self, request, *args, **kwargs):
        mind_id = self.kwargs.get("mind_id")
        data = request.data
        with transaction.atomic():
            # Perform bulk delete
            self._bulk_delete(data.get("delete", []), mind_id)
            # Perform bulk add
            self._bulk_add(data.get("add", []), mind_id)
            # Perform bulk reorder
            self._bulk_reorder(data.get("reorder", []), mind_id)
        return Response({"status": "success"}, status=status.HTTP_200_OK)

    @action(detail=False, methods=["post"], url_path="generate_output")
    def generate_output(self, request, *args, **kwargs):
        mind_id_str = self.kwargs.get("mind_id")
        client_id = request.data.get("client_id")
        input_text = request.data.get("text")
        is_standalone = request.data.get("is_standalone")

        input_query = md(input_text).rstrip()

        # if input_query is a empty string, return error
        if not input_query:
            return Response(
                {"error": "Please provide a valid input."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # get user id
        user = request.user
        user_uuid = str(user.id)
        is_retrieval = user.is_retrieval

        if user.credits < 1:
            return Response(
                {"error": "You don't have enough credits to make this request."},
                status=status.HTTP_402_PAYMENT_REQUIRED,
            )

        # Retrieve the Session instance using mind_id
        mind_id = get_object_or_404(Session, pk=mind_id_str)

        try:
            with transaction.atomic():
                try:
                    query_block = Block.objects.get(
                        client_id=client_id, mind_id=mind_id, type=Block.MindType.QUERY
                    )
                except Block.DoesNotExist:
                    return Response(
                        {"error": "Query block not found."},
                        status=status.HTTP_404_NOT_FOUND,
                    )

                history_blocks = Block.objects.filter(
                    mind_id=mind_id,
                    order__lt=query_block.order,
                    is_prompt=True,
                ).order_by("order", "created_at")

                bot_response_text, metadata = self._generate_bot_response(
                    history_blocks,
                    input_query,
                    user_uuid,
                    user.current_model,
                    is_standalone,
                    is_retrieval,
                )

                # Generate a new UUID for the output block to ensure it's unique
                output_uuid = uuid.uuid4()

                # Check for an existing output block that is a direct reply
                existing_output_block = Block.objects.filter(
                    reply_to=query_block, mind_id=mind_id, type=Block.MindType.OUTPUT
                ).first()

                if existing_output_block:
                    # Update existing block if found
                    existing_output_block.text = bot_response_text
                    # update metadata
                    existing_output_block.metadata = metadata
                    existing_output_block.model = user.current_model
                    existing_output_block.save(update_fields=["text", "metadata"])
                    output_block = existing_output_block
                else:
                    # Create a new block if no suitable output block exists
                    output_block = Block.objects.create(
                        client_id=output_uuid,  # Ensuring client_id and id are the same
                        id=output_uuid,  # Explicitly setting id to the same UUID as client_id
                        mind_id=mind_id,
                        type=Block.MindType.OUTPUT,
                        order=query_block.order,
                        reply_to=query_block,
                        text=bot_response_text,
                        user=request.user,
                        metadata=metadata,
                        model=user.current_model,
                    )

                query_block.query_to = output_block
                query_block.save(update_fields=["query_to"])

                block_id = str(output_block.id)
                mind_name = str(mind_id.name)
                tenant_id = str(user.id)
                update_point.delay(
                    md(bot_response_text).rstrip(),
                    block_id,
                    mind_id_str,
                    mind_name,
                    tenant_id,
                )

                # derease user.credits
                if (
                    user.current_model == "gpt4o"
                    or user.current_model == "claude_sonnet"
                    or user.current_model == "gemini_pro"
                ):
                    user.credits = F("credits") - 5

                elif user.current_model == "claude_opus":
                    user.credits = F("credits") - 6

                else:
                    user.credits = F("credits") - 1

                user.save(update_fields=["credits"])
                # update the Session updated_at field to now
                mind_id.save(update_fields=["updated_at"])

            serializer = self.get_serializer(output_block)

            # streaming_response = stream_response()

            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except TimeoutError:
            return Response({"error": "Time out, please try again later."}, status=408)

    def _generate_bot_response(
        self,
        history_blocks,
        input_query,
        user_uuid,
        model_type,
        is_standalone,
        is_retrieval,
    ):
        time_start = time.time()
        client = Clients()
        instructor_params = InstructorParams()

        if is_standalone:
            chat_history = ""
            messages = []
        else:
            chat_history, messages = add_history(history_blocks, 12000, 2000)

        # print("chat history:", chat_history)
        if is_retrieval:
            quries = run_segment_query(
                input_query, user_uuid, client, instructor_params, chat_history
            )
            # if quries:
            task_signatures = []
            num_quries = len(quries)
            if num_quries < 1 or not quries:
                quries = [input_query]

            for q in quries:
                print("sub queries+++++++++:", q)

            for query in quries:
                task = incarna_ai.s(input_query, query, user_uuid, num_quries)
                task_signatures.append(task)
            job = group(task_signatures)
            result = job.apply_async()
            results = result.get(timeout=60)
        else:
            results = None

        html, citations = repsonse_by_model(
            client,
            instructor_params,
            input_query,
            results,
            messages,
            model_type,
        )
        # else:
        #     html, citations = repsonse_by_model(
        #         client, instructor_params, input_query, None, messages, model_type
        #     )

        print("Time:")
        print(time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start)))

        return html, citations

    def _bulk_delete(self, delete_ids, mind_id):
        for block_id in delete_ids:
            delete_point(block_id)

        Block.objects.filter(client_id__in=delete_ids, mind_id=mind_id).delete()

    def _bulk_add(self, add_blocks, mind_id):
        mind_id = get_object_or_404(Session, pk=mind_id)
        for block_data in add_blocks:
            block = Block(
                client_id=block_data["client_id"],
                id=block_data["id"],
                metadata=block_data["metadata"],
                order=block_data["order"],
                query_to=block_data.get("query_to"),
                reply_to=block_data.get("reply_to"),
                text=block_data["text"],
                type=block_data["type"],
                mind_id=mind_id,
                user=self.request.user,  # Assuming the user is set to the request user
            )
            block.save()

    def _bulk_reorder(self, reorder_info, mind_id):
        for reorder in reorder_info:
            for client_id, order in reorder.items():
                Block.objects.filter(client_id=client_id, mind_id=mind_id).update(
                    order=order
                )


def run_segment_query(
    input_query,
    user_uuid,
    client: Clients,
    instructor_params: InstructorParams,
    chat_history,
):
    filter_0 = models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(
                    value=str(user_uuid),
                ),
            ),
            models.FieldCondition(
                key="type",
                match=models.MatchValue(
                    value="doc",
                ),
            ),
        ]
    )

    res_0 = hybrid_search(
        input_query,
        275,
        275,
        None,
        filter_0,
        rerank=False,
        with_payload=["file_name"],
    )

    # add file names:
    file_names = ""
    file_names_set = set()
    for r_0 in res_0:
        for r in r_0:
            file_name = r.payload["file_name"]
            file_names_set.add(file_name)
            if file_name not in file_names:
                file_names += f"{file_name}, "
    print("File names-------------:", file_names)

    # try:
    searches = segment_query(
        client.instrutor_client_gcp_anthropic,
        instructor_params.gcp_anthropic_haiku["provider"],
        instructor_params.gcp_anthropic_haiku["model_name"],
        chat_history,
        file_names,
        input_query,
    )
    quries = execute_segment_query(searches)

    # except Exception as e:
    #     print("segment_query error:", e)
    #     return [input_query]

    if not quries:
        return [input_query]

    return quries


def add_history(history_blocks, max_length_history=8000, max_length_history_2=2000):
    chat_history = ""
    messages = []
    llm_role_map = {"text": "user", "query": "user", "output": "assistant"}
    # llm_role_map = {"text": "system", "query": "user", "output": "assistant"}
    total_tokens_messages = 0
    total_tokens_chat_history = 0

    for b in history_blocks:
        history_text = md(b.text).rstrip()

        text_len = tiktoken_len(history_text)
        history_role = llm_role_map[b.type]

        if text_len == 0:
            continue

        # Check if adding this block exceeds the max_length for messages
        if total_tokens_messages + text_len > max_length_history:
            # If it does, we need to remove older entries from messages
            while total_tokens_messages + text_len > max_length_history and messages:
                # Remove the oldest message
                removed_message = messages.pop(0)
                removed_text = (
                    f"\n{removed_message['role']}:\n{removed_message['content']}\n"
                )
                removed_text_len = tiktoken_len(removed_message["content"])
                total_tokens_messages -= (
                    removed_text_len  # Decrement total_tokens_messages
                )

        # Check if adding this block exceeds the max_length for chat_history
        if total_tokens_chat_history + text_len > max_length_history_2:
            # If it does, we need to remove older entries from chat_history
            while (
                total_tokens_chat_history + text_len > max_length_history_2
                and chat_history
            ):
                # Find the position of the first message to remove
                first_message_end = chat_history.find("\n", 1)
                if first_message_end == -1:
                    break
                removed_text = chat_history[: first_message_end + 1]
                removed_text_len = tiktoken_len(removed_text.strip())
                chat_history = chat_history[first_message_end + 1 :]
                total_tokens_chat_history -= (
                    removed_text_len  # Decrement total_tokens_chat_history
                )

        # After ensuring we have enough space, add the new block
        messages.append({"role": history_role, "content": history_text})
        chat_history += f"\n{history_role:}\n```{history_text}\n```"
        total_tokens_messages += text_len  # Increment total_tokens_messages
        total_tokens_chat_history += text_len  # Increment total_tokens_chat_history

    return chat_history, messages


def repsonse_by_model(
    client: Clients,
    instructor_params,
    input_query,
    results,
    messages,
    model_type,
):
    instructor_params = InstructorParams()

    if results:
        snippets, citations = generate_snippets(results)

        if messages:
            RETRIEVAL_QA_SYS = f"""You are a helpful multilingual assistant. The below sources are retrieved in user's database.
            
            If you think the above chat history and below sources from the user database are relevant to the user input, please respond to the user based these context; otherwise, respond in your own knowledge to the user input. But DO NOT make up anything that inrelevant to user input and refuse to answer!
            
            Sources:
            {snippets}
            """
        else:
            RETRIEVAL_QA_SYS = f"""You are a helpful multilingual assistant. The below sources are retrieved in user's database.
   
            If you think the below sources from the user database are relevant to the user input, please respond to the user based these context; otherwise, respond in your own knowledge to the user input. But DO NOT make up anything that inrelevant to user input and refuse to answer!

            Sources:
            {snippets}
            """

        messages_rag = [
            {"role": "system", "content": RETRIEVAL_QA_SYS},
            {"role": "user", "content": input_query},
        ]

        messages.extend(messages_rag)

    else:
        messages.extend(
            [
                {"role": "user", "content": input_query},
            ]
        )
        snippets, citations = None, None

    messages_res = [
        {
            "role": "system",
            "content": "You are a helpful multilingual assistant designed by IncarnaMind.",
        },
    ]

    messages_res.extend(messages)

    if model_type == "gpt4o":
        response = client.client_azure_4o.chat.completions.create(
            model=instructor_params.azure_4o["model_name"],
            messages=messages_res,
            temperature=0.5,
        )
        answer = response.choices[0].message.content

    elif model_type == "claude_haiku":
        # litellm.set_verbose = True
        response = completion(
            model="vertex_ai/" + instructor_params.gcp_anthropic_haiku["model_name"],
            messages=messages_res,
            temperature=0.5,
            vertex_ai_project="clever-bounty-411614",
            vertex_ai_location="us-central1",
        )
        answer = response.choices[0].message.content

    elif model_type == "claude_sonnet":
        # litellm.set_verbose = True
        response = completion(
            model="vertex_ai/" + instructor_params.gcp_anthropic_sonnet["model_name"],
            messages=messages_res,
            temperature=0.5,
            vertex_ai_project="clever-bounty-411614",
            vertex_ai_location="us-east5",
        )
        answer = response.choices[0].message.content

    elif model_type == "claude_opus":
        # litellm.set_verbose = True
        response = completion(
            model="vertex_ai/" + instructor_params.gcp_anthropic_opus["model_name"],
            messages=messages_res,
            temperature=0.5,
            vertex_ai_project="clever-bounty-411614",
            vertex_ai_location="us-east5",
        )
        answer = response.choices[0].message.content

    elif model_type == "gemini_flash":
        response = completion(
            model="vertex_ai/" + instructor_params.gcp_gemini_15_flash["model_name"],
            messages=messages_res,
            temperature=0.5,
            vertex_ai_project="clever-bounty-411614",
            vertex_ai_location="us-central1",
        )
        answer = response.choices[0].message.content

    elif model_type == "gemini_pro":

        response = completion(
            model="vertex_ai/" + instructor_params.gcp_gemini_15_pro["model_name"],
            messages=messages,
            temperature=0.5,
            vertex_ai_project="clever-bounty-411614",
            vertex_ai_location="us-central1",
        )
        answer = response.choices[0].message.content

    else:
        response = client.client_azure_35.chat.completions.create(
            model=instructor_params.azure_35["model_name"],
            messages=messages,
            temperature=0.5,
        )
        answer = response.choices[0].message.content

        # response = completion(
        #     # api_key=os.environ.get("AZURE_OPENAI_API_KEY_4"),
        #     model=instructor_params.azure_4["model_name"],
        #     messages=messages,
        # )

    print(answer)

    answer = convert_latex_to_dollar(answer)

    markdown = mistune.create_markdown(plugins=["math"])
    html = markdown(answer)

    print("--------------------------------")
    print(html)
    html = html.replace("\\\n", "\\\\\n").replace(" \\ ", " \\\\\n")

    pattern = r'<span class="math">\\\((.*?)\\\)</span>'
    replacement = r"$\1$"
    html = re.sub(pattern, replacement, html)

    pattern = r"<pre><code>(.*?)</code></pre>"
    replacement = r"$\1$"
    html = re.sub(pattern, replacement, html)

    # # replace $$ to $
    html = html.replace("$$", "$")

    return html, citations


def stream_response():
    response = completion(
        model="gpt-4-turbo",
        messages=[{"content": "Hello, how are you?", "role": "user"}],
        stream=True,
    )

    def generate():
        for part in response:
            content = part.choices[0].delta.content or ""
            yield content

    return StreamingHttpResponse(generate(), content_type="text/plain")


def convert_latex_to_dollar(katex_str):

    # Function to handle display math conversion
    def replace_display_math(match):
        content = match.group(1)
        # Replace standalone newlines with spaces, but keep those inside LaTeX environments
        content = re.sub(r"(?<!\\)\n(?!\\)", " ", content)

        # replace double $ with single $
        content = f"${content.strip()}$\n"
        return content.replace("$$", "$")

    # Function to handle inline math conversion
    def replace_inline_math(match):
        content = match.group(1)
        # Replace standalone newlines with spaces, but keep those inside LaTeX environments
        content = re.sub(r"(?<!\\)\n(?!\\)", " ", content)
        return f"${content.strip()}$"

    # Convert display math
    display_math_pattern = re.compile(r"\\\[([\s\S]*?)\\\]")
    katex_str = display_math_pattern.sub(replace_display_math, katex_str)

    # Convert inline math
    inline_math_pattern = re.compile(r"\\\(([\s\S]*?)\\\)")
    katex_str = inline_math_pattern.sub(replace_inline_math, katex_str)

    # print("katex_str:", katex_str)

    return katex_str


# Replace the strings between $ signs with the modified ones
