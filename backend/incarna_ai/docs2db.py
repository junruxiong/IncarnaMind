import asyncio
import os

# import torch
# from qdrant_client import AsyncQdrantClient, QdrantClient, models
# from qdrant_client.models import Batch
# from transformers import AutoModelForMaskedLM, AutoTokenizer

# from vectordb import connect_cohere_client

# #
# doc_model_id = "naver/efficient-splade-VI-BT-large-doc"
# doc_tokenizer = AutoTokenizer.from_pretrained(doc_model_id)
# doc_model = AutoModelForMaskedLM.from_pretrained(doc_model_id)
# #
# query_model_id = "naver/efficient-splade-VI-BT-large-query"
# query_tokenizer = AutoTokenizer.from_pretrained(query_model_id)
# query_model = AutoModelForMaskedLM.from_pretrained(query_model_id)

# from tqdm import tqdm

# cohere_client = connect_cohere_client()
# client = QdrantClient(
#     url="https://adf4e6da-8b7e-48ca-b57c-c873bdae3146.us-east4-0.gcp.cloud.qdrant.io:6333",
#     api_key="***REMOVED***",
# )


# # def compute_vector(text, doc_tokenizer, doc_model):
#     """
#     Computes a vector from logits and attention mask using ReLU, log, and max operations.

#     Args:
#     logits (torch.Tensor): The logits output from a model.
#     attention_mask (torch.Tensor): The attention mask corresponding to the input tokens.

#     Returns:
#     torch.Tensor: Computed vector.
#     """
#     tokens = doc_tokenizer(text, return_tensors="pt")
#     output = doc_model(**tokens)
#     logits, attention_mask = output.logits, tokens.attention_mask
#     relu_log = torch.log(1 + torch.relu(logits))
#     weighted_log = relu_log * attention_mask.unsqueeze(-1)
#     max_val, _ = torch.max(weighted_log, dim=1)
#     vec = max_val.squeeze()

#     return vec, tokens


# query_text = "Qdrant is the a vector database "
# query_vec, query_tokens = compute_vector(query_text, query_tokenizer, query_model)
# query_indices = query_vec.nonzero().numpy().flatten()
# query_values = query_vec.detach().numpy()[query_indices]

# results = client.search_batch(
#     collection_name="incarnamind_s",
#     requests=[
#         models.SearchRequest(
#             vector=models.NamedVector(
#                 name="text-dense",
#                 vector=cohere_client.embed(
#                     model="embed-multilingual-v3.0",
#                     input_type="search_query",
#                     texts=[query_text],
#                 ).embeddings[0],
#             ),
#             limit=5,
#         ),
#         models.SearchRequest(
#             vector=models.NamedSparseVector(
#                 name="text-sparse",
#                 vector=models.SparseVector(
#                     indices=query_indices.tolist(),
#                     values=query_values.tolist(),
#                 ),
#             ),
#             limit=5,
#         ),
#     ],
# )
# print(results)
