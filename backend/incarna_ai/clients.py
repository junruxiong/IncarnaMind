import asyncio
import math
import os
import time
from collections import Counter

import google.generativeai as genai
import instructor
import numpy as np
import vertexai
import weaviate.classes as wvc
from anthropic import Anthropic, AnthropicVertex, AsyncAnthropic, AsyncAnthropicVertex
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from vertexai.generative_models import GenerativeModel

os.environ.get("ANTHROPIC_API_KEY")
os.environ.get("OPENAI_API_KEY")
os.environ.get("GEMINI_API_KEY")
# os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class Clients:
    def __init__(self):
        self.client_openai = OpenAI()
        self.aclient_openai = AsyncOpenAI()

        self.client_azure_35 = self._create_azure_client(
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"
        )

        self.client_azure_4o = self._create_azure_client(
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"
        )
        self.aclient_azure_35 = self._create_azure_aclient(
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"
        )
        self.aclient_azure_4o = self._create_azure_aclient(
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"
        )

        self.client_gcp_gemini_15_pro = self._create_gcp_gemini_client(
            "gemini-1.0-pro-002"
        )
        self.client_gcp_gemini_15_flash = self._create_gcp_gemini_client(
            "gemini-1.5-flash-001"
        )

        self.client_anthropic = Anthropic()
        self.aclient_anthropic = AsyncAnthropic()

        self.instrutor_client_openai = instructor.from_openai(OpenAI())
        self.instrutor_aclient_openai = instructor.from_openai(AsyncOpenAI())
        self.instrutor_client_azure_35 = instructor.from_openai(self.client_azure_35)
        self.instrutor_aclient_azure_35 = instructor.from_openai(
            AsyncAzureOpenAI(
                **self._get_azure_config(
                    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"
                )
            )
        )
        self.instrutor_client_azure_4o = instructor.from_openai(self.client_azure_4o)
        self.instrutor_aclient_azure_4o = instructor.from_openai(
            AsyncAzureOpenAI(
                **self._get_azure_config(
                    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"
                )
            )
        )
        self.instrutor_client_anthropic = instructor.from_anthropic(Anthropic())

        self.instrutor_client_gcp_anthropic = instructor.from_anthropic(
            AnthropicVertex(region="us-central1", project_id="clever-bounty-411614")
        )
        self.instrutor_aclient_gcp_anthropic = instructor.from_anthropic(
            AsyncAnthropicVertex(
                region="us-central1", project_id="clever-bounty-411614"
            )
        )

        self.instrutor_aclient_anthropic = instructor.from_anthropic(AsyncAnthropic())

        self.instrutor_client_gcp_gemini_15_flash = instructor.from_gemini(
            # client=self.client_gcp_gemini_15_flash,
            client=genai.GenerativeModel(
                model_name="models/gemini-1.5-flash-latest",
            ),
            mode=instructor.Mode.GEMINI_JSON,
        )
        # self.instrutor_client_gcp_gemini_15_pro = instructor.from_gemini(
        #     client=genai.GenerativeModel(
        #         model_name="models/gemini-1.5-pro-latest",
        #     ),
        #     mode=instructor.Mode.GEMINI_JSON,
        # )

    def _create_azure_client(self, endpoint_env_var, azure_openai_api_key):
        return AzureOpenAI(
            **self._get_azure_config(endpoint_env_var, azure_openai_api_key)
        )

    def _create_azure_aclient(self, endpoint_env_var, azure_openai_api_key):
        return AsyncAzureOpenAI(
            **self._get_azure_config(endpoint_env_var, azure_openai_api_key)
        )

    def _create_gcp_gemini_client(self, model_name):
        vertexai.init(**self._get_gcp_config())
        return GenerativeModel(
            model_name=model_name,
        )

    def _get_azure_config(self, endpoint_env_var, azure_openai_api_key):
        return {
            "azure_endpoint": os.getenv(endpoint_env_var),
            "api_key": os.getenv(azure_openai_api_key),
            "api_version": "2024-02-01",
        }

    def _get_gcp_config(self):
        return {
            "project": "clever-bounty-411614",
            "location": "us-central1",
        }


class InstructorParams:
    def __init__(self):
        self.azure_35 = {
            "provider": "azure",
            "model_name": "gpt-35-turbo",
        }

        self.azure_4o = {
            "provider": "azure",
            "model_name": "gpt-4o",
        }
        self.openai_35 = {
            "provider": "openai",
            "model_name": "gpt-3.5-turbo",
        }

        self.gcp_anthropic_haiku = {
            "provider": "gcp_anthropic",
            "model_name": "claude-3-haiku@20240307",
        }
        self.gcp_anthropic_sonnet = {
            "provider": "gcp_anthropic",
            "model_name": "claude-3-5-sonnet@20240620",
        }
        self.gcp_anthropic_opus = {
            "provider": "gcp_anthropic",
            "model_name": "claude-3-opus@20240229",
        }

        self.anthropic_haiku = {
            "provider": "anthropic",
            "model_name": "claude-3-haiku-20240307",
        }
        self.anthropic_sonnet = {
            "provider": "anthropic",
            "model_name": "claude-3-sonnet-20240229",
        }
        self.anthropic_opus = {
            "provider": "anthropic",
            "model_name": "claude-3-opus-20240229",
        }

        self.gcp_gemini_15_pro = {
            "provider": "gcp_gemini",
            "model_name": "gemini-1.5-pro-preview-0514",
        }
        self.gcp_gemini_15_flash = {
            "provider": "gcp_gemini",
            "model_name": "gemini-1.5-flash-001",
        }
