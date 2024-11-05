import json
import logging
import os
import time

import openai
from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)

from prompterator.constants import (  # isort:skip
    CONFIGURABLE_MODEL_PARAMETER_PROPERTIES,
    ModelProperties,
    PrompteratorLLM,
    StructuredOutputImplementation as soi,
    StructuredOutputConfig,
)


class ChatGPTMixin(PrompteratorLLM):
    openai_variant: str = "openai"
    specific_model_name: str = None

    def __init__(self):
        if self.openai_variant == "openai":
            # We want to warn the user but not fail -- maybe they didn't provide an API
            # key because they don't intend to use OpenAI models.
            try:
                api_key = os.environ["OPENAI_API_KEY"]
            except KeyError:
                logger.warning(
                    "You don't have the 'OPENAI_API_KEY' environment variable set. "
                    "You won't be able to use OpenAI API models."
                )
                api_key = "<missing OpenAI API key>"

            self.client = OpenAI(api_key=api_key)
        elif self.openai_variant == "azure":
            use_default_credentials = (
                os.getenv("AZURE_OPENAI_USE_DEFAULT_CREDENTIALS", "False").lower() == "true"
            )
            token = None
            # We want to warn the user but not fail -- maybe they didn't provide an API
            # key or a base endpoint because they don't intend to use Azure OpenAI models.
            try:
                if use_default_credentials:
                    from azure.identity import DefaultAzureCredential

                    default_credential = DefaultAzureCredential()
                    token = default_credential.get_token(
                        "https://cognitiveservices.azure.com/.default"
                    )
                else:
                    api_key = os.environ["AZURE_OPENAI_API_KEY"]
            except KeyError:
                logger.warning(
                    "You don't have the 'AZURE_OPENAI_API_KEY' environment variable "
                    "set. You won't be able to use Azure OpenAI API models."
                )
                api_key = "<missing Azure OpenAI API key>"
            try:
                endpoint = os.environ["AZURE_OPENAI_API_BASE"]
            except KeyError:
                logger.warning(
                    "You don't have the 'AZURE_OPENAI_API_BASE' environment variable "
                    "set. You won't be able to use Azure OpenAI API models."
                )
                endpoint = "<missing Azure OpenAI API base endpoint>"

            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")

            if use_default_credentials:
                self.client = AzureOpenAI(
                    azure_ad_token=token.token,
                    azure_deployment=self.specific_model_name or self.name,
                    api_version=api_version,
                    azure_endpoint=endpoint,
                )
            else:
                self.client = AzureOpenAI(
                    api_version=api_version, azure_endpoint=endpoint, api_key=api_key
                )
        else:
            ValueError(
                f"Unsupported OpenAI variant '{self.openai_variant}'. Supported values "
                f"are 'openai' and 'azure'."
            )

        super().__init__()

    @staticmethod
    def get_function_calling_tooling_name(json_schema):
        return json_schema["title"]

    @staticmethod
    def build_function_calling_tooling(json_schema, function_name):
        """
        @param function_name: name for the openai tool
        @param json_schema: contains desired output schema in proper Json Schema format
        @return: list[tools] is (a single function in this case) callable by OpenAI model
            in function calling mode.
        """
        function = json_schema.copy()
        description = function.pop("description", function_name)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": description,
                    "parameters": function["parameters"],
                },
            }
        ]

        return tools

    @staticmethod
    def build_response_format(json_schema):
        """
        @param json_schema: contains desired output schema in proper Json Schema format
        @return: dict with desired response format directly usable with OpenAI API
        """
        schema = {"name": json_schema.pop("title"), "schema": json_schema, "strict": True}
        response_format = {"type": "json_schema", "json_schema": schema}

        return response_format

    @staticmethod
    def enrich_model_params_of_function_calling(structured_output_config, model_params):
        if structured_output_config.enabled:
            if structured_output_config.method == soi.FUNCTION_CALLING:
                schema = json.loads(structured_output_config.schema)
                function_name = ChatGPTMixin.get_function_calling_tooling_name(schema)

                model_params["tools"] = ChatGPTMixin.build_function_calling_tooling(
                    schema, function_name
                )
                model_params["tool_choice"] = {
                    "type": "function",
                    "function": {"name": function_name},
                }
            if structured_output_config.method == soi.RESPONSE_FORMAT:
                schema = json.loads(structured_output_config.schema)
                model_params["response_format"] = ChatGPTMixin.build_response_format(schema)
        return model_params

    @staticmethod
    def process_response(structured_output_config, response_data):
        if structured_output_config.enabled:
            if structured_output_config.method == soi.FUNCTION_CALLING:
                response_text = response_data.choices[0].message.tool_calls[0].function.arguments
            elif structured_output_config.method == soi.RESPONSE_FORMAT:
                response_text = response_data.choices[0].message.content
            else:
                response_text = response_data.choices[0].message.content
        else:
            response_text = response_data.choices[0].message.content
        return response_text

    def call(self, idx, input, **kwargs):
        structured_output_config: StructuredOutputConfig = kwargs["structured_output"]
        model_params = kwargs["model_params"]

        try:
            model_params = ChatGPTMixin.enrich_model_params_of_function_calling(
                structured_output_config, model_params
            )
        except json.JSONDecodeError as e:
            logger.error(
                "Error occurred while loading provided json schema. "
                f"Provided schema {structured_output_config.schema}"
                "%d. Returning an empty response.",
                idx,
                exc_info=e,
            )
            return {"idx": idx}

        try:
            response_data = self.client.chat.completions.create(
                model=self.specific_model_name or self.name, messages=input, **model_params
            )
        except openai.RateLimitError as e:
            logger.error(
                "OpenAI API rate limit reached when generating a response for text with index "
                "%d. Returning an empty response. To generate a proper response, please wait a"
                "minute and try again.",
                idx,
                exc_info=e,
            )
            return {"idx": idx}
        except Exception as e:
            logger.error(
                "An unexpected error occurred when generating a response for text with index "
                "%d. Returning an empty response.",
                idx,
                exc_info=e,
            )
            return {"idx": idx}

        try:
            response_text = ChatGPTMixin.process_response(structured_output_config, response_data)
            return {"response": response_text, "data": response_data, "idx": idx}
        except KeyError as e:
            logger.error(
                "Error occurred while processing response,"
                "response does not follow expected format"
                f"Response: {response_data}"
                "%d. Returning an empty response.",
                idx,
                exc_info=e,
            )
            return {"idx": idx}

    def format_prompt(self, system_prompt, user_prompt, **kwargs):
        messages = []
        if len(system_prompt.strip()) > 0:
            messages.append({"role": "system", "content": system_prompt})
        if len(user_prompt.strip()) > 0:
            messages.append({"role": "user", "content": user_prompt})
        return messages


class GPT4o(ChatGPTMixin):
    name = "gpt-4o"
    properties = ModelProperties(
        name="gpt-4o",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
        position_index=1,
        supported_structured_output_implementations=[
            soi.NONE,
            soi.FUNCTION_CALLING,
            soi.RESPONSE_FORMAT,
        ],
    )


class GPT4oAzure(ChatGPTMixin):
    name = "gpt-4o (Azure)"
    properties = ModelProperties(
        name="gpt-4o (Azure)",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
        position_index=6,
        supported_structured_output_implementations=[
            soi.NONE,
            soi.FUNCTION_CALLING,
            soi.RESPONSE_FORMAT,
        ],
    )
    openai_variant = "azure"
    specific_model_name = "gpt-4o"


class GPT4oMini(ChatGPTMixin):
    name = "gpt-4o-mini"
    properties = ModelProperties(
        name="gpt-4o-mini",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
        position_index=2,
        supported_structured_output_implementations=[
            soi.NONE,
            soi.FUNCTION_CALLING,
            soi.RESPONSE_FORMAT,
        ],
    )


class GPT4oMiniAzure(ChatGPTMixin):
    name = "gpt-4o-mini (Azure)"
    properties = ModelProperties(
        name="gpt-4o-mini (Azure)",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
        position_index=7,
        supported_structured_output_implementations=[
            soi.NONE,
            soi.FUNCTION_CALLING,
            soi.RESPONSE_FORMAT,
        ],
    )
    openai_variant = "azure"
    specific_model_name = "gpt-4o-mini"


class GPT35Turbo(ChatGPTMixin):
    name = "gpt-3.5-turbo"
    properties = ModelProperties(
        name="gpt-3.5-turbo",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
        position_index=3,
        supported_structured_output_implementations=[
            soi.NONE,
            soi.FUNCTION_CALLING,
        ],
    )


class GPT35TurboAzure(ChatGPTMixin):
    name = "gpt-3.5-turbo (Azure)"
    properties = ModelProperties(
        name="gpt-3.5-turbo (Azure)",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
        position_index=8,
        supported_structured_output_implementations=[
            soi.NONE,
            soi.FUNCTION_CALLING,
        ],
    )
    openai_variant = "azure"
    specific_model_name = "gpt-35-turbo"


class GPT4(ChatGPTMixin):
    name = "gpt-4"
    properties = ModelProperties(
        name="gpt-4",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
        position_index=4,
        supported_structured_output_implementations=[
            soi.NONE,
            soi.FUNCTION_CALLING,
        ],
    )


class GPT4Azure(ChatGPTMixin):
    name = "gpt-4 (Azure)"
    properties = ModelProperties(
        name="gpt-4 (Azure)",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
        position_index=9,
        supported_structured_output_implementations=[
            soi.NONE,
            soi.FUNCTION_CALLING,
        ],
    )
    openai_variant = "azure"
    specific_model_name = "gpt-4"


class GPT4Vision(ChatGPTMixin):
    name = "gpt-4-vision-preview"
    properties = ModelProperties(
        name="gpt-4-vision-preview",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
        position_index=5,
    )

    def format_prompt(self, system_prompt, user_prompt, **kwargs):
        messages = []
        if len(system_prompt.strip()) > 0:
            messages.append({"role": "system", "content": system_prompt})
        if len(user_prompt.strip()) > 0:
            data_row = kwargs["data_row"]
            base64_image = data_row["image"]
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": base64_image}},
                    ],
                }
            )

        return messages


class MockGPT35Turbo(ChatGPTMixin):
    name = "mock-gpt-3.5-turbo"
    properties = ModelProperties(
        name="mock-gpt-3.5-turbo",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
    )

    def call(self, idx, input, **kwargs):
        response_text = "response to: " + input[-1]["content"]
        response_data = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": response_text, "role": "assistant"},
                }
            ],
            "created": 1680529242,
            "id": "chatcmpl-71EiIPFuyj51vpyxfD1J1voDm5Akp",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 138,
                "prompt_tokens": 65,
                "total_tokens": 203,
            },
        }
        time.sleep(1)
        return {"response": response_text, "data": response_data, "idx": idx}


__all__ = [
    "GPT4o",
    "GPT4oAzure",
    "GPT4oMini",
    "GPT4oMiniAzure",
    "GPT35Turbo",
    "GPT4",
    "GPT35TurboAzure",
    "GPT4Azure",
    "GPT4Vision",
    "MockGPT35Turbo",
]
