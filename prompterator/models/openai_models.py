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

    def call(self, idx, input, **kwargs):
        model_params = kwargs["model_params"]
        try:
            response_data = self.client.chat.completions.create(
                model=self.specific_model_name or self.name, messages=input, **model_params
            )
            response_text = response_data.choices[0].message.content

            return {"response": response_text, "data": response_data, "idx": idx}
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
    )


class GPT4oAzure(ChatGPTMixin):
    name = "gpt-4o (Azure)"
    properties = ModelProperties(
        name="gpt-4o (Azure)",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
        position_index=6,
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
    )


class GPT4oMiniAzure(ChatGPTMixin):
    name = "gpt-4o-mini (Azure)"
    properties = ModelProperties(
        name="gpt-4o-mini (Azure)",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
        position_index=7,
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
    )


class GPT35TurboAzure(ChatGPTMixin):
    name = "gpt-3.5-turbo (Azure)"
    properties = ModelProperties(
        name="gpt-3.5-turbo (Azure)",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
        position_index=8,
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
    )


class GPT4Azure(ChatGPTMixin):
    name = "gpt-4 (Azure)"
    properties = ModelProperties(
        name="gpt-4 (Azure)",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
        position_index=9,
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
