import logging
import time

import openai

logger = logging.getLogger(__name__)

from prompterator.constants import (  # isort:skip
    CONFIGURABLE_MODEL_PARAMETER_PROPERTIES,
    ModelProperties,
    PrompteratorLLM,
)


class ChatGPTMixin(PrompteratorLLM):
    def call(self, idx, input, **kwargs):
        model_params = kwargs["model_params"]
        try:
            response_data = openai.ChatCompletion.create(
                model=self.properties.name, messages=input, **model_params
            )
            response_text = [choice["message"]["content"] for choice in response_data["choices"]][
                0
            ]
            return {"response": response_text, "data": response_data, "idx": idx}
        except openai.error.RateLimitError as e:
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


class GPT35Turbo(ChatGPTMixin):
    name = "gpt-3.5-turbo"
    properties = ModelProperties(
        name="gpt-3.5-turbo",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
    )


class GPT4(ChatGPTMixin):
    name = "gpt-4"
    properties = ModelProperties(
        name="gpt-4",
        is_chat_model=True,
        handles_batches_of_inputs=False,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
    )


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


__all__ = ["GPT35Turbo", "GPT4", "MockGPT35Turbo"]
