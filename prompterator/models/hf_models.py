# Supports models deployed via Sagemaker / DJI (https://docs.djl.ai/)
import json
import logging
import os
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

from prompterator.constants import (  # isort:skip
    CONFIGURABLE_MODEL_PARAMETER_PROPERTIES,
    MAX_TOKENS_KEY,
    MODEL_KEY,
    RESPONSE_CREATION_TIMESTAMP_KEY,
    TEMPERATURE_KEY,
    ModelProperties,
    PrompteratorLLM,
)


class HFModelsMixin(PrompteratorLLM):
    url_env_variable = ""

    def call(self, idx, input, **kwargs):
        url = self.properties.url
        if url is None:
            raise Exception(f"{self.url_env_variable} env variable not set")

        model_params = kwargs["model_params"]
        headers = {"Content-Type": "application/javascript"}
        payload = {"prompts": input, "parameters": model_params}
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        parsed_results = json.loads(response.text)
        response_data = parsed_results["parameters"]
        response_data[MODEL_KEY] = self.properties.name
        # unlike OpenAI models, StableLM doesn't return timestamp, so let's add it manually just for
        # the record
        response_data[RESPONSE_CREATION_TIMESTAMP_KEY] = int(datetime.now().timestamp())

        return [
            {"response": generated_text, "data": response_data}
            for generated_text in parsed_results["outputs"]
        ]


class StableLM7BSTFv7Epoch3(HFModelsMixin):
    name = "stablelm-7b-sft-v7-epoch-3"
    url_env_variable = "STABLELM_API_URL"
    properties = ModelProperties(
        name="stablelm-7b-sft-v7-epoch-3",
        is_chat_model=False,
        handles_batches_of_inputs=True,
        url=os.environ.get("STABLELM_API_URL"),
        max_batch_size=1,
        configurable_params={
            "max_new_tokens": CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[MAX_TOKENS_KEY].copy(
                update={"max": 4096, "default": 150}
            ),
            TEMPERATURE_KEY: CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[TEMPERATURE_KEY],
        },
        non_configurable_params={
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "do_sample": True,
            "top_p": 0.90,
            "top_k": 50,
        },
    )

    def format_prompt(self, system_prompt, user_prompt, **kwargs):
        return f"<|prompter|>{system_prompt}<|endoftext|>\n\n\n<|assistant|>"


class MPT7BInstruct(HFModelsMixin):
    name = "mpt-7b-instruct"
    url_env_variable = "MPT7BINSTRUCT_API_URL"
    properties = ModelProperties(
        name="mpt-7b-instruct",
        is_chat_model=False,
        handles_batches_of_inputs=True,
        url=os.environ.get("MPT7BINSTRUCT_API_URL"),
        max_batch_size=1,
        configurable_params={
            "max_new_tokens": CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[MAX_TOKENS_KEY].copy(
                update={"default": 150}
            ),
            TEMPERATURE_KEY: CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[TEMPERATURE_KEY],
        },
        non_configurable_params={
            "early_stopping": True,
            "no_repeat_ngram_size": 4,
            "do_sample": True,
            "top_p": 0.85,
            "top_k": 50,
        },
    )

    def format_prompt(self, system_prompt, user_prompt, **kwargs):
        return f"{system_prompt}\n\n\n"


class GPTNeoXTChatBase20B8bit(HFModelsMixin):
    name = "gpt-neoxt-chat-base-20b-8bit"
    url_env_variable = "GPT_NEOXT_API_URL"
    properties = ModelProperties(
        name="gpt-neoxt-chat-base-20b-8bit",
        is_chat_model=False,  # this one IS a chat model but we'll use it as an instruction one
        handles_batches_of_inputs=True,
        url=os.environ.get("GPT_NEOXT_API_URL"),
        max_batch_size=1,
        configurable_params={
            "max_new_tokens": CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[MAX_TOKENS_KEY].copy(
                update={"default": 150}
            ),
            TEMPERATURE_KEY: CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[TEMPERATURE_KEY],
        },
        non_configurable_params={
            "early_stopping": True,
            "no_repeat_ngram_size": 4,
            "do_sample": True,
            "top_p": 0.85,
            "top_k": 50,
            # this prevents the model from generating multi-turn conversations; we want it to
            # generate just one <bot> turn:
            "stop_sequences": ["<bot>:", "<human>:"],
        },
    )

    def format_prompt(self, system_prompt, user_prompt, **kwargs):
        # We give users 2 options here:
        # - write the full prompt including the <human>:...<bot>:... structure
        # - write some text without this structure, which we'll treat as the <human> part of
        #   the prompt
        return (
            f"<human>: {system_prompt}\n<bot>: "
            if not all(tok in system_prompt for tok in ["<human>", "<bot>"])
            else system_prompt
        )


__all__ = ["StableLM7BSTFv7Epoch3", "MPT7BInstruct", "GPTNeoXTChatBase20B8bit"]
