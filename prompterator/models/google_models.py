# Supports Google Vertex AI model
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


class TextBison(PrompteratorLLM):
    name = "text-bison"
    url_env_variable = "TEXT_BISON_URL"
    properties = ModelProperties(
        name="text-bison",
        is_chat_model=False,
        handles_batches_of_inputs=True,
        url=os.environ.get("TEXT_BISON_URL"),
        max_batch_size=1,
        configurable_params={
            "maxOutputTokens": CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[MAX_TOKENS_KEY].copy(
                update={"max": 2048, "default": 256}
            ),
            TEMPERATURE_KEY: CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[TEMPERATURE_KEY],
        },
        non_configurable_params={
            "topK": 40,
            "topP": 0.95,
        },
    )

    def call(self, idx, input, **kwargs):
        url = self.properties.url
        if url is None:
            raise Exception(f"{self.url_env_variable} env variable not set")

        auth_token = os.environ.get("GOOGLE_VERTEX_AUTH_TOKEN")
        if auth_token is None:
            raise Exception("GOOGLE_VERTEX_AUTH_TOKEN env variable not set")

        model_params = kwargs["model_params"]

        headers = {
            "Authorization": "Bearer " + auth_token,
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }

        payload = {
            "instances": [
                {
                    "prompt": instance,
                }
                for instance in input
            ],
            "parameters": model_params,
        }

        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        parsed_results = json.loads(response.text)
        response_data = parsed_results
        response_data[MODEL_KEY] = self.properties.name
        # unlike OpenAI models, Google Vertex doesn't return timestamp, so let's add it manually just for
        # the record
        response_data[RESPONSE_CREATION_TIMESTAMP_KEY] = int(datetime.now().timestamp())

        return [
            {"response": prediction["content"], "data": response_data}
            for prediction in parsed_results["predictions"]
        ]

    def format_prompt(self, system_prompt, user_prompt, **kwargs):
        return f"{system_prompt}"


__all__ = ["TextBison"]
