# Supports AWS Bedrock models
import json
import logging
from datetime import datetime

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


class TitanTg1Large(PrompteratorLLM):
    name = "titan-tg1-large"
    properties = ModelProperties(
        name="titan-tg1-large",
        is_chat_model=False,
        handles_batches_of_inputs=True,
        max_batch_size=1,
        configurable_params={
            "maxTokenCount": CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[MAX_TOKENS_KEY].copy(
                update={"max": 2048, "default": 256}
            ),
            TEMPERATURE_KEY: CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[TEMPERATURE_KEY],
        },
        non_configurable_params={
            "topP": 0.9,
        },
    )

    def call(self, idx, input, **kwargs):
        import boto3.session

        # to be thread safe https://medium.com/@life-is-short-so-enjoy-it/aws-boto3-misunderstanding-about-thread-safe-a7261d7391fd
        session = boto3.session.Session()
        bedrock = session.client("bedrock")

        request = {
            "inputText": input,
            "textGenerationConfig": {**kwargs["model_params"]},
        }

        res = bedrock.invoke_model(
            modelId=f"amazon.{self.properties.name}",
            contentType="application/json",
            accept="*/*",
            body=json.dumps(request),
        )

        response_data = json.loads(res["body"].read())

        response_data[MODEL_KEY] = self.properties.name
        # unlike OpenAI models, AWS Bedrock doesn't return timestamp, so let's add it manually just for
        # the record
        response_data[RESPONSE_CREATION_TIMESTAMP_KEY] = int(datetime.now().timestamp())

        return [
            {"response": res["outputText"], "data": response_data}
            for res in response_data["results"]
        ]

    def format_prompt(self, system_prompt, user_prompt, **kwargs):
        return f"{system_prompt}"


class ClaudeMixin(PrompteratorLLM):
    def call(self, idx, input, **kwargs):
        import boto3.session

        # to be thread safe https://medium.com/@life-is-short-so-enjoy-it/aws-boto3-misunderstanding-about-thread-safe-a7261d7391fd
        session = boto3.session.Session()
        bedrock = session.client("bedrock")

        request = {
            "prompt": input,
            **kwargs["model_params"],
        }

        res = bedrock.invoke_model(
            modelId=f"anthropic.{self.properties.name}",
            contentType="application/json",
            accept="*/*",
            body=json.dumps(request),
        )

        response_data = json.loads(res["body"].read())

        response_data[MODEL_KEY] = self.properties.name
        # unlike OpenAI models, AWS Bedrock doesn't return timestamp, so let's add it manually just for
        # the record
        response_data[RESPONSE_CREATION_TIMESTAMP_KEY] = int(datetime.now().timestamp())

        return [
            {"response": res["outputText"], "data": response_data}
            for res in response_data["results"]
        ]

    def format_prompt(self, system_prompt, user_prompt, **kwargs):
        return f"Human: {system_prompt}\n\nAssistant:"


class ClaudeInstantV1(ClaudeMixin):
    name = "claude-instant-v1"
    properties = ModelProperties(
        name="claude-instant-v1",
        is_chat_model=False,
        handles_batches_of_inputs=True,
        max_batch_size=1,
        configurable_params={
            "max_tokens_to_sample": CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[MAX_TOKENS_KEY].copy(
                update={"max": 2048, "default": 256}
            ),
            TEMPERATURE_KEY: CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[TEMPERATURE_KEY],
        },
        non_configurable_params={
            "top_k": 250,
            "top_p": 0.999,
            "stop_sequences": ["\n\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31",
        },
    )


class ClaudeV1(ClaudeMixin):
    name = "claude-v1"
    properties = ModelProperties(
        name="claude-v1",
        is_chat_model=False,
        handles_batches_of_inputs=True,
        max_batch_size=1,
        configurable_params={
            "max_tokens_to_sample": CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[MAX_TOKENS_KEY].copy(
                update={"max": 2048, "default": 256}
            ),
            TEMPERATURE_KEY: CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[TEMPERATURE_KEY],
        },
        non_configurable_params={
            "top_k": 250,
            "top_p": 0.999,
            "stop_sequences": ["\n\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31",
        },
    )


class ClaudeV2(ClaudeMixin):
    name = "claude-v2"
    properties = ModelProperties(
        name="claude-v2",
        is_chat_model=False,
        handles_batches_of_inputs=True,
        max_batch_size=1,
        configurable_params={
            "max_tokens_to_sample": CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[MAX_TOKENS_KEY].copy(
                update={"max": 2048, "default": 256}
            ),
            TEMPERATURE_KEY: CONFIGURABLE_MODEL_PARAMETER_PROPERTIES[TEMPERATURE_KEY],
        },
        non_configurable_params={
            "top_k": 250,
            "top_p": 0.999,
            "stop_sequences": ["\n\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31",
        },
    )


__all__ = ["TitanTg1Large", "ClaudeInstantV1", "ClaudeV1", "ClaudeV2"]
