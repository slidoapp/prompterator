# Supports Cohere Generate
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from prompterator.constants import (  # isort:skip
    CONFIGURABLE_MODEL_PARAMETER_PROPERTIES,
    MODEL_KEY,
    RESPONSE_CREATION_TIMESTAMP_KEY,
    ModelProperties,
    PrompteratorLLM,
)


class Cohere(PrompteratorLLM):
    name = "cohere"
    properties = ModelProperties(
        name="cohere",
        is_chat_model=False,
        handles_batches_of_inputs=True,
        max_batch_size=1,
        configurable_params=CONFIGURABLE_MODEL_PARAMETER_PROPERTIES.copy(),
    )

    def call(self, idx, input, **kwargs):
        try:
            import cohere
        except ImportError:
            raise Exception("cohere package not installed, please run `pip install cohere`")

        co = cohere.Client()
        response = co.generate(prompt=input, **kwargs["model_params"])

        response_data = response
        response_data[MODEL_KEY] = self.properties.name
        # unlike OpenAI models, Cohere doesn't return timestamp, so let's add it manually just for
        # the record
        response_data[RESPONSE_CREATION_TIMESTAMP_KEY] = int(datetime.now().timestamp())

        return [{"response": res.text, "data": response_data} for res in response_data]

    def format_prompt(self, system_prompt, user_prompt, **kwargs):
        return f"{system_prompt}"


__all__ = ["Cohere"]
