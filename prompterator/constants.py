import os
from typing import Any, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel
from pydantic.generics import GenericModel

DATA_STORE_DIR = os.path.expanduser(os.getenv("PROMPTERATOR_DATA_DIR", "~/prompterator-data"))

DataT = TypeVar("DataT")


class ConfigurableModelParameter(GenericModel, Generic[DataT]):
    default: DataT
    min: DataT
    max: DataT
    step: DataT


class ModelProperties(BaseModel):
    name: str
    is_chat_model: bool = False
    handles_batches_of_inputs: bool = False
    url: Optional[str] = None
    max_batch_size: Optional[int] = None
    max_retries: Optional[int] = 3

    configurable_params: Dict[str, ConfigurableModelParameter] = {}
    non_configurable_params: Dict[str, Any] = {}

    # By default, models are sorted by their position index, which is used to order them in the UI.
    # The 1e6 default value is used to ensure that models without a position index are sorted last.
    position_index: int = int(1e6)


class PrompteratorLLM:
    name: str
    properties: ModelProperties

    def format_prompt(self, user_prompt, system_prompt, **kwargs):
        raise NotImplementedError()

    def call(self, input, **kwargs):
        raise NotImplementedError()


MAX_TOKENS_KEY = "max_tokens"
TEMPERATURE_KEY = "temperature"

# Properties needed to set up Streamlit UI for configuring the given model params.
# These are the most often tweaked params; we leave any other params of a given model as
# non-configurable. For backwards-compatibility, do not change the default values; they are our
# best guess at reasonable values.
CONFIGURABLE_MODEL_PARAMETER_PROPERTIES = {
    MAX_TOKENS_KEY: ConfigurableModelParameter(default=256, min=1, max=2048, step=1),
    TEMPERATURE_KEY: ConfigurableModelParameter(default=0.3, min=0.01, max=1.0, step=0.01),
}

UNKNOWN_MODEL_NAME = "unknown model"


RESPONSE_CREATION_TIMESTAMP_KEY = "created"

TEXT_ORIG_COL = "text"
RAW_TEXT_GENERATED_COL = "raw_response"
TEXT_GENERATED_COL = "response"
COLS_TO_SHOW_KEY = "columns_to_show"
SYSTEM_PROMPT_TEMPLATE_COL = "system_prompt_template"
USER_PROMPT_TEMPLATE_COL = "user_prompt_template"
RESPONSE_DATA_COL = "response_data"
LABEL_COL = "human_label"
REUSED_PAST_LABEL_COL = "reused_label"
TIMESTAMP_COL = "timestamps"
MODEL_KEY = "model"
PROMPT_CREATOR_KEY = "creator"

DEFAULTS = {PROMPT_CREATOR_KEY: "unknown"}

DATAFILE_METADATA_KEY = "metadata"
DATAFILE_DATA_KEY = "data"
PROMPT_NAME_KEY = "prompt_name"
PROMPT_COMMENT_KEY = "prompt_comment"

DEFAULT_USER_PROMPT = "{{" + TEXT_ORIG_COL + "}}"

REQUIRED_CSV_COLS = [TEXT_ORIG_COL]
DATAFILE_ESSENTIAL_COLUMNS = {
    TEXT_ORIG_COL: str,
    TEXT_GENERATED_COL: str,
    RESPONSE_DATA_COL: object,
    LABEL_COL: bool,
}
# these are the columns that users won't be able to show or inject into their prompts
COLS_NOT_FOR_PROMPT_INTERPOLATION = [
    RAW_TEXT_GENERATED_COL,
    TEXT_GENERATED_COL,
    SYSTEM_PROMPT_TEMPLATE_COL,
    USER_PROMPT_TEMPLATE_COL,
    RESPONSE_DATA_COL,
    LABEL_COL,
    REUSED_PAST_LABEL_COL,
]
LABEL_GOOD = "good"
LABEL_BAD = "bad"
DUMMY_DATA_COLS = [TEXT_ORIG_COL, TEXT_GENERATED_COL, LABEL_COL]
LABEL_VALUE_COLOURS = {
    LABEL_GOOD: "#56E7AB",
    LABEL_BAD: "#FE8080",
}

TEXT_DIFF_COLOURS = {"add": "#56E7AB", "delete": "#FE8080"}

DEFAULT_ROW_NO = 0
DATA_POINT_TEXT_AREA_HEIGHT = 180
PROMPT_TEXT_AREA_HEIGHT = 300
PROMPT_PREVIEW_TEXT_AREA_HEIGHT = 200

DATAFILE_FILTER_ALL = "all"
