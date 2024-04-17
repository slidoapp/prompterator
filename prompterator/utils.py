import ast
import concurrent.futures
import itertools
import json
import logging
import os
import re
import socket
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from functools import partial
from typing import Any

import jinja2
import openai
import pandas as pd
import requests
import streamlit as st

import prompterator.constants as c
from prompterator.constants import ModelProperties, PrompteratorLLM

logger = logging.getLogger(__name__)


def get_text_orig(row):
    return row[c.TEXT_ORIG_COL]


def get_text_generated(row):
    return row[c.TEXT_GENERATED_COL]


def load_datafile(file_name):
    with open(file_name, "r") as f:
        contents = json.load(f)
        data = pd.DataFrame.from_dict(contents[c.DATAFILE_DATA_KEY], orient="index")
        data.index = data.index.astype(int)
        return data, contents[c.DATAFILE_METADATA_KEY]


def ensure_legacy_datafile_has_all_columns(df):
    if c.RAW_TEXT_GENERATED_COL not in df.columns:
        df.insert(
            df.columns.get_loc(c.TEXT_GENERATED_COL),
            c.RAW_TEXT_GENERATED_COL,
            df[c.TEXT_GENERATED_COL],
        )

    return df


def load_dataframe(file):
    df = pd.read_csv(file, index_col=0)
    df[c.TEXT_GENERATED_COL] = df[c.TEXT_GENERATED_COL].apply(lambda val: eval(val)[0])
    return df


def categorical_conditional_highlight(row, cond_column_name, palette):
    return [
        ("background-color:" + palette.get(row.loc[cond_column_name]))
        if row.loc[cond_column_name] in palette
        else ""
    ] * len(row)


def generate_responses(
    model_properties: ModelProperties,
    model_instance: PrompteratorLLM,
    inputs,
    model_params,
    progress_bar,
):
    model_params = {
        **model_properties.configurable_params,
        **model_properties.non_configurable_params,
        **model_params,
    }
    if model_properties.handles_batches_of_inputs:
        results = generate_responses_using_batching(
            model_properties, model_instance, inputs, model_params, progress_bar
        )
    else:
        results = generate_responses_using_parallelism(
            model_properties, model_instance, inputs, model_params, progress_bar
        )

    return results


def split_inputs_into_batches(inputs, batch_size):
    return [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]


def update_generation_progress_bar(bar, current, total):
    bar.progress(
        current / total,
        text=f"generating texts ({current}/{total})",
    )


def generate_responses_using_batching(
    model_properties: ModelProperties,
    model_instance: PrompteratorLLM,
    inputs,
    model_params,
    progress_bar,
):
    inputs = list(inputs.values())
    if model_properties.max_batch_size is not None:
        input_batches = split_inputs_into_batches(inputs, model_properties.max_batch_size)
    else:
        input_batches = [inputs]

    result_batches = []
    for batch in input_batches:
        n_attempts = 0
        while n_attempts < model_properties.max_retries:
            try:
                result_batch = model_instance.call(n_attempts, batch, model_params=model_params)
                result_batches.append(result_batch)
                break
            except Exception as e:
                n_attempts += 1
                logger.error(
                    f"Attempt %d/%d failed",
                    n_attempts,
                    model_properties.max_retries,
                    exc_info=e,
                )

        # if all attempts failed, return empty results for the batch
        if n_attempts == model_properties.max_retries:
            result_batches.append([{}] * len(batch))

        n_results = sum(len(r) for r in result_batches)
        update_generation_progress_bar(progress_bar, n_results, len(inputs))

    results = list(itertools.chain.from_iterable(result_batches))
    return {i: res for i, res in enumerate(results)}


def generate_responses_using_parallelism(
    model_properties: ModelProperties,
    model_instance: PrompteratorLLM,
    inputs,
    model_params,
    progress_bar,
):
    # use concurrent threads to speed up generation; adapted from this code example:
    # https://discuss.streamlit.io/t/streamlit-session-state-with-multiprocesssing/29230/7
    results = {}
    processed_jobs = []

    generate_func = partial(
        model_instance.call,
        model_properties=model_properties,
        model_params=model_params,
    )

    with ThreadPoolExecutor(max_workers=len(inputs)) as executor:
        # start all jobs
        for i, input in inputs.items():
            pj = executor.submit(generate_func, idx=i, input=input)
            processed_jobs.append(pj)

        # retrieve results and show progress
        for future in concurrent.futures.as_completed(processed_jobs):
            try:
                res = future.result()
                results[res["idx"]] = res
                update_generation_progress_bar(progress_bar, len(results), len(inputs))

            except BrokenProcessPool as ex:
                logger.error(f"Generating text failed", exc_info=ex)

    return results


def get_correctness_summary(df):
    return "{good}/{all}".format(
        good=len(df.query(f"{c.LABEL_COL} == '{c.LABEL_GOOD}'")), all=len(df)
    )


def save_labelled_data():
    ts = datetime.now().strftime("%-m-%-d-%y_%-H:%M:%S")
    file_name = f"{c.DATA_STORE_DIR}/{ts}.json"
    model = st.session_state[c.MODEL_KEY]
    configurable_params = {name: st.session_state[name] for name in model.configurable_params}
    contents = {
        c.DATAFILE_METADATA_KEY: {
            c.SYSTEM_PROMPT_TEMPLATE_COL: st.session_state.system_prompt,
            c.USER_PROMPT_TEMPLATE_COL: st.session_state.user_prompt,
            c.TIMESTAMP_COL: ts,
            c.PROMPT_NAME_KEY: st.session_state[c.PROMPT_NAME_KEY],
            c.PROMPT_COMMENT_KEY: st.session_state[c.PROMPT_COMMENT_KEY],
            c.MODEL_KEY: model.name,
            c.PROMPT_CREATOR_KEY: get_prompt_creator(),
            c.COLS_TO_SHOW_KEY: st.session_state[c.COLS_TO_SHOW_KEY],
            **configurable_params,
        },
        c.DATAFILE_DATA_KEY: json.loads(st.session_state.df.to_json(orient="index")),
    }
    with open(file_name, "w") as f:
        json.dump(contents, f)


def ensure_datafiles_directory_exists():
    os.makedirs(c.DATA_STORE_DIR, exist_ok=True)


def get_creation_timestamp(path):
    # works around the strange behaviour on Mac where os.path.getctime returns time of last
    # modification instead of creation time (
    # https://stackoverflow.com/questions/946967/get-file-creation-time-with-python-on-mac)
    return os.stat(path).st_birthtime


def get_prompt_creator():
    return socket.gethostname()


def get_datafile_filtering_attribute_from_datafile(df, attr_name):
    """
    For now, we simply retrieve the entries from the metadata. If the structure of the data
    changes, this function will become more involved.
    """
    return df[c.DATAFILE_METADATA_KEY].get(attr_name, "unknown")


def create_model_input(
    model_properties: ModelProperties,
    model_instance: PrompteratorLLM,
    user_prompt_template,
    system_prompt_template,
    data_row,
):
    system_prompt = jinja_env().from_string(system_prompt_template).render(**data_row.to_dict())
    user_prompt = jinja_env().from_string(user_prompt_template).render(**data_row.to_dict())

    return model_instance.format_prompt(system_prompt, user_prompt, data_row=data_row)


@st.cache_resource
def jinja_env() -> jinja2.Environment:
    def fromjson(text: str) -> Any:
        try:
            return json.loads(text)
        except json.decoder.JSONDecodeError as e:
            raise ValueError(
                f"The string you passed into `fromjson` is not a valid JSON string: " f"`{text}`"
            ) from e

    def fromAstString(text: str) -> Any:
        try:
            return ast.literal_eval(text)
        except Exception as e:
            raise ValueError(
                f"The string you passed into `fromAstString` is not a valid "
                f"input: `{text}`. Generally, try passing a valid string "
                f"representation of a "
                f"Python dictionary/list/set/tuple or other simple types. For more "
                f"details, refer to "
                f"[`ast.literal_eval`](https://docs.python.org/3/library/ast.html#ast.literal_eval)."
            ) from e

    env = jinja2.Environment()
    env.globals["fromjson"] = fromjson
    env.globals["fromAstString"] = fromAstString
    return env


def get_datafile_filtering_options(attr_name):
    """
    Defaults for the individual filtering parameters:
    - creator: current user if the user has some datafiles; otherwise, 'all'
    - model: 'all'
    """
    options_from_datafiles = [
        {"name": name, "frequency": freq}
        for name, freq in Counter(
            [
                get_datafile_filtering_attribute_from_datafile(df, attr_name)
                for df in st.session_state.datafiles.values()
            ]
        ).items()
    ]
    if attr_name == "creator":
        current_creator = get_prompt_creator()
        default_option = (
            c.DATAFILE_FILTER_ALL
            if current_creator not in options_from_datafiles
            else current_creator
        )
    else:
        default_option = c.DATAFILE_FILTER_ALL

    options_data = [
        {"name": c.DATAFILE_FILTER_ALL, "frequency": len(st.session_state.datafiles)}
    ] + sorted(options_from_datafiles, key=lambda option: option["name"])
    default_option_index = [option["name"] for option in options_data].index(default_option)
    return options_data, default_option_index


def get_dummy_dataframe():
    """
    Constructs a one-row placeholder dataframe used for playing with models without any proper
    data loaded into the IDE.
    """
    df = pd.DataFrame(
        [{c.TEXT_ORIG_COL: "dummy input text", c.TEXT_GENERATED_COL: None, c.LABEL_COL: None}]
    )

    # to fail if the list of required columns changes; we'll then update the hard-coded dict above
    assert set(df.columns) == set(c.DUMMY_DATA_COLS)

    return df


def insert_hidden_html_marker(helper_element_id, target_streamlit_element=None):
    """
    Because targeting streamlit elements (e.g. to style them) is hard, we use a trick.

    We create a dummy child elements with a known ID that we can easily target.
    """
    if target_streamlit_element:
        with target_streamlit_element:
            st.markdown(f"""<div id='{helper_element_id}'/>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div id='{helper_element_id}'/>""", unsafe_allow_html=True)

    st.markdown(
        f"""
        <style>
            /* hide the dummy element */
            div:has(> div.stMarkdown > div[data-testid="stMarkdownContainer"] > div#{helper_element_id}) {{
                display: none;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_traceback_for_markdown(text):
    text = re.sub(r" ", "&nbsp;", text)
    return re.sub(r"\n", "\n\n", text)
