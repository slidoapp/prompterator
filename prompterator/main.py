import json
import logging
import os
import traceback as tb
from collections import OrderedDict
from datetime import datetime

import pandas as pd
import streamlit as st
from diff_match_patch import diff_match_patch
from jinja2 import meta

import prompterator.constants as c
import prompterator.models as m
import prompterator.utils as u
from prompterator.postprocess_output import postprocess as postprocess_generated_text

# needed to use the simple custom component
# from apps.scripts.components_callbacks import register_callback
# from components.rate_buttons import rate_buttons

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s %(message)s", level=logging.INFO
)

st.set_page_config(
    page_title="Prompterator",
    layout="wide",
    page_icon="static/images/prompterator-icon.png",
    initial_sidebar_state="expanded",
)


def set_session_state(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v


def update_displayed_data_point():
    st.session_state.row = st.session_state.df.iloc[st.session_state.row_number]
    st.session_state.current_row_label = st.session_state.row[c.LABEL_COL]
    st.session_state.text_orig = u.get_text_orig(st.session_state.row)
    st.session_state.text_generated = u.get_text_generated(st.session_state.row)


def show_next_row():
    current_idx = st.session_state.rows_for_labelling.index(st.session_state.row_number)
    if current_idx < len(st.session_state.rows_for_labelling) - 1:
        st.session_state.row_number = st.session_state.rows_for_labelling[current_idx + 1]
        update_displayed_data_point()


def show_prev_row():
    current_idx = st.session_state.rows_for_labelling.index(st.session_state.row_number)
    if current_idx > 0:
        st.session_state.row_number = st.session_state.rows_for_labelling[current_idx - 1]
        update_displayed_data_point()


def assign_label(label_value):
    st.session_state.df.at[st.session_state.row_number, c.LABEL_COL] = label_value
    show_next_row()


def initialise_session_from_uploaded_file(df):
    for essential_col, col_type in c.DATAFILE_ESSENTIAL_COLUMNS.items():
        if essential_col not in df.columns:
            df[essential_col] = pd.Series(dtype=col_type)

    st.session_state["df"] = df
    st.session_state[c.COLS_TO_SHOW_KEY] = [c.TEXT_ORIG_COL]
    st.session_state.row = st.session_state.df.iloc[c.DEFAULT_ROW_NO]

    if st.session_state.responses_generated_externally:
        st.session_state.enable_labelling = True
        initialise_labelling()
    else:
        st.session_state.enable_labelling = False


def initialise_labelling():
    row = st.session_state.df.iloc[c.DEFAULT_ROW_NO]
    text_orig = u.get_text_orig(row)
    text_generated = u.get_text_generated(row)
    set_session_state(
        enable_labelling=True,
        row_number=c.DEFAULT_ROW_NO,
        row=row,
        current_row_label="none",
        text_orig=text_orig,
        text_generated=text_generated,
        n_data_points=len(st.session_state.df),
        rows_for_labelling=list(range(len(st.session_state.df))),
    )

    if st.session_state.responses_generated_externally:
        st.session_state[c.MODEL_KEY] = m.MODELS[c.UNKNOWN_MODEL_NAME].copy()


def set_up_dynamic_session_state_vars():
    st.session_state.n_labelled = len(
        st.session_state.df.query(
            f"{c.LABEL_COL}.notnull() | {c.REUSED_PAST_LABEL_COL}.notnull()"
            if st.session_state.get("reuse_past_labels", False)
            else f"{c.LABEL_COL}.notnull()"
        )
    )

    # we need to initialise this one, too, because it wouldn't persist in session_state in the
    # cases where no element with key `text_generated` exists -- when the diff viewer is shown.
    if "row" in st.session_state:
        set_session_state(text_generated=u.get_text_generated(st.session_state.row))


def run_prompt(progress_ui_area):
    progress_bar = progress_ui_area.progress(0, text="generating texts")

    system_prompt_template = st.session_state.system_prompt
    user_prompt_template = st.session_state.user_prompt

    structured_output_enabled = st.session_state.structured_output_enabled
    prompt_json_schema = None
    selected_structured_output_method = None
    if structured_output_enabled:
        prompt_json_schema = st.session_state.prompt_json_schema
        selected_structured_output_method = c.StructuredOutputImplementation(
            st.session_state.selected_structured_output_method
        )

    structured_output_params = c.StructuredOutputConfig(
        structured_output_enabled, prompt_json_schema, selected_structured_output_method
    )

    if not st.session_state.system_prompt.strip() and not st.session_state.user_prompt.strip():
        st.error("Both prompts are empty, not running the text generation any further.")
        return

    model = st.session_state[c.MODEL_KEY]
    model_instance = m.MODEL_INSTANCES[model.name]
    model_params = {param: st.session_state[param] for param in model.configurable_params}
    df_old = st.session_state.df.copy()

    try:
        inputs = {
            i: u.create_model_input(
                model, model_instance, user_prompt_template, system_prompt_template, row
            )
            for i, row in df_old.iterrows()
        }
        model_inputs = c.ModelInputs(
            inputs=inputs, structured_output_data=structured_output_params
        )
    except Exception as e:
        traceback = u.format_traceback_for_markdown(tb.format_exc())
        st.error(
            f"Couldn't prepare model inputs due to this error: {e}\n\nFull error "
            f"message:\n\n{traceback}"
        )
        return

    if len(model_inputs.inputs) == 0:
        st.error("No input data to generate texts from!")
        return

    results = u.generate_responses(model, model_instance, model_inputs, model_params, progress_bar)

    if all(r == {} for r in results.values()):  # no generated texts
        st.error("Something went wrong while generating texts. Try again.")
        return

    st.session_state.df = st.session_state.df[0:0].copy()

    # process results and update the dataframe
    for i, row in df_old.iterrows():
        if i not in results or "response" not in results[i]:
            st.error(
                f"No generated text found for row {i}. For error details see the terminal. "
                f'Original text: "{row[c.TEXT_ORIG_COL]}"'
            )

        row[c.RAW_TEXT_GENERATED_COL] = results[i].get("response", "GENERATION ERROR")
        if "response" in results[i]:
            try:
                postprocessed_text = postprocess_generated_text(results[i]["response"])
            except Exception:
                print(
                    f"Postprocessing the generated text failed. Generated text: "
                    f"'{results[i]}'\nException: {tb.format_exc()}"
                )
                postprocessed_text = "POSTPROCESSING ERROR"
        else:
            postprocessed_text = "GENERATION ERROR"
        row[c.TEXT_GENERATED_COL] = postprocessed_text
        row[c.RESPONSE_DATA_COL] = results[i].get("data")
        row[c.LABEL_COL] = None
        st.session_state.df.loc[len(st.session_state.df)] = row

    initialise_labelling()


def load_datafiles_into_session():
    datafiles = [
        f
        for f in os.listdir(c.DATA_STORE_DIR)
        if os.path.isfile(os.path.join(c.DATA_STORE_DIR, f)) and f.endswith(".json")
    ]

    # sort files from least to most recent
    datafiles.sort(
        key=lambda f: u.get_creation_timestamp(os.path.join(c.DATA_STORE_DIR, f)),
        reverse=False,
    )

    if st.session_state.get("datafiles") is None:
        st.session_state["datafiles"] = OrderedDict()
    for file in datafiles:
        if file not in st.session_state.datafiles:
            df, metadata = u.load_datafile(os.path.join(c.DATA_STORE_DIR, file))
            df = u.ensure_legacy_datafile_has_all_columns(df)
            st.session_state.datafiles[file] = {
                c.DATAFILE_DATA_KEY: df,
                c.DATAFILE_METADATA_KEY: metadata,
            }
            # move the most recent data file (so far) to the top
            st.session_state.datafiles.move_to_end(file, last=False)


def show_selected_datafile(file_name):
    st.session_state.enable_labelling = True
    df = st.session_state.datafiles[file_name][c.DATAFILE_DATA_KEY].copy(deep=True)
    metadata = st.session_state.datafiles[file_name][c.DATAFILE_METADATA_KEY]

    row = df.iloc[c.DEFAULT_ROW_NO]
    text_orig = u.get_text_orig(row)
    text_generated = u.get_text_generated(row)

    has_reused_labels = any(df[c.REUSED_PAST_LABEL_COL].notnull())

    set_session_state(
        df=df,
        row_number=c.DEFAULT_ROW_NO,
        row=row,
        current_row_label=row[c.LABEL_COL],
        text_orig=text_orig,
        text_generated=text_generated,
        n_data_points=len(df),
        rows_for_labelling=list(range(len(df))),
        user_prompt=metadata[c.USER_PROMPT_TEMPLATE_COL],
        system_prompt=metadata[c.SYSTEM_PROMPT_TEMPLATE_COL],
        columns_to_show=metadata.get(c.COLS_TO_SHOW_KEY, [c.TEXT_ORIG_COL]),
        reuse_past_labels=has_reused_labels,
        skip_past_label_rows=False,
        **{
            c.PROMPT_NAME_KEY: metadata[c.PROMPT_NAME_KEY],
            c.PROMPT_COMMENT_KEY: metadata[c.PROMPT_COMMENT_KEY],
            c.PROMPT_CREATOR_KEY: metadata.get(
                c.PROMPT_CREATOR_KEY, c.DEFAULTS[c.PROMPT_CREATOR_KEY]
            ),
        },
        **{c.MODEL_KEY: m.MODELS[metadata[c.MODEL_KEY]].copy()},
        **{key: metadata.get(key) for key in m.MODELS[metadata[c.MODEL_KEY]].configurable_params},
    )


def set_up_ui_model_parameters():
    with st.sidebar:
        with st.expander(label="Model configuration"):
            selected_model = st.selectbox(
                label="Model",
                index=0,  # first model selected by default
                options=sorted(m.MODELS.values(), key=lambda x: x.position_index),
                format_func=lambda model: model.name,
                key=c.MODEL_KEY,
            )
            for (
                model_param,
                model_param_properties,
            ) in selected_model.configurable_params.items():
                # Once we wanna support categorical params, we'll have to remove the hard-coded
                # use of st.number_input but for now it's OK.
                st.number_input(
                    label=model_param,
                    min_value=model_param_properties.min,
                    max_value=model_param_properties.max,
                    value=model_param_properties.default,
                    step=model_param_properties.step,
                    key=model_param,
                )


def set_up_ui_filter_saved_datafiles():
    with st.sidebar.expander("Filter datafiles"):
        for filter_attr in [c.PROMPT_CREATOR_KEY, c.MODEL_KEY]:
            options_data, default_option_index = u.get_datafile_filtering_options(filter_attr)
            selected_option = st.selectbox(
                label=filter_attr,
                options=options_data,
                index=default_option_index,
                format_func=lambda option: f"{option['name']} ({option['frequency']}x)",
            )

            # if user selected 'all', we do no further filtering here
            if selected_option["name"] != c.DATAFILE_FILTER_ALL:
                st.session_state.datafiles = OrderedDict(
                    (k, v)
                    for k, v in st.session_state.datafiles.items()
                    if u.get_datafile_filtering_attribute_from_datafile(v, filter_attr)
                    == selected_option["name"]
                )


def set_up_ui_saved_datafiles():
    for file_name, contents in st.session_state.datafiles.items():
        df = contents[c.DATAFILE_DATA_KEY]
        metadata = contents[c.DATAFILE_METADATA_KEY]
        with st.sidebar.container():
            st.markdown(
                """
                <style>
                    div[data-testid="stVerticalBlock"]:has(
                    > div[data-testid="stHorizontalBlock"]
                    > div[data-testid="column"]
                    > div[style*="flex-direction: column"]
                    > div[data-testid="stVerticalBlock"]
                    > div.element-container
                    > div.stMarkdown
                    > div[data-testid="stMarkdownContainer"]
                    > div.datafile-comment) {
                        background-color: #C4DCDF;
                        padding: 5px 10px 20px;
                        border-radius: 5px;
                        margin-top: -25px;
                    }
                    .datafile-timestamp {
                        font-size: 10px;
                        line-height: 16px;
                        color: #1C2126;
                        font-weight: bold;
                    }
                    .datafile-name {
                        font-weight: bold;
                        color: #000000;
                        line-height: 24px;
                    }
                    .datafile-comment {
                        color: #000000;
                        line-height: 24px;
                    }
                    .datafile-key-info-label {
                        padding: 2px 10px;
                        font-size: 14px;
                        line-height: 26px;
                        background: #729CA2;
                        border: 1px solid rgba(0, 0, 0, 0.08);
                        color: white;
                        border-radius: 32px;
                    }
                    .datafile-key-info-label:not(:first-child) {
                        margin-left: 2px;
                    }
                    [data-testid="stDecoration"] {
                        background-image: linear-gradient(90deg, #47F1A8, #14EDAD);
                    }
                </style>
            """,
                unsafe_allow_html=True,
            )

            with st.container():
                col1, col2 = st.columns([6, 1])
                with col1:
                    # empty name would break HTML rendering, replace it
                    prompt_name = metadata.get(c.PROMPT_NAME_KEY, "") or "&nbsp;"

                    # prettify the timestamp
                    input_timestamp = metadata[c.TIMESTAMP_COL]
                    input_format = "%Y-%m-%d-%H-%M-%S"
                    fallback_input_format = "%m-%d-%y_%H:%M:%S"
                    output_format = "%I:%M %p — %b %d, %Y"
                    try:
                        timestamp = datetime.strptime(input_timestamp, input_format)
                    except ValueError:
                        timestamp = datetime.strptime(input_timestamp, fallback_input_format)

                    st.markdown(
                        f"""
                    <div class='datafile-timestamp'>
                        {timestamp.strftime(output_format)}
                    </div>
                    <div class='datafile-name'>
                        {prompt_name}
                        <span class='datafile-correctness'>
                            ({u.get_correctness_summary(df)} ✅)
                        </span>
                    </div>
                    <div class='datafile-comment'>{metadata[c.PROMPT_COMMENT_KEY]}</div>
                    """,
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.button(
                        "🔄",
                        on_click=show_selected_datafile,
                        kwargs={"file_name": file_name},
                        key=f"open_{file_name}",
                        type="primary",
                        help="Open datafile in the main panel",
                    )

                key_info_labels = "<div>"
                for emoji, text in zip(
                    ["🧑", "🦜"], [metadata.get(c.PROMPT_CREATOR_KEY), metadata.get(c.MODEL_KEY)]
                ):
                    key_info_labels += (
                        f"<span class='datafile-key-info-label'>{emoji} " f"{text}</span>"
                    )
                key_info_labels += "</div>"
                st.markdown(key_info_labels, unsafe_allow_html=True)


def set_up_ui_generation():
    col1, col2, col3 = st.columns([3, 3, 1])
    col1.text_input(
        placeholder="name your prompt version",
        label="Prompt name",
        label_visibility="collapsed",
        key=c.PROMPT_NAME_KEY,
    )
    col2.text_input(
        placeholder="additional comment for this prompt version",
        label="Prompt comment",
        label_visibility="collapsed",
        key=c.PROMPT_COMMENT_KEY,
    )

    selected_model: c.ModelProperties = st.session_state.model
    available_structured_output_settings = (
        selected_model.supported_structured_output_implementations
    )

    structured_output_available = (
        c.StructuredOutputImplementation.FUNCTION_CALLING in available_structured_output_settings
        or c.StructuredOutputImplementation.RESPONSE_FORMAT in available_structured_output_settings
    )

    structured_output_enabled = col3.toggle(
        label="Structured output",
        value=False,
        key="structured_output_enabled",
        disabled=(not structured_output_available),
    )

    progress_ui_area = st.empty()

    col1, col2 = st.columns([3, 2])
    col1.text_area(
        label="System prompt",
        placeholder="Your system prompt goes here...",
        key="system_prompt",
        height=c.PROMPT_TEXT_AREA_HEIGHT,
    )

    model_supports_user_prompt = st.session_state[c.MODEL_KEY].is_chat_model
    default_user_prompt = c.DEFAULT_USER_PROMPT if model_supports_user_prompt else ""

    col2.text_area(
        label="User prompt",
        placeholder="Your user prompt goes here...",
        value=default_user_prompt,
        key="user_prompt",
        height=c.PROMPT_TEXT_AREA_HEIGHT,
        disabled=not model_supports_user_prompt,
    )

    if structured_output_enabled and structured_output_available:
        json_input = st.container()
        json_input.text_area(
            label="JSON Schema",
            placeholder=c.DEFAULT_JSON_SCHEMA,
            key="prompt_json_schema",
            height=c.PROMPT_TEXT_AREA_HEIGHT,
        )
        struct_options_1, struct_options_2 = st.columns([3, 2])
        struct_options_1.select_slider(
            "Select which method to use for structured output",
            options=[setting.value for setting in available_structured_output_settings],
            key="selected_structured_output_method",
        )

        if u.validate_json(st.session_state.prompt_json_schema):
            struct_options_2.success("JSON is valid", icon="🟢")
        else:
            struct_options_2.error("JSON is invalid", icon="🔴")

    if "df" in st.session_state:
        prompt_parsing_error_message_area = st.empty()
        col1, col2 = st.columns([3, 1])
        cols_for_interpolation = list(
            set(st.session_state.df.columns).difference(c.COLS_NOT_FOR_PROMPT_INTERPOLATION)
        )
        col1.write(
            f"These are the columns available in the data, feel free to include them in "
            f"your prompt(s): `{'`, `'.join(cols_for_interpolation)}`."
        )
        with col2:
            st.toggle(label="show prompt preview", value=False, key="show_prompt_preview")

        if st.session_state.show_prompt_preview:
            col1, col2 = st.columns([3, 2])
            set_up_prompt_preview(
                col1,
                st.session_state.system_prompt,
                prompt_parsing_error_message_area,
                prompt_kind="system",
            )
            set_up_prompt_preview(
                col2,
                st.session_state.user_prompt,
                prompt_parsing_error_message_area,
                prompt_kind="user",
            )

    st.button(
        label="Run prompt",
        on_click=run_prompt,
        kwargs={"progress_ui_area": progress_ui_area},
        disabled=st.session_state[c.MODEL_KEY].name == c.UNKNOWN_MODEL_NAME,
        type="primary",
    )


def create_diff_viewer(viewer_label):
    dmp = diff_match_patch()
    patches = dmp.diff_main(
        st.session_state.get("text_orig", ""),
        st.session_state.get("text_generated", ""),
    )
    dmp.diff_cleanupSemantic(patches)

    def _get_coloured_patch(patch):
        code, text = patch
        if code == 0:
            return text
        elif code == 1:
            return f"<span style='background-color:{c.TEXT_DIFF_COLOURS['add']};'>{text}</span>"
        else:
            return f"<span style='background-color:{c.TEXT_DIFF_COLOURS['delete']};'>{text}</span>"

    return (
        f"""
        <span style='font-size: 0.88em;
            color: rgba(49, 51, 63, 0.4);
            margin-bottom: -8px;
            display: block;'>
                {viewer_label}
        </span>
        <div style='border: none;
            padding: 10px;
            border-radius: 5px;
            background: #f0f2f6;
            margin-bottom: 10px;
            min-height: {c.DATA_POINT_TEXT_AREA_HEIGHT}px'>
        """
        + "".join([_get_coloured_patch(p) for p in patches])
        + "</div>"
    )


def set_up_prompt_vars_area(st_container, error_container):
    try:
        parsed_content = u.jinja_env().parse(
            st.session_state.system_prompt + st.session_state.user_prompt
        )
        used_vars = meta.find_undeclared_variables(parsed_content)
    except Exception as e:
        traceback = u.format_traceback_for_markdown(tb.format_exc())
        error_container.error(
            f"Couldn't parse the Jinja templates in the prompt(s). Ensure the "
            f"templates are valid. Short error message: {e}\n\n"
            f"Full error message:\n\n{traceback}"
        )
        used_vars = set()

    if c.TEXT_ORIG_COL in used_vars:
        used_vars.remove(c.TEXT_ORIG_COL)

    if len(used_vars) > 0:
        # create text of used prompt's variables and their values
        vars_values = ""
        for var in used_vars:
            vars_values += var + ":\n    " + str(st.session_state.row.get(var, "none")) + "\n"

        st_container.text_area(
            label=f"Attributes other than `{c.TEXT_ORIG_COL}` used in the prompt(s)",
            key="attributes",
            value=vars_values,
            disabled=True,
            height=c.DATA_POINT_TEXT_AREA_HEIGHT,
        )


def set_up_prompt_preview(st_container, prompt, error_container, prompt_kind="system"):
    if "row" in st.session_state:
        try:
            prompt = u.jinja_env().from_string(prompt).render(**st.session_state.row.to_dict())
        except Exception as e:
            traceback = u.format_traceback_for_markdown(tb.format_exc())
            error_container.error(
                f"Couldn't show {prompt_kind} prompt preview due to an error "
                f"when parsing and rendering the Jinja templates. Ensure that "
                f"your templates are valid. Short error message: {e}\n\n"
                f"Full error message:\n\n{traceback}"
            )
            prompt = "ERROR"
    else:
        prompt = ""

    st_container.text_area(
        label=f"{prompt_kind.title()} prompt preview",
        key=f"{prompt_kind}_prompt_jinja2",
        value=prompt,
        disabled=True,
        height=c.PROMPT_PREVIEW_TEXT_AREA_HEIGHT,
    )


def display_image(st_container, base64_str):
    st_container.markdown(
        f"""
        <img src="{base64_str}" style="height: 250px; margin: 0 auto; padding: 10px; display: block;"/>
        """,
        unsafe_allow_html=True,
    )


def _get_input_columns_from_df(df):
    return [col for col in df.columns.tolist() if col not in c.COLS_NOT_FOR_PROMPT_INTERPOLATION]


def _get_past_labels():
    relevant_columns = _get_input_columns_from_df(st.session_state.df) + [c.TEXT_GENERATED_COL]
    rows_needing_labels = [
        tuple(row) for _, row in st.session_state.df[relevant_columns].iterrows()
    ]
    past_labels_for_rows = {row: None for row in rows_needing_labels}
    relevant_columns_set = set(relevant_columns)

    datafiles_newest_to_oldest = sorted(
        st.session_state.datafiles.values(),
        key=lambda datafile: datafile[c.DATAFILE_METADATA_KEY][c.TIMESTAMP_COL],
        reverse=True,
    )
    for datafile in datafiles_newest_to_oldest:
        df = datafile[c.DATAFILE_DATA_KEY]

        if not relevant_columns_set.issubset(set(df.columns)):
            continue

        for _, entire_historical_row in df.iterrows():
            historical_row = tuple(entire_historical_row[relevant_columns])

            if (
                historical_row in rows_needing_labels
                # take the first (most recent) label that we encounter for current row and don't
                # overwrite it with subsequent labels encountered for the same row
                and past_labels_for_rows[historical_row] is None
            ):
                past_label = (
                    entire_historical_row[c.LABEL_COL]
                    or entire_historical_row[c.REUSED_PAST_LABEL_COL]
                )
                if past_label is not None:
                    past_labels_for_rows[historical_row] = past_label

    return [past_labels_for_rows[row] for row in rows_needing_labels]


def _handle_reuse_past_labels_toggle():
    if st.session_state.reuse_past_labels:
        past_labels = _get_past_labels()
        st.session_state.df[c.REUSED_PAST_LABEL_COL] = past_labels


def _handle_skip_past_label_rows_toggle():
    if st.session_state.skip_past_label_rows:
        rows_for_labelling = [
            i
            for i in range(len(st.session_state.df))
            if st.session_state.df.iloc[i][c.REUSED_PAST_LABEL_COL] is None
        ]

        # ensure we've got at least one available row so we can display something in the UI
        if not rows_for_labelling:
            rows_for_labelling = [st.session_state.row_number]

        st.session_state.rows_for_labelling = rows_for_labelling

        st.session_state.row_number = [
            row_idx
            for row_idx in st.session_state.rows_for_labelling
            if row_idx >= st.session_state.row_number
        ][0]
    else:
        st.session_state.rows_for_labelling = list(range(len(st.session_state.df)))


def set_up_ui_labelling():
    prompt_parsing_error_message_area = st.empty()
    col1_orig, col2_orig = st.columns([1, 1])
    text_orig_length = len(st.session_state.get("text_orig", ""))
    col1_orig.text_area(
        label=f"Original text (`{c.TEXT_ORIG_COL}` column) ({text_orig_length} chars)",
        key="text_orig",
        disabled=True,
        height=c.DATA_POINT_TEXT_AREA_HEIGHT,
    )
    set_up_prompt_vars_area(col2_orig, prompt_parsing_error_message_area)

    if "image" in st.session_state.row:
        display_image(col2_orig, st.session_state.row["image"])

    labeling_area = st.container()
    u.insert_hidden_html_marker(
        helper_element_id="labeling-area-marker", target_streamlit_element=labeling_area
    )

    st.markdown(
        """
        <style>
            /* use the helper elements of the main UI area and of the labeling area */
            /* to create a relatively nice selector */
            [data-testid="stVerticalBlock"]:has(div#main-ui-area-marker) [data-testid="stVerticalBlock"]:has(div#labeling-area-marker) { 
                padding: 10px;
                border-radius: 10px;
                border: 4px solid rgba(10, 199, 120, 0.68);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    col1_label, col2_label = labeling_area.columns([1, 1])
    generated_text_area = col1_label.container()
    text_generated_length = len(st.session_state.get("text_generated", ""))
    length_change_percentage = (text_generated_length - text_orig_length) / text_orig_length * 100
    length_change_percentage_str = (
        f"{'+' if length_change_percentage >= 0 else ''}{length_change_percentage:.0f}"
    )
    generated_text_label = (
        f"Generated text ({text_generated_length} chars | {length_change_percentage_str}%)"
    )

    if not st.session_state.get("show_diff", False):
        generated_text_area.text_area(
            label=generated_text_label,
            key="text_generated",
            value=st.session_state.get("text_generated", ""),
            disabled=True,
            height=c.DATA_POINT_TEXT_AREA_HEIGHT,
        )
    else:
        generated_text_area.markdown(
            create_diff_viewer(generated_text_label), unsafe_allow_html=True
        )

    with generated_text_area:
        st.toggle(label="show diff", value=False, key="show_diff")

    labelling_container = col2_label.container()
    labelling_container.markdown("##")
    col1, col2, col3 = labelling_container.columns([1, 1, 10])
    col1.button(
        "👍",
        key="mark_good",
        on_click=assign_label,
        kwargs={"label_value": c.LABEL_GOOD},
    )
    # Here's how we'd use the simple custom component
    # with col1:
    #     rate_buttons(label="👍", state=st.session_state.current_row_label, key="mark_good",
    #                  on_click=assign_label, kwargs={"label_value": label_good})
    col2.button("👎", key="mark_bad", on_click=assign_label, kwargs={"label_value": c.LABEL_BAD})
    col3.progress(
        st.session_state.n_labelled / len(st.session_state.df)
        if len(st.session_state.df) > 0
        else 0,
        text=f"{st.session_state.n_labelled}/{len(st.session_state.df)} labelled",
    )
    col4, col5, col6, col7, col8 = labelling_container.columns([1, 1, 2, 4, 4])
    col4.button("⬅️", key="prev_data_point", on_click=show_prev_row)
    col5.button("➡️", key="next_data_point", on_click=show_next_row)
    col6.write(f"#{st.session_state.row_number + 1}: {st.session_state.current_row_label}")
    col7.toggle(
        label="reuse past labels",
        value=False,
        key="reuse_past_labels",
        on_change=_handle_reuse_past_labels_toggle,
    )
    col8.toggle(
        label="skip past label rows",
        value=False,
        key="skip_past_label_rows",
        on_change=_handle_skip_past_label_rows_toggle,
    )
    labelling_container.button(
        "Save ⤵️", key="save_labelled_data", on_click=u.save_labelled_data, type="primary"
    )


def show_col_selection():
    if st.session_state.get("df") is not None:
        available_columns = st.session_state.df.columns.tolist()
        available_columns = [
            col for col in available_columns if col not in c.COLS_NOT_FOR_PROMPT_INTERPOLATION
        ]
        col_to_show = st.session_state.columns_to_show
        st.session_state[c.COLS_TO_SHOW_KEY] = st.multiselect(
            "Columns to show", options=available_columns, default=col_to_show
        )


def show_dataframe():
    if st.session_state.get("df") is not None:
        columns_to_show = st.session_state.columns_to_show.copy()
        columns_to_show.extend([c.TEXT_GENERATED_COL, c.LABEL_COL])
        if st.session_state.get("reuse_past_labels", False):
            columns_to_show.extend([c.REUSED_PAST_LABEL_COL])
        df_to_show = st.session_state.df[columns_to_show]
    else:
        df_to_show = u.get_dummy_dataframe()
        st.session_state.responses_generated_externally = False
        initialise_session_from_uploaded_file(df_to_show)

    st.dataframe(
        df_to_show.style.apply(
            u.categorical_conditional_highlight,
            cond_column_name=c.LABEL_COL,
            palette=c.LABEL_VALUE_COLOURS,
            axis=1,
        ),
        column_config={"image": st.column_config.ImageColumn("Preview Image")},
    )


def process_uploaded_file():
    if st.session_state.uploaded_file is not None:
        df = pd.read_csv(st.session_state.uploaded_file)
        df = df.where(pd.notnull(df), None)
        assert c.TEXT_ORIG_COL in df.columns
        st.session_state.responses_generated_externally = c.TEXT_GENERATED_COL in df.columns
        initialise_session_from_uploaded_file(df)


# Add the logo and bring the padding down (we've got the title in the logo anyway)
st.markdown(
    """
<style>
[data-testid="stSidebar"] > div > div:nth-child(2) {
    padding: 1rem 1rem 1.5rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)
st.sidebar.image("static/images/prompterator-horizontal.png")

with st.sidebar.expander(label="Input data uploader", expanded=True):
    st.file_uploader(
        label=f"Upload a CSV file with at least these columns: {c.REQUIRED_CSV_COLS}",
        key="uploaded_file",
        on_change=process_uploaded_file,
    )

# create a helper element at the top of the main UI section to later help us target the area in
# selectors
u.insert_hidden_html_marker(helper_element_id="main-ui-area-marker")

u.ensure_datafiles_directory_exists()
load_datafiles_into_session()
set_up_ui_model_parameters()
set_up_ui_filter_saved_datafiles()
set_up_ui_saved_datafiles()
set_up_ui_generation()

if st.session_state.get("enable_labelling", False):
    set_up_dynamic_session_state_vars()
    set_up_ui_labelling()

show_col_selection()
show_dataframe()
