# Prompterator

<p align="center">
  <img src="https://github.com/slidoapp/prompterator/blob/main/static/images/prompterator-logo.png?raw=true" width="210px" />
</p>


<p align="center">

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  <a href="https://pycqa.github.io/isort/"><img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" /></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
  [![Scc Count Badge](https://sloc.xyz/github/slidoapp/prompterator/)](https://github.com/slidoapp/prompterator/)
  [![Scc Count Badge](https://sloc.xyz/github/slidoapp/prompterator/?category=cocomo)](https://github.com/slidoapp/prompterator/)

</p>


<strong> Prompterator is a Streamlit-based prompt-iterating IDE. It runs locally but connects to external APIs exposed by various LLMs. </strong>

<p align="center">
  <img src="https://raw.githubusercontent.com/slidoapp/prompterator/main/static/images/prompterator-legend.png" /> <br />
  <em>A screenshot of the <code>prompterator</code> interface, with highligted features / areas of interest: 1. Data Upload, 2. Compose Prompt, 3. Run Prompt, 4. Evaluate and 5. Prompt History.</em>
</p>


## Requirements

Create a virtual environment that uses Python 3.10. Then, install project requirements:

1. ```shell
   pip install poetry==1.4.2
   ```

1. ```shell
   poetry install --no-root
   ```

## How to run

### 1. Set environment variables (optional)

If you use PyCharm, consider storing these in your
[run configuration](https://www.jetbrains.com/help/pycharm/run-debug-configuration.html).
- `OPENAI_API_KEY`: Optional. Only if you want to use OpenAI models (ChatGPT, GPT-4, etc.) via 
  OpenAI APIs.
- `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_API_BASE`: Optional. Only if you want to use OpenAI
  models (ChatGPT, GPT-4, etc.) via Azure OpenAI APIs.
- `PROMPTERATOR_DATA_DIR`: Optional. Where to store the files with your prompts and generated
  texts. Defaults to `~/prompterator-data`. If you plan to work on prompts for different tasks
  or datasets, it's a good idea to use a separate directory for each one.

If you do not happen to have access to an OpenAI API key, feel free to use the `mock-gpt-3.5-turbo`
model, which is a mocked version of the OpenAI's GPT-3.5 model. This is also very helpful when
developing Prompterator itself.

### 2. Run the Streamlit app
From the root of the repository, run:

```shell
make run
```

If you want to run the app directly from PyCharm, create a run configuration:
1. Right-click `prompterator/main.py` -> More Run/Debug -> Modify Run Configuration
2. Under "Interpreter options", enter `-m poetry run streamlit run`
3. Optionally, configure environment variables as described above
4. Save and use 🚀

## Using model-specific configuration

To use the models Prompterator supports out of the box, you generally need to
at least specify an API key and/or the endpoint Prompterator ought to use when contacting them.

The sections below specify how to do that for each supported model family.

### OpenAI

- To use OpenAI APIs, set the `OPENAI_API_KEY` environment variable as per the
  [docs](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).
- To use Azure OpenAI APIs, set:
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_API_BASE` -- the base endpoint URL, excluding the `/openai/deployments/...` ending
  - `AZURE_OPENAI_API_VERSION` if your version differs from the default `2023-05-15`; see the 
    [docs](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning)

### Google Vertex

- Set the `GOOGLE_VERTEX_AUTH_TOKEN` environment variable to the output of `gcloud auth print-access-token`.
- Set the `TEXT_BISON_URL` environment variable to the URL that belongs to your `PROJECT_ID`, as per the [docs](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text#generative-ai-text-prompt-drest)

### AWS Bedrock

To use the AWS Bedrock-provided models, a version of `boto3` that supports AWS Bedrock needs to be installed.

### Cohere

- Set the `COHERE_API_KEY` environment variable to your Cohere api key as per the [docs](https://docs.cohere.com/reference/generate).

Note that to use the Cohere models, the [Cohere package](https://cohere-sdk.readthedocs.io/en/latest/cohere.html#installation) needs to be installed as well.

## Using vision models and displaying images
Prompterator supports vision models (for now `gpt-4-vision-preview`) that can take text and an image as input and output text. To use them, you need to upload a csv file with the following columns:
- `text`: is basic requirement just like in other models
- `image`: full base64 encoding of an image (example: `data:image/jpeg;base64,/9j/4AA...`)
The image will be rendered inside the displayed dataframe and next to the "generated text" area

(*Note: you also need an `OPENAI_API_KEY` environment variable to use `gpt-4-vision-preview`*)