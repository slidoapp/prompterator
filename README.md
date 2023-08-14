# Prompterator

<p align="center">
  <img src="https://github.com/slidoapp/prompterator/blob/main/static/images/prompterator-logo.png?raw=true" width="210px" />
</p>

Prompterator is a Streamlit-based prompt-iterating IDE. It runs locally but connects to
external APIs exposed by various LLMs.

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
- `OPENAI_API_KEY`: Optional. Only if you want to use OpenAI models (ChatGPT, GPT-4, etc.).
- `PROMPTERATOR_DATA_DIR`: Optional. Where to store the files with your prompts and generated
  texts. Defaults to `~/prompterator-data`. If you plan to work on prompts for different tasks
  or datasets, it's a good idea to use a separate directory for each one.

If you do not happen to have access to `OPENAI_API_KEY`, feel free to use the
`mock-gpt-3.5-turbo` model, which is a mocked version of the OpenAI's GPT-3.5
model. This is also very helpful when developing Prompterator itself.

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
