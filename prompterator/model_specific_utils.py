import json


def build_function_calling_tooling(json_schema: str):
    """
    @param json_schema: contains desired output schema in proper Json Schema format
    @return: (tools, function name) where
        - tools is list of tools (single function in this case) callable by OpenAI model
        in function calling mode.
        - function name is the name of the desired function to be called
    """
    schema = json.loads(json_schema)
    function = schema.copy()
    function_name = function.pop("title")
    description = (
        function.pop("description")
        if function.get("description", None) is not None
        else function_name
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": function_name,
                "description": description,
                "parameters": function,
            },
        }
    ]

    return tools, function_name


def build_response_format(json_schema: str):
    """
    @param json_schema: contains desired output schema in proper Json Schema format
    @return: dict with desired response format directly usable with OpenAI API
    """
    json_schema = json.loads(json_schema)
    schema = {"name": json_schema.pop("title"), "schema": json_schema, "strict": True}
    response_format = {"type": "json_schema", "json_schema": schema}

    return response_format
