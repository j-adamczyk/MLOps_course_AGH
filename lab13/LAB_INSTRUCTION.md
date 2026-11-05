# Laboratory 13 instruction

## Introduction

This lab will cover building LLM-based systems, including efficient LLM inference,
enabling tool usage, and securing them with guardrails. Since this is a vast area, we
will focus on a few selected technologies but note that there is much more to learn.

To set up the environment, use `uv` to install dependencies. Change `pyproject.toml` to
use either CPU or GPU version of PyTorch, depending on your hardware. If you run into
significant problems, e.g., vLLM on macOS, you can also use ChatGPT or other cloud-hosted
LLM, but note that in your lab submission.

## Efficient inference

[vLLM serving system](https://docs.vllm.ai/en/stable/) is a fast inference server and
client library for LLMs. It uses PagedAttention, continuous batching, and other techniques
to accelerate inference and increase throughput.

We will use [Qwen3-1.7B model](https://huggingface.co/Qwen/Qwen3-1.7B) throughout this
lab, since it has:

- reasonable size
- good performance
- thinking (reasoning) capabilities
- easy to turn off reasoning by adding `/no_think` to the prompt

You can switch to a [FP8-quantized variant](https://huggingface.co/Qwen/Qwen3-1.7B-FP8) or a
smaller [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B)
if necessary.

### HuggingFace transformers inference

Traditional inference with HuggingFace `transformers` library is quite unwieldy, and
has quite a bit of boilerplate code. For example, to use inference with reasoning (and
returning thinking trace), we do:

```python
import torch.cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"

# use half-precision inference and GPU if available
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device)

# prepare the model input
prompt = "How important is LLMOps on scale 0-10?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # use thinking / reasoning?
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,  # if not provided, use max context 32k
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parse only output, not thinking content
try:
    # find token 151668 "</think>"
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
print("Answer:\n", content)

```

We can simplify this
with [HuggingFace pipeline interface](https://qwen.readthedocs.io/en/latest/inference/transformers.html).

```python
import torch.cuda
from transformers import pipeline

model = pipeline(
    task="text-generation",
    model="Qwen/Qwen3-1.7B",
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

prompt = "How important is LLMOps on scale 0-10?"
messages = [{"role": "user", "content": prompt}]

messages = model(messages, max_new_tokens=32768)[0]["generated_text"]
content = messages[-1]["content"].strip()
print("Answer:\n", content)

```

However, this is still not optimal for production:

- compute-inefficient, without advanced features
  like [continuous batching and chunked prefill](https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests)
- memory-inefficient, using pre-allocated KV cache
- no built-in server, logging, debugging
- no out-of-the-box model compilation, quantization, or other advanced features

### vLLM inference

For production-grade deployments, we can use vLLM for efficient serving. It typically runs
as a standalone server, [using Docker image](https://docs.vllm.ai/en/stable/deployment/docker/)
with one of the popular LLMs. However, it is really a FastAPI server, and we can just run it
as a normal Python process. It
exposes [OpenAI-compatible API](https://docs.vllm.ai/en/stable/getting_started/quickstart/#openai-compatible-server),
so we can use it with [official OpenAI SDK](https://platform.openai.com/docs/api-reference/responses?lang=python).
There are some natural limitations to this, but for most use cases you can use exactly
the same code for both ChatGPT and local LLMs this way.

Let's run our model with vLLM, with settings:

- port `8000` (default)
- limit max generation length to 8k tokens (fits on 8 GB VRAM), adjust if needed

```bash
vllm serve Qwen/Qwen3-1.7B --port 8000 --max-model-len 8192
```

To use it, we can use the OpenAI library directly. It also enables additional capabilities
with `extra_headers` and `extra_body` parameters,
e.g. [turning thinking on/off](https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes)
for Qwen models.

```python
from openai import OpenAI

# those settings use vLLM server
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

chat_response = client.chat.completions.create(
    model="",  # use the default server model
    messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How important is LLMOps on scale 0-10? "},
    ],
    max_completion_tokens=1000,
    # turn off thinking for Qwen with /no_think
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)
content = chat_response.choices[0].message.content.strip()
print("Response:\n", content)

```

And that's it! Additional vLLM features are more optional and for particular production use cases.
For simple cases "I want to efficiently serve an LLM" this is all you need. Note that for real-world
deployments you must **always** consider security,
such as [API keys](https://docs.vllm.ai/en/stable/usage/security/#overview),
and persistent logging.

Note that Qwen pecularity is that it always outputs `<think>` and `</think>` tags at the beginning,
even if reasoning is turned off. You can get rid of them by parsing the output text manually.

#### Exercise 1 (1 point)

vLLM offers [dynamic quantization support](https://docs.vllm.ai/en/latest/features/quantization/),
where the model is quantized before inference. This is less efficient than one-time static quantization
but works with any model. It speeds up inference and also reduces the memory footprint of the model
itself, freeing up GPU memory for a larger KV cache and higher throughput.

In particular, it supports [4-bit dynamic quantization](https://docs.vllm.ai/en/latest/features/quantization/bnb/)
with [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes). Apply that
option.

Compare original and quantized models:

1. Available KV cache size in GB and tokens, vLLM logs it on startup, e.g. `Available KV cache memory:`.
2. Inference time. Prepare 10 prompts and measure the total serving time.

You can also inspect system logs with tools like RAM monitor or `nvidia-smi` to check vLLM usage.

---

## Tool usage

LLMs are powerful tools, but they are always **text-in, text-out** models. When talking about
"AI workflows", "AI agents", "function calling", and so on, LLM is only outputting text, typically
JSON, indicating that it requires external information. The system around LLM has to actually
execute that, which is an important part of LLMOps, including, e.g., efficient usage, auth, security,
logging, and tracing of such interactions.

### Manual tool calling

If we are working primarily or exclusively with internal tools and APIs, we can just implement everything
ourselves. OpenAI SDK and vLLM include parsers for tool calls and results, automating a lot of necessary
actions.

For example steps for Qwen3 + vLLM ([docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html#id1))
are:

1. Describe available tools using [JSON schema](https://json-schema.org/learn/getting-started-step-by-step).
2. Run model with `--enable-auto-tool-choice` and `--tool-call-parser hermes` arguments:

```bash
vllm serve Qwen/Qwen3-1.7B \
  --port 8000 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

3. Pass tools definitions to LLM as `tools` parameter.

Framework will serialize tool definitions and pass them to the model automatically. It also automatically
parses tool calls under `"tool_calls"` key as `ChatCompletionMessageToolCall` objects. For details on
OpenAI Python SDK, see [official docs](https://platform.openai.com/docs/api-reference/responses)
and [API Markdown docs](https://github.com/openai/openai-python/blob/main/api.md).

Run the vLLM server as above and run the code. Note that this example:

- fetches the current date - one of the most commonly used tools, making the model time-aware
- uses one tool to inform the other - first get the current date, then weather forecast
- logs intermediate messages to showcase the format of OpenAI messages, this should not be user-facing,
  but can be logged in production for debugging purposes

```python
import datetime
import json
from typing import Callable

from openai import OpenAI


def make_llm_request(prompt: str) -> str:
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

    messages = [
        {"role": "developer", "content": "You are a weather assistant."},
        {"role": "user", "content": prompt},
    ]

    tool_definitions, tool_name_to_func = get_tool_definitions()

    # guard: loop limit, we break as soon as we get an answer
    for _ in range(10):
        response = client.chat.completions.create(
            model="",
            messages=messages,
            tools=tool_definitions,  # always pass all tools in this example
            tool_choice="auto",
            max_completion_tokens=1000,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        resp_message = response.choices[0].message
        messages.append(resp_message.model_dump())

        print(f"Generated message: {resp_message.model_dump()}")
        print()

        # parse possible tool calls (assume only "function" tools)
        if resp_message.tool_calls:
            for tool_call in resp_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                # call tool, serialize result, append to messages
                func = tool_name_to_func[func_name]
                func_result = func(**func_args)

                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(func_result),
                        "tool_call_id": tool_call.id,
                    }
                )
        else:
            # no tool calls, we're done
            return resp_message.content

    # we should not get here
    last_response = resp_message.content
    return f"Could not resolve request, last response: {last_response}"


def get_tool_definitions() -> tuple[list[dict], dict[str, Callable]]:
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "get_current_date",
                "description": 'Get current date in the format "Year-Month-Day" (YYYY-MM-DD).',
                "parameters": {},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather_forecast",
                "description": "Get weather forecast at given country, city, and date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {
                            "type": "string",
                            "description": "The country the city is in.",
                        },
                        "city": {
                            "type": "string",
                            "description": "The city to get the weather for.",
                        },
                        "date": {
                            "type": "string",
                            "description": (
                                "The date to get the weather for, "
                                'in the format "Year-Month-Day" (YYYY-MM-DD). '
                                "At most 4 weeks into the future."
                            ),
                        },
                    },
                    "required": ["country", "city", "date"],
                },
            },
        },
    ]

    tool_name_to_callable = {
        "get_current_date": current_date_tool,
        "get_weather_forecast": weather_forecast_tool,
    }

    return tool_definitions, tool_name_to_callable


def current_date_tool() -> str:
    return datetime.date.today().isoformat()


def weather_forecast_tool(country: str, city: str, date: str) -> str:
    if country.lower() in {"united kingdom", "uk", "england"}:
        return "Fog and rain"
    else:
        return "Sunshine"


if __name__ == "__main__":
    prompt = "What will be weather in Birmingham in two weeks?"
    response = make_llm_request(prompt)
    print("Response:\n", response)

    print()

    prompt = "What will be weather in Warsaw the day after tomorrow?"
    response = make_llm_request(prompt)
    print("Response:\n", response)

    print()

    prompt = "What will be weather in New York in two months?"
    response = make_llm_request(prompt)
    print("Response:\n", response)

```

#### Exercise 2 (1 point)

Implement `read_remote_csv` and `read_remote_parquet` tools that read a CSV/Parquet file from URL
and return it to the LLM as text. Use Polars for this.

You can use any publicly hosted files, or make your own. Public examples:

- CSV - our own [ApisTox dataset](https://github.com/j-adamczyk/ApisTox_dataset/blob/master/outputs/dataset_final.csv))
- Parquet - [New York yellow taxi rides](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) (select
  a few dozen rows for testing)

Test your LLM with the tool, asking a few questions about the dataset. First, you need to provide it with
the URL to point to the file.

---

### Model Context Protocol (MCP)

Instead of manually defining and calling tools, we can
use [Model Context Protocol (MCP)](https://docs.vllm.ai/en/stable/framework/mcp.html).
It has considerable advantages for more complex use cases:

1. Responsibility for defining or changing tool is bound to the server, not client,
   allowing us to make changes or develop the service transparently to end users.
2. Generation of tool definitions based on Python code and docstrings.
3. Automatic parsing of tool definitions, tool calls, and outputs.
4. Plug-and-play integration with MCP hosts like Claude Desktop.
5. Using external MCP servers, e.g. [Tavily web search](https://www.tavily.com/)
   or [Postgres Pro](https://github.com/crystaldba/postgres-mcp).
6. Enables exposing your own internal tools in standardized format for internal
   and external users.

[Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) supports all features
of the MCP standard. Alternatively, [FastMCP framework](https://gofastmcp.com/getting-started/welcome)
can be used, offering extended functionalities, particularly in terms of authentication and authorization.
Here, we will use the latter, since it has nice docs and is very similar to regular MCP SDK.

MCP has two primary connection types:

1. [stdio](https://gofastmcp.com/deployment/running-server#stdio-transport-default) - for local applications
   like Claude Desktop, communicating over system `stdin`/`stdout`
2. [Streaming HTTP](https://gofastmcp.com/deployment/running-server#http-transport-streamable) - for
   remote applications, deployed servers, and deployed services

You can also find legacy SSE transport,
but [it has been deprecated](https://blog.fka.dev/blog/2025-06-06-why-mcp-deprecated-sse-and-go-with-streamable-http/?ref=blog.globalping.io)
in favor of streaming HTTP.

Let's see an example, rewriting the previous example using MCP protocol. First, the MCP server:

```python
from typing import Annotated

from fastmcp import FastMCP

mcp = FastMCP("Weather forecast")


@mcp.tool(description="Get weather forecast at given country, city, and date")
def get_weather_forecast(
        country: Annotated[str, "The country the city is in."],
        city: Annotated[str, "The city to get the weather for."],
        date: Annotated[
            str,
            "The date to get the weather for, "
            'in the format "Year-Month-Day" (YYYY-MM-DD). '
            "At most 4 weeks into the future.",
        ],
) -> str:
    if country.lower() in {"united kingdom", "uk", "england"}:
        return "Fog and rain"
    else:
        return "Sunshine"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8001)

```

Now, we need to adapt our LLM code to use MCP. A few things to note:

- only [responses OpenAI API](https://gofastmcp.com/integrations/openai) supports MCP directly
  at the current time, but its not well integrated with Qwen or many other LLMs
- adding paths to the server uses decorators, similarly to FastAPI
- tool definitions are automatically generated from the function declaration and docstrings
- input types can be primitives or Pydantic models, with
  descriptions [provided using Annotated type](https://gofastmcp.com/servers/tools#simple-string-descriptions)
- Pydantic models and [Field construct](https://gofastmcp.com/servers/tools#advanced-metadata-with-field)
  can be used to enforce constraints on input values
- return type by default uses **structured output**, e.g. string is returned as `{"result": value}`,
  and alternatively Pydantic models or dictionaries can be returned directly

All of this is translated into JSON schema automatically, so you can rely on Python and Pydantic
instead of writing JSON manually.

MCP heavily invests in async support for efficiency, and FastMCP is async-first. Further, we
may want to stream tokens to users live, particularly for long completions. Code below uses
async for this reason, since only that has full support of all features.

Further, OpenAI SDK tool declarations are not fully
MCP-compatible currently in chat completion mode, so we need to adapt them a bit. Frameworks
differ in their support to cover those issues (Ollama is really nice), but here we will deploy
a production-grade async server.

Unfortunately, Qwen [only supports Completions API](https://qwen.readthedocs.io/en/latest/deployment/vllm.html),
while [only Responses API supports MCP](https://gofastmcp.com/integrations/openai).
While it's technically possible to fix this mismatch, it's simpler to just switch to ChatGPT API
for now. This will hopefully be standardized in the future, since other LLMs often run into similar
edge cases.

To get time information, we could
use [official Time MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/time),
but it runs only in `stdio` mode. Logically separating the two makes sense, as many clients can
share a single time server.

#### Exercise 3 (1 point)

Write an MCP server running in streaming HTTP mode, exposing two tools:

- `get_current_date` - returns current date in the format "Year-Month-Day" (YYYY-MM-DD)
- `get_current_datetime` - returns current date and time in ISO 8601 format (up to seconds),
  i.e., YYYY-MM-DDTHH:MM:SS

Run it on port 8002.

---

We need to modify our LLM serving code to use MCP. Fortunately, vLLM has great integration with
MCP, since OpenAI SDK supports it out of the box. We just need to
provide [MCP connection details](https://platform.openai.com/docs/guides/tools?tool-type=remote-mcp),
which work similarly with remote MCP servers (using HTTP) in OpenAI, vLLM, Claude Desktop, etc.
For production deployments, you should read those definitions from a file, rather than hardcoding them.
Note that you need to put `OPENAI_API_KEY` environment variable in a `.env` file next to this script.
Remember not to push it to GitHub!

```python
import asyncio
import json
from contextlib import AsyncExitStack
from openai import OpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MCPManager:
    def __init__(self, servers: dict[str, str]):
        self.servers = servers
        self.clients = {}
        self.tools = []  # in OpenAI format
        self._stack = AsyncExitStack()

    async def __aenter__(self):
        for url in self.servers.values():
            # initialize MCP session with Streamable HTTP client
            read, write, session_id = await self._stack.enter_async_context(
                streamable_http_client(url)
            )
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            # use /list_tools MCP endpoint to get tools
            # parse each one to get OpenAI-compatible schema
            tools_resp = await session.list_tools()
            for t in tools_resp.tools:
                self.clients[t.name] = session
                self.tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.inputSchema,
                        },
                    }
                )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stack.aclose()

    async def call_tool(self, name: str, args: dict) -> dict | str:
        # call the MCP tool with given arguments
        result = await self.clients[name].call_tool(name, arguments=args)
        return result.content[0].text


async def make_llm_request(prompt: str) -> str:
    mcp_servers = {
        "weather_forecast": "http://localhost:8001/mcp",
        "date_time_server": "http://localhost:8002/mcp",
    }

    vllm_client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

    async with MCPManager(mcp_servers) as mcp:
        messages = [
            {
                "role": "developer",
                "content": (
                    "You are a helpful assistant. Use tools if you need to."
                    # "If the task is impossible based on your knowledge and tools, "
                    # "return that information."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # guard: loop limit, we break as soon as we get an answer
        for _ in range(10):
            response = vllm_client.chat.completions.create(
                model="",
                messages=messages,
                tools=mcp.tools,
                tool_choice="auto",
                max_completion_tokens=1000,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            response = response.choices[0].message
            if not response.tool_calls:
                return response.content

            messages.append(response)
            for tool_call in response.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"Executing tool '{func_name}'")
                func_result = await mcp.call_tool(func_name, func_args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": str(func_result),
                    }
                )


if __name__ == "__main__":
    prompt = "What will be weather in Birmingham in two weeks?"
    response = asyncio.run(make_llm_request(prompt))
    print("Response:\n", response)

    print()

    prompt = "What will be weather in Warsaw the day after tomorrow?"
    response = asyncio.run(make_llm_request(prompt))
    print("Response:\n", response)

    print()

    prompt = "What will be weather in New York in two months?"
    response = asyncio.run(make_llm_request(prompt))
    print("Response:\n", response)

```

Now, when you run the MCP server and use this code, it will automatically list tool definitions
and provide them to the server. You can have full control over this
with [custom MCP clients](https://gofastmcp.com/clients/client) in MCP host, i.e., your app. For
example, you could implement a custom UI for the tools or add tool selection like RAG-MCP.

Note that while asynchronous manager for MCP is a bit complicated here, the actual LLM code is
quite simple. It's also trivial to extend it with further tools.

#### Exercise 4 (1 point)

Create a visualization MCP server, with `line_plot` endpoint. It should create a line plot from
given data using Matplotlib and return it as an
image, [encoded with base64](https://stackoverflow.com/a/38061400/9472066).

Required parameters: `data`, one or more lists of numbers
Optional parameters: `title`, `x_label`, `y_label`, and `legend` (bool, whether to show legend)

Test it with sample data. You can decode the image and save it to disk.

## LLM security with guardrails

**Guardrails** are a set of rules that limit the scope of the LLM capabilities. They include
input and output filtering, censoring unwanted content (e.g., toxicity, NSFW, mentioning competitors),
fact-checking (grounding) to avoid hallucinations, and more. They are the primary defense against
attacks like prompt injection, jailbreaking, data exfiltration, and more.

Guardrails are often implemented in practice as either:

- simple rules, e.g., regex, keyword detection
- ML models, often using encoder-only transformers or small LLMs

You can implement guardrails by yourself, since this is basically just text parsing. This allows you
to add any business rules you need. Let's see an example.

```python
from openai import OpenAI


def check_output_guardrail_competitor_mention(client: OpenAI, prompt: str) -> bool:
    chat_response = client.chat.completions.create(
        model="",  # use the default server model
        messages=[
            {
                "role": "developer",
                "content": "You are a old fishing fanatic, focusing on fish exclusively, talking only about fish.",
            },
            {
                "role": "user",
                "content": (
                    "Does the following text mention any food other than fish quite positively? "
                    f"Output ONLY 0 (no mention) or 1 (mention).\n{prompt}"
                ),
            },
        ],
        max_completion_tokens=1000,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = chat_response.choices[0].message.content.strip()
    try:
        return int(content) == 0  # pass if we don't detect any problem
    except ValueError:
        return True  # passes by default


def make_llm_request(prompt: str) -> str:
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

    messages = [
        {
            "role": "developer",
            "content": "You are a old fishing fanatic, focusing on fish exclusively, talking only about fish.",
        },
        {"role": "user", "content": prompt},
    ]

    chat_response = client.chat.completions.create(
        model="",  # use the default server model
        messages=messages,
        max_completion_tokens=1000,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = chat_response.choices[0].message.content.strip()

    passed_guardrail = check_output_guardrail_competitor_mention(client, content)
    if not passed_guardrail:
        print("Did not pass guardrail, fixing")
        messages += [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": "Previous text contained mention of something other than fish, fix that. "
                "Output only the new fishing fanatic recommendation, without clearly showing any bias. "
                "No additional comments, acknowledgements etc.",
            },
        ]
        chat_response = client.chat.completions.create(
            model="",  # use the default server model
            messages=messages,
            max_completion_tokens=1000,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = chat_response.choices[0].message.content.strip()

    return content


if __name__ == "__main__":
    prompt = "What should I have for dinner today?"
    response = make_llm_request(prompt)
    print("Response:\n", response)

```

[Guardrails AI](https://github.com/guardrails-ai/guardrails) is a framework for implementing
guardrails in Python, offering many built-in rules, including remote APIs and commercial offerings.
To configure it for the first time, register at their page and run `guardrails configure`, input
your API key. Note that we will also use remote validators.

Let's try [LlamaGuard validator](https://guardrailsai.com/hub/validator/guardrails/llamaguard_7b).
It uses LLaMa-7B Guard model to check if the output mentions, e.g., drugs or criminal planning.
Remote inferencing runs this validator on Guardrails AI servers, so you may want to stick to
local-only validators for sensitive data.

To install this guardrail, run:

```bash
guardrails hub install hub://guardrails/llamaguard_7b
```

Using guardrail in the LLM flow is very simple:

```python
from guardrails import Guard, OnFailAction
from guardrails.hub import LlamaGuard7B
from openai import OpenAI


def make_llm_request(prompt: str) -> str:
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

    messages = [
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    chat_response = client.chat.completions.create(
        model="",  # use the default server model
        messages=messages,
        max_completion_tokens=1000,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = chat_response.choices[0].message.content.strip()

    # applies all LLaMa Guard checks by default
    guard = Guard().use(LlamaGuard7B, on_fail=OnFailAction.EXCEPTION)

    try:
        guard.validate(content)
        return content
    except Exception as e:
        return f"Sorry, I cannot help you with that, reason: {e}"


if __name__ == "__main__":
    prompt = "How can I get weeds out of my garbage bag after cutting my lawn?"
    response = make_llm_request(prompt)
    print("Response:\n", response)

    print()

    prompt = "How can I get weed for when cutting my lawn?"
    response = make_llm_request(prompt)
    print("Response:\n", response)

```

#### Exercise 5 (1 point)

Add guardrails to our fishing fanatic LLM:
- [restrict to topic](https://guardrailsai.com/hub/validator/tryolabs/restricttotopic)
- [detect jailbreak](https://guardrailsai.com/hub/validator/guardrails/detect_jailbreak)

Try to omit the security and get a non-fish response.
