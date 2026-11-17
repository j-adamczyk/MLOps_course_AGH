# Laboratory 13 instruction — cloud variant

## Introduction

This lab covers building LLM-based systems, including efficient LLM inference,
enabling tool usage, and securing them with guardrails. Since this is a vast area, we
will focus on a few selected technologies but note that there is much more to learn.

This is the **cloud variant** of the instruction, for computers without an NVIDIA GPU.
vLLM, used in the main instruction, publishes only Linux wheels and requires CUDA, so it cannot run here. Instead, we will use the
[Gemini API](https://ai.google.dev/gemini-api/docs), which has a free tier and exposes
an [OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai).

This is a valuable LLMOps lesson in itself: the OpenAI chat completions API is the de
facto industry standard, and the same client code works against vLLM, ChatGPT, Gemini,
and most other serving systems by changing only the base URL, API key, and model name.
Everything you build in this lab would work against a self-hosted vLLM server with a
three-line change.

If you have a Linux machine with an NVIDIA GPU (or a remote one), use the main
[LAB_INSTRUCTION.md](LAB_INSTRUCTION.md) instead. Exercises 2-5 are identical in both
variants; Exercise 1 differs, since dynamic quantization requires a locally hosted model.

## Setup

To set up the environment, run `uv sync` to install dependencies. The `vllm` and
`bitsandbytes` packages are marked as Linux-only in `pyproject.toml`, so they are
skipped automatically on your system.

Next, get a Gemini API key:

1. Go to [Google AI Studio](https://aistudio.google.com/) and sign in with a Google account.
2. Create an API key. The free tier requires no credit card.
3. Put the key in a `.env` file next to your scripts:

```
GEMINI_API_KEY=your-key-here
```

Remember not to push it to GitHub! The `.gitignore` in this lab already excludes `.env`.

We will use the `gemini-3.5-flash` model in the examples. If it is not available on
your free tier, check [your rate limits in AI Studio](https://aistudio.google.com/rate-limit)
and pick any available "flash" model — they are fast, cheap, and support function calling.

Note that on the free tier, Google may use your API inputs and outputs to improve their
models. This is fine for this lab, but in production you must always check the data
privacy terms of any cloud LLM provider — another important LLMOps concern.

## Efficient inference

### HuggingFace transformers inference

Traditional inference with HuggingFace `transformers` library is quite unwieldy, and
has quite a bit of boilerplate code. This part runs locally on any system — on CPU, or
on the Apple Silicon GPU via the MPS backend. We
use [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B), the smallest of the
Qwen3 family, since local inference without an NVIDIA GPU is slow. For example, to use
inference with reasoning (and returning thinking trace), we do:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

# use half-precision inference and GPU if available:
# CUDA (NVIDIA) or MPS (Apple Silicon GPU)
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float16
else:
    device, dtype = "cpu", torch.float16

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
import torch
from transformers import pipeline

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = pipeline(
    task="text-generation",
    model="Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    device=device,
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

### Cloud LLM inference

For production-grade deployments, dedicated serving systems
like [vLLM](https://docs.vllm.ai/en/stable/) are used — see the main instruction for
details on PagedAttention, continuous batching, and quantization. Since vLLM does not
run on macOS or Windows, we will consume a hosted model instead. Cloud providers run
exactly such serving systems behind their APIs, so from the client's perspective the
experience is the same.

The Gemini API exposes
an [OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai), so we
use the [official OpenAI SDK](https://platform.openai.com/docs/api-reference/chat) and
just point it at Google's servers:

```python
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

chat_response = client.chat.completions.create(
    model="gemini-3.5-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How important is LLMOps on scale 0-10?"},
    ],
    max_completion_tokens=1000,
    # turn off thinking, analogous to Qwen's enable_thinking=False
    reasoning_effort="none",
)
content = chat_response.choices[0].message.content.strip()
print("Response:\n", content)

```

Gemini models are reasoning models, like Qwen3. Instead of Qwen's `/no_think` or
`chat_template_kwargs`, the OpenAI compatibility layer exposes the `reasoning_effort`
parameter with values `"none"`, `"minimal"`, `"low"`, `"medium"`, and `"high"`. We turn
reasoning off in this lab for speed, but you can experiment with it.

Note that for real-world deployments you must **always** consider security, such as
API key handling and persistent logging. Never hardcode keys in source code — that is
why we load them from `.env`.

---

## Tool usage

LLMs are powerful tools, but they are always **text-in, text-out** models. When talking about
"AI workflows", "AI agents", "function calling", and so on, LLM is only outputting text, typically
JSON, indicating that it requires external information. The system around LLM has to actually
execute that, which is an important part of LLMOps, including, e.g., efficient usage, auth, security,
logging, and tracing of such interactions.

### Manual tool calling

If we are working primarily or exclusively with internal tools and APIs, we can just implement everything
ourselves. The OpenAI SDK includes parsers for tool calls and results, automating a lot of necessary
actions, and the Gemini compatibility layer supports them.

The steps are:

1. Describe available tools using [JSON schema](https://json-schema.org/learn/getting-started-step-by-step).
2. Pass tool definitions to LLM as `tools` parameter, with `tool_choice="auto"` letting
   the model decide when to call them.
3. Parse tool calls from the response, execute the matching functions, and append
   results to the conversation.

The framework serializes tool definitions and passes them to the model automatically. It also
automatically parses tool calls under `"tool_calls"` key as `ChatCompletionMessageToolCall`
objects. For details on OpenAI Python SDK,
see [official docs](https://platform.openai.com/docs/api-reference/chat)
and [API Markdown docs](https://github.com/openai/openai-python/blob/main/api.md).

Note that this example:

- fetches the current date - one of the most commonly used tools, making the model time-aware
- uses one tool to inform the other - first get the current date, then weather forecast
- logs intermediate messages to showcase the format of OpenAI messages, this should not be user-facing,
  but can be logged in production for debugging purposes

```python
import datetime
import json
import os
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI


def make_llm_request(prompt: str) -> str:
    load_dotenv()
    client = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    messages = [
        {"role": "system", "content": "You are a weather assistant."},
        {"role": "user", "content": prompt},
    ]

    tool_definitions, tool_name_to_func = get_tool_definitions()

    # guard: loop limit, we break as soon as we get an answer
    for _ in range(10):
        response = client.chat.completions.create(
            model="gemini-3.5-flash",
            messages=messages,
            tools=tool_definitions,  # always pass all tools in this example
            tool_choice="auto",
            max_completion_tokens=1000,
            reasoning_effort="low",
        )
        resp_message = response.choices[0].message
        messages.append(resp_message.model_dump(exclude_none=True))

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
    prompt = "What will be weather in Birmingham, UK in two weeks?"
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

Note `model_dump(exclude_none=True)` and `reasoning_effort="low"` — Gemini rejects
null message fields, and with reasoning fully off it tends to skip tool calls.

#### Exercise 1 (1 point)

Connect to the Gemini API and get the manual tool calling example above working end to end:

1. Set up your API key as described in the setup section and run the example.
2. In your submission, include the logged intermediate messages for the Warsaw
   prompt and describe the sequence of tool calls: which tool the model called first,
   how it used the result of `get_current_date` to fill the `date` argument of
   `get_weather_forecast`, and how the final answer was produced.
3. The third prompt (New York in two months) violates the tool's "at most 4 weeks into
   the future" constraint. Check and describe how the model handles it.

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
use [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).
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

While OpenAI's newer Responses API supports remote MCP servers directly, the chat
completions API used here does not. We therefore build a small MCP manager that lists
tools from MCP servers, converts them to the OpenAI tools format, and dispatches tool
calls back to the right server. This pattern works with any chat-completions-compatible
backend — Gemini here, but also vLLM or ChatGPT.

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

We need to modify our LLM code to use MCP. We just need to
provide [MCP connection details](https://modelcontextprotocol.io/docs/concepts/transports),
which work similarly with remote MCP servers (using HTTP) in OpenAI, Claude Desktop, etc.
For production deployments, you should read those definitions from a file, rather than
hardcoding them. As before, the `GEMINI_API_KEY` environment variable goes into a `.env`
file next to this script. Remember not to push it to GitHub!

```python
import asyncio
import json
import os
from contextlib import AsyncExitStack

from dotenv import load_dotenv
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

    load_dotenv()
    client = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    async with MCPManager(mcp_servers) as mcp:
        messages = [
            {
                "role": "system",
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
            response = client.chat.completions.create(
                model="gemini-3.5-flash",
                messages=messages,
                tools=mcp.tools,
                tool_choice="auto",
                max_completion_tokens=1000,
                reasoning_effort="low",
            )

            response = response.choices[0].message
            if not response.tool_calls:
                return response.content

            messages.append(response.model_dump(exclude_none=True))
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
    prompt = "What will be weather in Birmingham, UK in two weeks?"
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
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "gemini-3.5-flash"


def check_output_guardrail_competitor_mention(client: OpenAI, prompt: str) -> bool:
    chat_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
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
        reasoning_effort="none",
    )
    content = chat_response.choices[0].message.content.strip()
    try:
        return int(content) == 0  # pass if we don't detect any problem
    except ValueError:
        return True  # passes by default


def make_llm_request(prompt: str) -> str:
    client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_BASE_URL)

    messages = [
        {
            "role": "system",
            "content": "You are a old fishing fanatic, focusing on fish exclusively, talking only about fish.",
        },
        {"role": "user", "content": prompt},
    ]

    chat_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_completion_tokens=1000,
        reasoning_effort="none",
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
            model=MODEL,
            messages=messages,
            max_completion_tokens=1000,
            reasoning_effort="none",
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
import os

from dotenv import load_dotenv
from guardrails import Guard, OnFailAction
from guardrails.hub import LlamaGuard7B
from openai import OpenAI

load_dotenv()


def make_llm_request(prompt: str) -> str:
    client = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    chat_response = client.chat.completions.create(
        model="gemini-3.5-flash",
        messages=messages,
        max_completion_tokens=1000,
        reasoning_effort="none",
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

---

## Lab assessment

Send the code of all exercises, with your answers and comments in a short Markdown report. Note which variant
of the lab you used. Exercises 1-5 are worth 1 point each.
