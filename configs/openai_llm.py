import json
import re
import time
import random
from collections.abc import Sequence
from typing import overload
from types import SimpleNamespace

import openai
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionReasoningEffort,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import FunctionDefinition
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    MessageContentBlock,
    get_text_content_as_str,
    text_content_block_from_string,
)

# =========================
# Tunables (你可以按需调整)
# =========================

# 每次请求前的“固定节流”，降低 429 触发概率（单 key 单实验时很有效）
REQUEST_THROTTLE_SECONDS = 0.5  # 建议 0.2~0.5

# 429 时的额外睡眠上限（在 tenacity 退避之外再加一层）
RATE_LIMIT_EXTRA_SLEEP_MAX = 30.0

# chat_completion_request 的最大重试次数（这里给大一点，保证不中断；单实验跑久点也没关系）
MAX_CHAT_ATTEMPTS = 30

# tenacity 的最大退避上限（秒）
TENACITY_BACKOFF_MAX = 60


def _tool_call_to_openai(tool_call: FunctionCall) -> ChatCompletionMessageToolCallParam:
    if tool_call.id is None:
        raise ValueError("`tool_call.id` is required for OpenAI-compatible APIs.")
    return ChatCompletionMessageToolCallParam(
        id=tool_call.id,
        type="function",
        function={
            "name": tool_call.function,
            "arguments": json.dumps(tool_call.args, ensure_ascii=False),
        },
    )


_REASONING_MODELS = {"o1", "o3"}


def _is_reasoning_model(model_name: str) -> bool:
    return any(model in model_name for model in _REASONING_MODELS)


@overload
def _content_blocks_to_openai_content_blocks(
    message: ChatUserMessage | ChatSystemMessage,
) -> list[ChatCompletionContentPartTextParam]: ...


@overload
def _content_blocks_to_openai_content_blocks(
    message: ChatAssistantMessage | ChatToolResultMessage,
) -> list[ChatCompletionContentPartTextParam] | None: ...


def _content_blocks_to_openai_content_blocks(
    message: ChatUserMessage | ChatAssistantMessage | ChatSystemMessage | ChatToolResultMessage,
) -> list[ChatCompletionContentPartTextParam] | None:
    """
    Convert AgentDojo message content to OpenAI-compatible content blocks.

    Supported inputs:
      - None -> None
      - str -> [{"type":"text","text":...}]
      - list[{"type":"text","content":...}] -> blocks
      - other -> str(...) as one text block
    """
    content = message.get("content")

    if content is None:
        return None

    if isinstance(content, str):
        return [ChatCompletionContentPartTextParam(type="text", text=content)]

    if isinstance(content, list):
        blocks: list[ChatCompletionContentPartTextParam] = []
        for el in content:
            if isinstance(el, dict):
                text = el.get("content") or ""
            else:
                text = str(el)
            blocks.append(ChatCompletionContentPartTextParam(type="text", text=text))
        return blocks

    return [ChatCompletionContentPartTextParam(type="text", text=str(content))]


def _message_to_openai(message: ChatMessage, model_name: str) -> ChatCompletionMessageParam:
    """
    IMPORTANT: For OpenAI-compatible providers (DashScope, etc.), do NOT emit role="developer".
    Many providers only accept: system/user/assistant/tool/function.
    """
    match message["role"]:
        case "system":
            return ChatCompletionSystemMessageParam(
                role="system",
                content=_content_blocks_to_openai_content_blocks(message),
            )

        case "user":
            return ChatCompletionUserMessageParam(
                role="user",
                content=_content_blocks_to_openai_content_blocks(message),
            )

        case "assistant":
            if message["tool_calls"] is not None and len(message["tool_calls"]) > 0:
                tool_calls = [_tool_call_to_openai(tool_call) for tool_call in message["tool_calls"]]
                return ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=_content_blocks_to_openai_content_blocks(message),
                    tool_calls=tool_calls,
                )
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=_content_blocks_to_openai_content_blocks(message),
            )

        case "tool":
            if message["tool_call_id"] is None:
                raise ValueError("`tool_call_id` should be specified for OpenAI-compatible APIs.")
            return ChatCompletionToolMessageParam(
                content=message["error"] or _content_blocks_to_openai_content_blocks(message),
                tool_call_id=message["tool_call_id"],
                role="tool",
                name=message["tool_call"].function,  # type: ignore
            )

        case _:
            raise ValueError(f"Invalid message type: {message}")


def _safe_parse_tool_arguments(arg_str: str | None) -> dict:
    """
    Robustly parse tool_call.function.arguments into a dict.

    Some OpenAI-compatible providers may return non-strict JSON in `arguments`, e.g.:
      - extra trailing text
      - multiple JSON objects concatenated
      - explanation appended
    This function extracts the first balanced JSON object and parses it.
    """
    if arg_str is None:
        return {}

    s = str(arg_str).strip()
    if not s:
        return {}

    # 1) strict JSON fast path
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {"_": obj}
    except Exception:
        pass

    # 2) brace matching for first {...}
    start = s.find("{")
    if start == -1:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {"_": obj}
        except Exception:
            return {}

    depth = 0
    end = -1
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end != -1:
        candidate = s[start : end + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else {"_": obj}
        except Exception:
            pass

    # 3) regex fallback
    m = re.search(r"\{.*\}", s)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else {"_": obj}
        except Exception:
            return {}

    return {}


def _openai_to_tool_call(tool_call: ChatCompletionMessageToolCall) -> FunctionCall:
    args = _safe_parse_tool_arguments(tool_call.function.arguments)
    return FunctionCall(
        function=tool_call.function.name,
        args=args,
        id=tool_call.id,
    )


def _assistant_message_to_content(message: ChatCompletionMessage) -> list[MessageContentBlock] | None:
    if message.content is None:
        return None
    return [text_content_block_from_string(message.content)]


def _openai_to_assistant_message(message: ChatCompletionMessage) -> ChatAssistantMessage:
    if message.tool_calls is not None:
        tool_calls = [_openai_to_tool_call(tool_call) for tool_call in message.tool_calls]
    else:
        tool_calls = None
    return ChatAssistantMessage(role="assistant", content=_assistant_message_to_content(message), tool_calls=tool_calls)


def _function_to_openai(f: Function) -> ChatCompletionToolParam:
    function_definition = FunctionDefinition(
        name=f.name,
        description=f.description,
        parameters=f.parameters.model_json_schema(),
    )
    return ChatCompletionToolParam(type="function", function=function_definition)


def _is_rate_limit(e: Exception) -> bool:
    # openai.RateLimitError is the canonical one
    if isinstance(e, openai.RateLimitError):
        return True
    # 有些兼容网关可能会包一层，兜底用字符串判断
    msg = str(e).lower()
    return ("429" in msg) or ("rate limit" in msg) or ("too many requests" in msg) or ("token-limit" in msg)


def _is_engine_abort_badrequest(e: Exception) -> bool:
    """
    DashScope/Qwen compatible endpoint sometimes returns HTTP 400 (BadRequestError),
    but the payload actually indicates server-side 500 abort like STOP_ENGINE_ABORT.
    We treat it as transient and retryable.
    """
    s = str(e)
    return (
        ("STOP_ENGINE_ABORT" in s)
        or ("InternalError.EngineAbort" in s)
        or ("ModelServingUnknownError" in s)
        or ("ModelServingError" in s)
        or ("InternalError.Algo.ModelServingError" in s)
    )


def _rate_limit_extra_sleep(attempt_idx: int) -> float:
    """
    在 tenacity 的指数退避之外，再叠一层“温和加速”的额外 sleep，
    让单 key 单实验更稳（减少连续撞限流）。
    """
    base = min(1.0 * (attempt_idx + 1), RATE_LIMIT_EXTRA_SLEEP_MAX)
    jitter = random.random() * 0.5
    return min(base + jitter, RATE_LIMIT_EXTRA_SLEEP_MAX)


@retry(
    wait=wait_random_exponential(multiplier=1, max=TENACITY_BACKOFF_MAX),
    stop=stop_after_attempt(MAX_CHAT_ATTEMPTS),
    reraise=True,
    # 注意：我们“默认不重试”BadRequest/422，但如果是 STOP_ENGINE_ABORT，我们会在函数体里转成 RuntimeError，让 tenacity 重试。
    retry=retry_if_not_exception_type((openai.BadRequestError, openai.UnprocessableEntityError)),
)
def chat_completion_request(
    client: openai.OpenAI,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    tools: Sequence[ChatCompletionToolParam],
    reasoning_effort: ChatCompletionReasoningEffort | None,
    temperature: float | None = 0.0,
):
    """
    强鲁棒版本：429 时不崩溃、自动慢下来、持续重试直到成功（或达到 MAX_CHAT_ATTEMPTS）。

    重要：我们不返回 dummy（避免污染实验结果）。
    例外：Azure content_filter 这种“必然失败”场景，返回安全拒答以不中断流水线。
    """
    kwargs = dict(
        model=model,
        messages=messages,
        tools=tools or NOT_GIVEN,
        tool_choice="auto" if tools else NOT_GIVEN,
        temperature=temperature or NOT_GIVEN,
    )
    if reasoning_effort is not None and _is_reasoning_model(model):
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        return client.chat.completions.create(**kwargs)

    except openai.BadRequestError as e:
        # 1) 你原来的 Azure content_filter 兜底保留
        err_text = str(e)
        if "content management policy" in err_text or "content_filter" in err_text:
            print("[OpenAI LLM] Hit Azure content_filter, returning safe refusal.")
            dummy_msg = SimpleNamespace(
                role="assistant",
                content="I'm unable to comply with this request due to content safety restrictions.",
                tool_calls=None,
            )
            dummy_choice = SimpleNamespace(index=0, message=dummy_msg, finish_reason="stop")
            dummy_resp = SimpleNamespace(id="dummy-content-filter", choices=[dummy_choice], model=model)
            return dummy_resp

        # 2) 关键修复：把“伪 400 的服务端 abort”转成可重试异常
        if _is_engine_abort_badrequest(e):
            print("[OpenAI LLM] EngineAbort detected inside BadRequestError; converting to retryable RuntimeError.")
            raise RuntimeError("Transient model serving abort (retryable)") from e

        # 3) 真正的 BadRequest：不重试，直接抛出
        raise

    except Exception as e:
        # 对 429：在 tenacity 重试之前，先额外 sleep 一下（提升稳态通过率）
        if _is_rate_limit(e):
            t = 2.0 + random.random() * 1.0
            print(f"[OpenAI LLM] RateLimit detected; sleeping {t:.2f}s before retry.")
            time.sleep(t)
        raise


class OpenAILLM(BasePipelineElement):
    """LLM pipeline element that uses an OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        reasoning_effort: ChatCompletionReasoningEffort | None = None,
        temperature: float | None = 0.0,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.reasoning_effort: ChatCompletionReasoningEffort | None = reasoning_effort

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        # 固定节流：降低单 key 瞬时 TPS/TPM 峰值
        if REQUEST_THROTTLE_SECONDS and REQUEST_THROTTLE_SECONDS > 0:
            time.sleep(REQUEST_THROTTLE_SECONDS)

        openai_messages = [_message_to_openai(message, self.model) for message in messages]
        openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]

        completion = chat_completion_request(
            self.client,
            self.model,
            openai_messages,
            openai_tools,
            self.reasoning_effort,
            self.temperature,
        )

        output = _openai_to_assistant_message(completion.choices[0].message)
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args


class OpenAILLMToolFilter(BasePipelineElement):
    def __init__(self, prompt: str, client: openai.OpenAI, model: str, temperature: float | None = 0.0) -> None:
        self.prompt = prompt
        self.client = client
        self.model = model
        self.temperature = temperature

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if REQUEST_THROTTLE_SECONDS and REQUEST_THROTTLE_SECONDS > 0:
            time.sleep(REQUEST_THROTTLE_SECONDS)

        messages = [*messages, ChatUserMessage(role="user", content=[text_content_block_from_string(self.prompt)])]

        openai_messages = [_message_to_openai(message, self.model) for message in messages]
        openai_tools = [_function_to_openai(tool) for tool in runtime.functions.values()]

        # ToolFilter 这里也可能触发 429 / EngineAbort；用同一套 retry 包装
        completion = chat_completion_request(
            self.client,
            self.model,
            openai_messages,
            openai_tools,
            reasoning_effort=None,
            temperature=self.temperature,
        )

        output = _openai_to_assistant_message(completion.choices[0].message)

        new_tools = {}
        for tool_name, tool in runtime.functions.items():
            if output["content"] is not None and tool_name in get_text_content_as_str(output["content"]):
                new_tools[tool_name] = tool

        runtime.update_functions(new_tools)
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args
