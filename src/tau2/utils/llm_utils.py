import json
import os
import re
from typing import Any, Optional

import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# litellm._turn_on_debug()

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = os.environ.get("TAU2_ALLOW_THINKING", "false").lower() == "true"

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")
else:
    logger.info("Extended thinking is enabled for Claude models")


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            msg_dict = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": tool_calls,
            }
            # Include thinking_blocks for Claude extended thinking
            # When extended thinking is enabled, Claude requires previous assistant
            # messages to include their thinking_blocks for continuation
            if message.raw_data:
                thinking_blocks = message.raw_data.get("thinking_blocks")
                if thinking_blocks:
                    msg_dict["thinking_blocks"] = thinking_blocks
            litellm_messages.append(msg_dict)
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude"):
        if ALLOW_SONNET_THINKING:
            # Enable extended thinking with configurable budget
            # Note: thinking requires temperature=1
            budget_tokens = int(os.environ.get("TAU2_THINKING_BUDGET", "1024"))
            kwargs.setdefault("thinking", {"type": "enabled", "budget_tokens": budget_tokens})
            kwargs["temperature"] = 1.0  # Required for thinking
            # Enable interleaved thinking for tool use (Claude 4 models)
            # This allows Claude to think after receiving tool results
            # See: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#interleaved-thinking
            kwargs.setdefault("extra_headers", {})
            kwargs["extra_headers"]["anthropic-beta"] = "interleaved-thinking-2025-05-14"
            logger.info(f"Extended thinking enabled for {model}: budget={budget_tokens}, interleaved=True")
        else:
            kwargs["thinking"] = {"type": "disabled"}
            logger.info(f"Extended thinking disabled for {model}")
    litellm_messages = to_litellm_messages(messages)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"
    try:
        response = completion(
            model=model,
            messages=litellm_messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
    except Exception as e:
        logger.error(e)
        raise e
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]
    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content
    tool_calls = response.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in tool_calls
    ]
    tool_calls = tool_calls or None

    # Capture reasoning_content from extended thinking (if available)
    # litellm may expose thinking in different ways depending on model
    reasoning_content = getattr(response.message, "reasoning_content", None)
    thinking_blocks = getattr(response.message, "thinking_blocks", None)

    raw_data = response.to_dict()

    # Check message dict in raw_data for thinking
    msg_dict = raw_data.get("message", {})
    if isinstance(msg_dict, dict):
        if "thinking_blocks" in msg_dict:
            thinking_blocks = msg_dict["thinking_blocks"]
        if "reasoning_content" in msg_dict:
            reasoning_content = msg_dict["reasoning_content"]

    # Check for model_extra (pydantic extra fields)
    if hasattr(response.message, "model_extra"):
        extra = response.message.model_extra
        if extra and isinstance(extra, dict):
            if "thinking_blocks" in extra:
                thinking_blocks = extra["thinking_blocks"]
            if "reasoning_content" in extra:
                reasoning_content = extra["reasoning_content"]

    # Check raw_data directly
    if "thinking_blocks" in raw_data and not thinking_blocks:
        thinking_blocks = raw_data["thinking_blocks"]
    if "reasoning_content" in raw_data and not reasoning_content:
        reasoning_content = raw_data["reasoning_content"]

    if reasoning_content:
        raw_data["reasoning_content"] = reasoning_content
    if thinking_blocks:
        raw_data["thinking_blocks"] = thinking_blocks

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=raw_data,
    )
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
