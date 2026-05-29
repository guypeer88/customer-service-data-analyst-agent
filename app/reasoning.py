from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, ToolMessage


class ReasoningStep(TypedDict, total=False):
    kind: Literal["tool_call", "observation"]
    name: str
    args: dict[str, Any]
    content: str


def extract_reasoning_steps(
    messages: list[Any],
    max_observation_chars: int = 1000,
) -> list[ReasoningStep]:
    """
    Extract visible tool calls and observations from LangChain messages.
    """
    steps: list[ReasoningStep] = []

    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                steps.append(
                    {
                        "kind": "tool_call",
                        "name": tool_call["name"],
                        "args": tool_call["args"],
                    }
                )

        if isinstance(message, ToolMessage):
            content = str(message.content)
            steps.append(
                {
                    "kind": "observation",
                    "name": getattr(message, "name", None) or "tool",
                    "content": content[:max_observation_chars]
                    + ("..." if len(content) > max_observation_chars else ""),
                }
            )

    return steps
