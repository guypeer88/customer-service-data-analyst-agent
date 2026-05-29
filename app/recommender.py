import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field, ValidationError

from app.profile_store import render_profile
from app.utils import build_llm, extract_json_object, safe_id


RECOMMENDATION_DIR = Path(".recommendations")


class PendingRecommendation(BaseModel):
    """
    A suggested query that is waiting for explicit user confirmation.
    """

    suggested_query: str = Field(description="The exact dataset query to run if confirmed.")
    rationale: str = Field(description="Short reason this query is useful now.")
    source_user_query: str = Field(description="The user message that requested or refined the suggestion.")


RECOMMENDATION_PROMPT = """
You recommend the next useful query for a customer service dataset analyst agent.

The agent can answer questions only about the Bitext customer service dataset and
the user's saved profile. Recommend exactly one follow-up dataset query.

Use the recent conversation and distilled user profile to make the suggestion
relevant. Do not execute the query. Do not include multiple options.

The suggested query must be answerable with these operations:
- list dataset categories
- list intents, optionally inside one category
- count rows filtered by category, intent, or text search
- show bounded examples from a category, intent, or text search
- show a distribution by category or intent
- summarize a category, intent, or text search sample

Use exact high-level categories when possible: ACCOUNT, CANCEL, CONTACT,
DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION.
For complaint-related queries, prefer the FEEDBACK category.
Write the suggested query as a natural user message, for example:
"Show me 5 examples from the REFUND category."
Do not copy the operation names from the list above as the final query.

Return only valid JSON with exactly these keys:
{
  "suggested_query": "one exact user query to run later",
  "rationale": "one short sentence explaining why it fits",
  "source_user_query": "the latest user message"
}
"""


REFINE_RECOMMENDATION_PROMPT = """
You refine one pending query recommendation for a customer service dataset
analyst agent.

The user has not confirmed execution yet. Update the pending recommended query
to reflect the user's latest preference. Do not execute the query. Do not offer
multiple options.

The refined query must be answerable with these operations:
- list dataset categories
- list intents, optionally inside one category
- count rows filtered by category, intent, or text search
- show bounded examples from a category, intent, or text search
- show a distribution by category or intent
- summarize a category, intent, or text search sample

Use exact high-level categories when possible: ACCOUNT, CANCEL, CONTACT,
DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION.
For complaint-related queries, prefer the FEEDBACK category.
Write the refined query as a natural user message, for example:
"Show me 5 examples from the REFUND category."
Do not copy the operation names from the list above as the final query.

Return only valid JSON with exactly these keys:
{
  "suggested_query": "one exact user query to run later",
  "rationale": "one short sentence explaining why it fits",
  "source_user_query": "the latest user message"
}
"""


def _pending_path(session_id: str, user_id: str) -> Path:
    return RECOMMENDATION_DIR / f"{safe_id(user_id)}__{safe_id(session_id)}.json"


def has_pending_recommendation(session_id: str, user_id: str) -> bool:
    """
    Return whether a session has a recommendation waiting for confirmation.
    """
    return _pending_path(session_id, user_id).exists()


def load_pending_recommendation(session_id: str, user_id: str) -> PendingRecommendation | None:
    """
    Load a pending recommendation, if one exists and is valid.
    """
    path = _pending_path(session_id, user_id)

    if not path.exists():
        return None

    try:
        return PendingRecommendation.model_validate_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValidationError):
        return None


def save_pending_recommendation(
    session_id: str,
    user_id: str,
    recommendation: PendingRecommendation,
) -> None:
    """
    Persist a recommendation until the user confirms, refines, or cancels it.
    """
    RECOMMENDATION_DIR.mkdir(exist_ok=True)
    path = _pending_path(session_id, user_id)
    path.write_text(
        recommendation.model_dump_json(indent=2),
        encoding="utf-8",
    )


def clear_pending_recommendation(session_id: str, user_id: str) -> None:
    """
    Clear any pending recommendation for this session and user.
    """
    path = _pending_path(session_id, user_id)

    if path.exists():
        path.unlink()


def build_recommendation_answer(recommendation: PendingRecommendation) -> str:
    """
    Render a pending recommendation for the CLI or Streamlit UI.
    """
    return (
        f"I suggest: {recommendation.suggested_query}\n\n"
        f"Why: {recommendation.rationale}\n\n"
        "Should I go ahead?"
    )


def recommend_next_query(
    user_query: str,
    recent_messages: list[BaseMessage],
    user_id: str,
) -> PendingRecommendation:
    """
    Suggest exactly one useful follow-up query without executing it.
    """
    return _recommend_with_prompt(
        prompt=RECOMMENDATION_PROMPT,
        payload={
            "latest_user_message": user_query,
            "user_profile": render_profile(user_id),
            "recent_conversation": _format_recent_messages(recent_messages),
        },
        fallback_source=user_query,
    )


def refine_recommendation(
    user_query: str,
    pending: PendingRecommendation,
    recent_messages: list[BaseMessage],
    user_id: str,
) -> PendingRecommendation:
    """
    Refine an existing recommendation without executing it.
    """
    return _recommend_with_prompt(
        prompt=REFINE_RECOMMENDATION_PROMPT,
        payload={
            "latest_user_message": user_query,
            "pending_recommendation": pending.model_dump(),
            "user_profile": render_profile(user_id),
            "recent_conversation": _format_recent_messages(recent_messages),
        },
        fallback_source=user_query,
    )


def _recommend_with_prompt(
    prompt: str,
    payload: dict[str, Any],
    fallback_source: str,
) -> PendingRecommendation:
    llm = build_llm(temperature=0.2, max_tokens=350)
    response = llm.invoke(
        [
            ("system", prompt),
            ("user", json.dumps(payload, ensure_ascii=False)),
        ]
    )

    try:
        json_text = extract_json_object(str(response.content))
        recommendation = PendingRecommendation.model_validate_json(json_text)

        if recommendation.suggested_query.strip():
            return _normalize_recommendation(recommendation)

    except (json.JSONDecodeError, ValidationError, ValueError):
        pass

    return _fallback_recommendation(
        text=json.dumps(payload, ensure_ascii=False),
        source_user_query=fallback_source,
    )


def _normalize_recommendation(
    recommendation: PendingRecommendation,
) -> PendingRecommendation:
    query = recommendation.suggested_query.strip()
    normalized = " ".join(query.lower().rstrip(".").split())

    if normalized == "list dataset categories":
        query = "What categories exist in the dataset?"

    if normalized.startswith("show bounded examples from the "):
        category = normalized.removeprefix("show bounded examples from the ")
        category = category.removesuffix(" category").upper()
        query = f"Show me 5 examples from the {category} category."

    return recommendation.model_copy(update={"suggested_query": query})


def _format_recent_messages(messages: list[BaseMessage], limit: int = 12) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []

    for message in messages[-limit:]:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ToolMessage):
            role = "tool"
        else:
            role = message.type

        item: dict[str, Any] = {
            "role": role,
            "content": _truncate_text(str(message.content), 1200),
        }

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            item["tool_calls"] = [
                {
                    "name": tool_call.get("name"),
                    "args": tool_call.get("args"),
                }
                for tool_call in tool_calls
            ]

        tool_name = getattr(message, "name", None)
        if tool_name:
            item["tool_name"] = tool_name

        formatted.append(item)

    return formatted


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "..."


def _fallback_recommendation(text: str, source_user_query: str) -> PendingRecommendation:
    lower_text = text.lower()

    if "refund" in lower_text or "money back" in lower_text:
        query = "What is the distribution of intents in the REFUND category?"
        rationale = "You have been looking at refund-related data, so the intent breakdown is a useful next step."
    elif "shipping" in lower_text or "delivery" in lower_text:
        query = "Show me 5 examples from the SHIPPING category."
        rationale = "Recent context points toward shipping or delivery, so examples can make the pattern concrete."
    elif "feedback" in lower_text or "complaint" in lower_text:
        query = "Summarize the FEEDBACK category."
        rationale = "Feedback and complaint context is best explored with a short qualitative summary."
    else:
        query = "What is the distribution of intents in the ACCOUNT category?"
        rationale = "An intent distribution gives a quick structured view of one major dataset category."

    return PendingRecommendation(
        suggested_query=query,
        rationale=rationale,
        source_user_query=source_user_query,
    )
