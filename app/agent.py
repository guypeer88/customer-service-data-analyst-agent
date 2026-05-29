import socket
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import GraphRecursionError
from langchain.agents import create_agent
from psycopg.conninfo import conninfo_to_dict
from psycopg_pool import ConnectionPool

from app.config import settings
from app.utils import build_llm
from app.router import RouteDecision, route_query
from app.tools import LANGCHAIN_TOOLS
from app.profile_store import render_profile
from app.profile_updater import update_profile_from_turn
from app.recommender import (
    build_recommendation_answer,
    clear_pending_recommendation,
    has_pending_recommendation,
    load_pending_recommendation,
    recommend_next_query,
    refine_recommendation,
    save_pending_recommendation,
)

AGENT_SYSTEM_PROMPT = """
You are a customer service dataset analyst agent.

You answer questions ONLY about the Bitext customer service dataset.

You have access to safe, bounded dataset-analysis tools:
- list_categories
- list_intents
- count_rows
- show_examples
- distribution
- get_rows_for_summary

Rules:
- Use tools for dataset facts unless the exact answer already appears in the current conversation history.
- For "how many" questions, always use count_rows directly with the relevant filters.
- Do not use distribution for simple total-count questions.
- Use distribution only when the user asks for a breakdown, distribution, or comparison.
- For examples, use show_examples.
- For follow-up example requests like "show me 3 more", inspect the most recent previous show_examples tool call in the conversation.
  Reuse its category, intent, and text_query arguments.
  Set offset to previous offset + previous n.
- For open-ended summaries, first use get_rows_for_summary, then summarize based only on those rows.
- Never request raw full dataset rows.
- If no matching rows are found, say so clearly.
- If the user asks something unrelated to the dataset, do not answer it.
- Keep the final answer concise and grounded in the tool results.
"""

def build_checkpointer() -> PostgresSaver:
    """
    Build a Postgres-backed LangGraph checkpointer.

    The connection pool stays alive for the process lifetime.
    """
    pool = ConnectionPool(
        conninfo=settings.database_url,
        max_size=10,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
        },
    )

    checkpointer = PostgresSaver(pool)
    checkpointer.setup()

    return checkpointer


def build_agent():
    """
    Build the LangGraph ReAct agent with dataset tools and Postgres memory.
    """
    llm = build_llm()
    checkpointer = build_checkpointer()

    return create_agent(
        model=llm,
        tools=LANGCHAIN_TOOLS,
        checkpointer=checkpointer,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )


_AGENT: Any | None = None


def get_agent():
    """
    Return a lazily built LangGraph agent.

    Lazy construction keeps simple imports from opening a database connection
    before the app actually needs to answer a dataset query.
    """
    global _AGENT

    if _AGENT is None:
        _AGENT = build_agent()

    return _AGENT


def _graph_config(session_id: str) -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": session_id,
        },
        "recursion_limit": 15,
    }


def _state_messages(agent: Any, config: dict[str, Any]) -> list[Any]:
    state = agent.get_state(config)
    return state.values.get("messages", [])


def _recent_messages(session_id: str) -> list[Any]:
    if not _database_port_open():
        return []

    try:
        return _state_messages(get_agent(), _graph_config(session_id))
    except Exception:
        return []


def _database_port_open() -> bool:
    try:
        conninfo = conninfo_to_dict(settings.database_url)
        host = conninfo.get("host") or "localhost"
        port = int(conninfo.get("port") or 5432)

        if host.startswith("/"):
            return True

        with socket.create_connection((host, port), timeout=0.5):
            return True

    except Exception:
        return False


def _base_response(
    route_decision: RouteDecision,
    answer: str,
    messages: list[Any] | None = None,
    new_messages: list[Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    response = {
        "route": route_decision.route,
        "reason": route_decision.reason,
        "answer": answer,
        "messages": messages or [],
        "new_messages": new_messages or [],
    }
    response.update(extra)
    return response


def invoke_agent(
    user_query: str,
    session_id: str = "default",
    user_id: str = "default",
) -> dict[str, Any]:
    """
    Route a user query and invoke the agent when the query is in scope.
    """
    pending_exists = has_pending_recommendation(session_id, user_id)
    route_decision = route_query(
        user_query,
        has_pending_recommendation=pending_exists,
    )

    if route_decision.route in {
        "recommendation_request",
        "recommendation_refine",
        "recommendation_confirm",
        "recommendation_cancel",
    }:
        return _handle_recommendation_route(
            user_query=user_query,
            session_id=session_id,
            user_id=user_id,
            route_decision=route_decision,
        )

    if pending_exists:
        clear_pending_recommendation(session_id, user_id)

    if route_decision.route == "out_of_scope":
        return _base_response(
            route_decision,
            (
                "I can only answer questions about the Bitext customer service dataset "
                "or saved user profile memory."
            ),
        )

    if route_decision.route == "profile_read":
        return _base_response(
            route_decision,
            f"Here is what I remember about you:\n\n{render_profile(user_id)}",
        )

    if route_decision.route == "profile_update":
        updated_profile = update_profile_from_turn(
            user_id=user_id,
            user_query=user_query,
            agent_answer="Profile information acknowledged.",
        )

        return _base_response(
            route_decision,
            "Got it - I updated your user profile memory.",
            profile=updated_profile.model_dump(),
        )

    return _invoke_dataset_agent(
        user_query=user_query,
        session_id=session_id,
        user_id=user_id,
        route_decision=route_decision,
    )


def _handle_recommendation_route(
    user_query: str,
    session_id: str,
    user_id: str,
    route_decision: RouteDecision,
) -> dict[str, Any]:
    if route_decision.route == "recommendation_request":
        recommendation = recommend_next_query(
            user_query=user_query,
            recent_messages=_recent_messages(session_id),
            user_id=user_id,
        )
        save_pending_recommendation(session_id, user_id, recommendation)

        return _base_response(
            route_decision,
            build_recommendation_answer(recommendation),
            recommendation=recommendation.model_dump(),
        )

    pending = load_pending_recommendation(session_id, user_id)

    if pending is None:
        return _base_response(
            route_decision,
            "I do not have a pending recommendation right now. Ask me what to query next first.",
        )

    if route_decision.route == "recommendation_refine":
        recommendation = refine_recommendation(
            user_query=user_query,
            pending=pending,
            recent_messages=_recent_messages(session_id),
            user_id=user_id,
        )
        save_pending_recommendation(session_id, user_id, recommendation)

        return _base_response(
            route_decision,
            build_recommendation_answer(recommendation),
            recommendation=recommendation.model_dump(),
        )

    if route_decision.route == "recommendation_cancel":
        clear_pending_recommendation(session_id, user_id)
        return _base_response(
            route_decision,
            "No problem - I will not run that recommendation.",
        )

    clear_pending_recommendation(session_id, user_id)
    execution_route = _route_recommended_query(pending.suggested_query)
    execution = _invoke_dataset_agent(
        user_query=pending.suggested_query,
        session_id=session_id,
        user_id=user_id,
        route_decision=execution_route,
    )

    return _base_response(
        route_decision,
        f"Executing suggested query: {pending.suggested_query}\n\n{execution['answer']}",
        messages=execution["messages"],
        new_messages=execution["new_messages"],
        recommendation=pending.model_dump(),
        executed_query=pending.suggested_query,
        executed_route=execution["route"],
    )


def _route_recommended_query(query: str) -> RouteDecision:
    route_decision = route_query(query, has_pending_recommendation=False)

    if route_decision.route in {"structured", "unstructured"}:
        return route_decision

    return RouteDecision(
        route="structured",
        reason="Executing the confirmed recommended dataset query.",
    )


def _invoke_dataset_agent(
    user_query: str,
    session_id: str,
    user_id: str,
    route_decision: RouteDecision,
) -> dict[str, Any]:
    agent = get_agent()
    config = _graph_config(session_id)

    before_messages = _state_messages(agent, config)
    before_len = len(before_messages)

    try:
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=f"Current distilled user profile:\n{render_profile(user_id)}"),
                    HumanMessage(content=user_query),
                ]
            },
            config=config,
        )

    except GraphRecursionError:
        all_messages = _state_messages(agent, config)
        new_messages = all_messages[before_len:]
        answer = (
            "I reached the maximum number of reasoning steps before producing a final answer. "
            "Please try a narrower dataset question."
        )

        update_profile_from_turn(
            user_id=user_id,
            user_query=user_query,
            agent_answer=answer,
        )

        return _base_response(
            route_decision,
            answer,
            messages=all_messages,
            new_messages=new_messages,
        )

    all_messages = result["messages"]
    new_messages = all_messages[before_len:]
    final_message = all_messages[-1]

    update_profile_from_turn(
        user_id=user_id,
        user_query=user_query,
        agent_answer=str(final_message.content),
    )

    return _base_response(
        route_decision,
        str(final_message.content),
        messages=all_messages,
        new_messages=new_messages,
    )
