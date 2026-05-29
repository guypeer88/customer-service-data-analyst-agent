import json

import streamlit as st
from typing import Any

from app.agent import invoke_agent
from app.reasoning import ReasoningStep, extract_reasoning_steps


def render_reasoning_steps(steps: list[ReasoningStep]) -> None:
    if not steps:
        return

    with st.expander("Reasoning steps", expanded=False):
        for index, step in enumerate(steps, start=1):
            if step["kind"] == "tool_call":
                st.markdown(f"**{index}. Tool call:** `{step['name']}`")
                st.json(step["args"])
            else:
                st.markdown(f"**{index}. Observation:** `{step['name']}`")
                st.code(step["content"], language="text")


def render_message(message: dict[str, Any], show_reasoning: bool) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            route = message.get("route")
            reason = message.get("reason")

            if route and reason:
                st.caption(f"route: {route} | {reason}")

            if show_reasoning:
                render_reasoning_steps(message.get("reasoning_steps", []))


def main() -> None:
    st.set_page_config(
        page_title="Customer Service Data Analyst Agent",
        page_icon="CS",
        layout="centered",
    )

    st.title("Customer Service Data Analyst Agent")

    with st.sidebar:
        session_id = st.text_input("Session ID", value="default")
        user_id = st.text_input("User ID", value=session_id)
        show_reasoning = st.toggle("Show reasoning", value=True)

        if st.button("Clear visible chat"):
            st.session_state.chat_messages = []

    conversation_key = f"{user_id}:{session_id}"
    if st.session_state.get("conversation_key") != conversation_key:
        st.session_state.conversation_key = conversation_key
        st.session_state.chat_messages = []

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        render_message(message, show_reasoning)

    prompt = st.chat_input("Ask about the customer service dataset")

    if prompt:
        user_message = {
            "role": "user",
            "content": prompt,
        }
        st.session_state.chat_messages.append(user_message)
        render_message(user_message, show_reasoning)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = invoke_agent(
                        user_query=prompt,
                        session_id=session_id,
                        user_id=user_id,
                    )
                    reasoning_steps = extract_reasoning_steps(
                        result["new_messages"],
                        max_observation_chars=2000,
                    )
                    assistant_message = {
                        "role": "assistant",
                        "content": str(result["answer"]),
                        "route": result["route"],
                        "reason": result["reason"],
                        "reasoning_steps": reasoning_steps,
                    }

                except Exception as exc:
                    assistant_message = {
                        "role": "assistant",
                        "content": f"Something went wrong: `{exc}`",
                        "route": "error",
                        "reason": "The app could not complete the request.",
                        "reasoning_steps": [],
                    }

            st.markdown(assistant_message["content"])
            st.caption(
                f"route: {assistant_message['route']} | {assistant_message['reason']}"
            )

            if show_reasoning:
                render_reasoning_steps(assistant_message["reasoning_steps"])

        st.session_state.chat_messages.append(
            json.loads(json.dumps(assistant_message, default=str))
        )


if __name__ == "__main__":
    main()
