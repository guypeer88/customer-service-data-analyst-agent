from app.agent import invoke_agent
from app.reasoning import extract_reasoning_steps


TEST_QUERIES = [
    "What categories exist in the dataset?",
    "How many refund requests did we get?",
    "Show me 3 examples from the SHIPPING category.",
    "What is the distribution of intents in the ACCOUNT category?",
    "Summarize the FEEDBACK category.",
    "Who is the president of France?",
]


def print_reasoning_steps(messages: list) -> None:
    """
    Print visible tool calls and observations.

    This does not print hidden chain-of-thought.
    It only prints tool names, arguments, and tool results.
    """
    for step in extract_reasoning_steps(messages, max_observation_chars=800):
        if step["kind"] == "tool_call":
            print("\n[tool call]")
            print("name:", step["name"])
            print("args:", step["args"])

        if step["kind"] == "observation":
            print("\n[observation]")
            print(step["content"])


def main() -> None:
    for i, query in enumerate(TEST_QUERIES):
        session_id = f"check-agent-demo-v2_{i}"

        print("=" * 100)
        print("Query:", query)

        result = invoke_agent(
            user_query=query,
            session_id=session_id,
        )

        print("\n[route]", result["route"])
        print("[route reason]", result["reason"])

        print_reasoning_steps(result["messages"])

        print("\n[answer]")
        print(result["answer"])


if __name__ == "__main__":
    main()
