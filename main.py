import argparse

from app.agent import invoke_agent
from app.reasoning import extract_reasoning_steps


def print_reasoning_steps(messages: list) -> None:
    """
    Print visible reasoning steps: tool calls and tool observations.

    This does not expose hidden chain-of-thought.
    It only prints the actual tools used and their returned observations.
    """
    for step in extract_reasoning_steps(messages):
        if step["kind"] == "tool_call":
            print("\n[tool call]")
            print(f"name: {step['name']}")
            print(f"args: {step['args']}")

        if step["kind"] == "observation":
            print("\n[observation]")
            print(step["content"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Customer Service Data Analyst Agent CLI",
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Persistent conversation session ID.",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="Persistent user profile ID. Defaults to the session ID.",
    )
    parser.add_argument(
        "--hide-reasoning",
        action="store_true",
        help="Hide tool calls and observations.",
    )

    args = parser.parse_args()
    user_id = args.user_id or args.session

    print("Customer Service Data Analyst Agent")
    print(f"Session: {args.session}")
    print(f"User ID: {user_id}")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 80)

    while True:
        user_query = input("\nYou: ").strip()

        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not user_query:
            continue

        result = invoke_agent(
            user_query=user_query,
            session_id=args.session,
            user_id=user_id,
        )

        print(f"\n[route] {result['route']}")
        print(f"[route reason] {result['reason']}")

        if not args.hide_reasoning:
            print_reasoning_steps(result["new_messages"])

        print("\nAgent:")
        print(result["answer"])


if __name__ == "__main__":
    main()
