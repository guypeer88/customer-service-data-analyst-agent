from app.agent import invoke_agent


def main() -> None:
    session_id = "memory-test-refund_4"

    queries = [
        "Show me 3 examples from the REFUND category.",
        "Show me 3 more.",
        "Show me 2 more.",
    ]

    for query in queries:
        print("=" * 100)
        print("Query:", query)

        result = invoke_agent(
            user_query=query,
            session_id=session_id,
        )

        print("[route]", result["route"])
        print("[answer]")
        print(result["answer"])


if __name__ == "__main__":
    main()