from app.agent import invoke_agent
from app.profile_store import render_profile


def main() -> None:
    session_id = "profile-test-session-v2"
    user_id = "profile-test-user-v2"

    queries = [
        "My name is Guy and I prefer concise technical explanations.",
        "I am especially interested in LangGraph agents and dataset analysis.",
        "What do you remember about me?",
    ]

    for query in queries:
        print("=" * 100)
        print("Query:", query)

        result = invoke_agent(
            user_query=query,
            session_id=session_id,
            user_id=user_id,
        )

        print("[route]", result["route"])
        print("[reason]", result["reason"])
        print("[answer]")
        print(result["answer"])

    print("=" * 100)
    print("Saved profile file content:")
    print(render_profile(user_id))


if __name__ == "__main__":
    main()