from uuid import uuid4

from app.agent import invoke_agent
from app.recommender import (
    PendingRecommendation,
    clear_pending_recommendation,
    load_pending_recommendation,
    save_pending_recommendation,
)
from app.reasoning import extract_reasoning_steps


def print_result(label: str, result: dict) -> None:
    print("=" * 100)
    print(label)
    print("[route]", result["route"])
    print("[reason]", result["reason"])
    print("[answer]")
    print(result["answer"])

    steps = extract_reasoning_steps(result["new_messages"], max_observation_chars=700)
    for step in steps:
        if step["kind"] == "tool_call":
            print("\n[tool call]")
            print("name:", step["name"])
            print("args:", step["args"])

        if step["kind"] == "observation":
            print("\n[observation]")
            print(step["content"])


def assert_route(result: dict, expected_route: str) -> None:
    if result["route"] != expected_route:
        raise AssertionError(
            f"Expected route {expected_route!r}, got {result['route']!r}: {result['reason']}"
        )


def main() -> None:
    base_id = f"recommendation-check-{uuid4().hex[:8]}"

    request_session = f"{base_id}-request"
    request_user = f"{base_id}-request-user"
    clear_pending_recommendation(request_session, request_user)

    request_result = invoke_agent(
        user_query="What should I query next?",
        session_id=request_session,
        user_id=request_user,
    )
    print_result("1. Request a recommendation", request_result)
    assert_route(request_result, "recommendation_request")

    pending = load_pending_recommendation(request_session, request_user)
    if pending is None:
        raise AssertionError("Expected a pending recommendation after recommendation_request.")

    refine_result = invoke_agent(
        user_query="Can you make it about shipping examples instead?",
        session_id=request_session,
        user_id=request_user,
    )
    print_result("2. Refine the pending recommendation", refine_result)
    assert_route(refine_result, "recommendation_refine")

    refined = load_pending_recommendation(request_session, request_user)
    if refined is None:
        raise AssertionError("Expected a pending recommendation after recommendation_refine.")

    cancel_result = invoke_agent(
        user_query="Not now, skip that.",
        session_id=request_session,
        user_id=request_user,
    )
    print_result("3. Cancel the pending recommendation", cancel_result)
    assert_route(cancel_result, "recommendation_cancel")

    if load_pending_recommendation(request_session, request_user) is not None:
        raise AssertionError("Expected pending recommendation to be cleared after cancellation.")

    confirm_session = f"{base_id}-confirm"
    confirm_user = f"{base_id}-confirm-user"
    clear_pending_recommendation(confirm_session, confirm_user)
    save_pending_recommendation(
        confirm_session,
        confirm_user,
        PendingRecommendation(
            suggested_query="What categories exist in the dataset?",
            rationale="Categories are a simple deterministic query for checking recommendation confirmation.",
            source_user_query="Seeded by checks.check_recommendations.",
        ),
    )

    confirm_result = invoke_agent(
        user_query="Sounds good, run it for me.",
        session_id=confirm_session,
        user_id=confirm_user,
    )
    print_result("4. Confirm and execute a pending recommendation", confirm_result)
    assert_route(confirm_result, "recommendation_confirm")

    if confirm_result.get("executed_query") != "What categories exist in the dataset?":
        raise AssertionError("Expected the seeded recommendation query to be executed.")

    if load_pending_recommendation(confirm_session, confirm_user) is not None:
        raise AssertionError("Expected pending recommendation to be cleared after confirmation.")


if __name__ == "__main__":
    main()
