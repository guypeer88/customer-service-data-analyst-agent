import json
from pydantic import ValidationError

from app.utils import build_llm, extract_json_object
from app.profile_store import UserProfile, load_profile, save_profile


PROFILE_UPDATE_PROMPT = """
You update a distilled user profile for a dataset analyst agent.

The profile should contain stable, useful facts only:
- user's name
- durable preferences
- recurring interests/topics
- useful notes for future conversations

Do not store:
- random one-off details
- temporary questions
- full conversation history
- tool outputs
- large data
- sensitive personal data unless the user explicitly asks to remember it

Return only valid JSON.
Do not use markdown.
Do not wrap the JSON in backticks.

The JSON must have exactly these keys:
{
  "name": null,
  "interests": [],
  "preferences": [],
  "frequent_topics": [],
  "notes": []
}

Rules:
- Preserve existing useful profile facts.
- Add new durable facts if the latest user message reveals them.
- Keep each list short.
- Avoid duplicates.
"""


def update_profile_from_turn(
    user_id: str,
    user_query: str,
    agent_answer: str,
) -> UserProfile:
    """
    Update the distilled user profile based on the latest turn.

    This is intentionally separate from checkpoint memory.
    """
    current_profile = load_profile(user_id)
    llm = build_llm()

    response = llm.invoke(
        [
            ("system", PROFILE_UPDATE_PROMPT),
            (
                "user",
                json.dumps(
                    {
                        "current_profile": current_profile.model_dump(),
                        "latest_user_message": user_query,
                        "latest_agent_answer": agent_answer,
                    },
                    ensure_ascii=False,
                ),
            ),
        ]
    )

    try:
        json_text = extract_json_object(str(response.content))
        updated_profile = UserProfile.model_validate_json(json_text)

    except (json.JSONDecodeError, ValidationError, ValueError):
        updated_profile = current_profile

    save_profile(user_id, updated_profile)

    return updated_profile