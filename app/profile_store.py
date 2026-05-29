import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.utils import safe_id


PROFILE_DIR = Path(".profiles")
PROFILE_DIR.mkdir(exist_ok=True)

class UserProfile(BaseModel):
    name: str | None = None
    interests: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    frequent_topics: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _profile_path(user_id: str) -> Path:
    return PROFILE_DIR / f"{safe_id(user_id)}.json"


def load_profile(user_id: str) -> UserProfile:
    """
    Load a persistent user profile.

    The profile is distilled memory, not a full transcript.
    """
    path = _profile_path(user_id)

    if not path.exists():
        return UserProfile()

    try:
        return UserProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return UserProfile()


def save_profile(user_id: str, profile: UserProfile) -> None:
    """
    Persist a user profile as JSON.
    """
    path = _profile_path(user_id)

    path.write_text(
        profile.model_dump_json(indent=2),
        encoding="utf-8",
    )


def render_profile(user_id: str) -> str:
    """
    Render the user profile as human-readable text.
    """
    profile = load_profile(user_id)
    profile_dict = profile.model_dump()

    lines: list[str] = []
    for key, val in profile_dict.items():
        formatted_key = key.replace('_', ' ').title()

        if isinstance(val, str):
            lines.append(f"{formatted_key}: {val}")

        elif isinstance(val, list) and len(val) > 0:
            lines.append(f"{formatted_key}:")
            bullet_points =[f"  - {item}" for item in val]
            lines.append("\n".join(bullet_points))

    if not lines:
        return "I don't have any saved profile details yet."
    
    return "\n".join(lines)
