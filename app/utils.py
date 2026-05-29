from langchain_openai import ChatOpenAI

from app.config import settings


def safe_id(value: str) -> str:
    """
    Convert a user/session id into a safe filename component.
    """
    safe = "".join(
        char for char in value
        if char.isalnum() or char in {"-", "_"}
    )

    return safe or "default"


def extract_json_object(text: str) -> str:
    """
    Extract the first JSON object from a model response.

    This makes the router a bit more robust if the model accidentally returns
    a short prefix/suffix around the JSON.
    """
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text!r}")

    return text[start : end + 1]


def build_llm(temperature: float = 0.0, max_tokens: int = 500) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.nebius_model,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
