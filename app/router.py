import json
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

from app.utils import build_llm, extract_json_object


RouteType = Literal[
    "structured",
    "unstructured",
    "out_of_scope",
    "profile_read",
    "profile_update",
    "recommendation_request",
    "recommendation_refine",
    "recommendation_confirm",
    "recommendation_cancel",
]


class RouteDecision(BaseModel):
    """
    Routing decision for a user query before the agent chooses tools.
    """
    route: RouteType = Field(
        description="The selected route for the user query.",
    )
    reason: str = Field(
        description="A short explanation for the routing decision.",
    )


ROUTER_SYSTEM_PROMPT = """
You are a routing classifier for a customer service dataset analyst agent.

Your job is to decide what kind of query the user asked.

The agent is allowed to answer questions about:
1. The Bitext customer service dataset.
2. The user's saved profile or memory.

The possible routes are:

structured:
Use this for concrete dataset-analysis questions that require counts, lists,
examples, filtering, categories, intents, or distributions.

Examples:
- What categories exist in the dataset?
- How many refund requests did we get?
- Show me 5 examples from the SHIPPING category.
- What is the distribution of intents in the ACCOUNT category?
- Show me examples of people wanting their money back.
- Show me 3 more.
- What about refunds?
- What is the total count of the last two?

unstructured:
Use this for open-ended dataset-related analysis or summarization.

Examples:
- Summarize the FEEDBACK category.
- How do customer service representatives typically respond to cancellation requests?
- Summarize complaint intents.
- What patterns do you see in refund-related conversations?

profile_read:
Use this when the user asks what is currently saved in their profile/memory.

Examples:
- What do you remember about me?
- What do you know about me?
- What is my name?
- What are my preferences?
- Remind me what you know about me.
- What am I interested in.

profile_update:
Use this when the user gives durable personal information that should update
their saved profile or memory.

Examples:
- My name is Guy.
- I prefer concise technical explanations.
- Remember that I am interested in LangGraph agents.
- I am especially interested in dataset analysis.

out_of_scope:
Use this for anything that is not answerable from the dataset or the user's profile.

Examples:
- Who is the president of France?
- Who won the 2024 Champions League?
- Write me a poem about customer service.
- What's the best CRM software for handling complaints?

recommendation_request:
Use this when the user asks for a recommended next dataset query.

Examples:
- What should I query next?
- What should I ask next?
- Suggest a follow-up question.
- Recommend a useful next query.

recommendation_refine:
Use this only when there is already a pending recommended query and the user asks
to change or refine it instead of executing it.

Examples:
- I'd rather see examples instead.
- Make it about refunds.
- Can you suggest a distribution query instead?

recommendation_confirm:
Use this only when there is already a pending recommended query and the user
confirms that it should be executed.

Examples:
- Yes, do it.
- Go ahead.
- Run that.
- Execute it.

recommendation_cancel:
Use this only when there is already a pending recommended query and the user
declines or cancels it.

Examples:
- No thanks.
- Cancel that.
- Don't run it.
- Never mind.

Important:
- Do not answer the user's query.
- Only classify the query.
- Recommendation confirm, refine, and cancel routes require a pending
  recommendation. If there is no pending recommendation, do not use those routes.
- If the query is about general customer service but not about analyzing the dataset,
  classify it as out_of_scope.
- If the query is a follow-up that appears to refer to earlier dataset analysis,
  classify it according to the likely dataset task.
- Return only valid JSON.
- Do not use markdown.
- Do not wrap the JSON in ```.

Return exactly this JSON shape:
{"route": "structured","reason": "brief reason"}
"""


def route_query(
    user_query: str,
    has_pending_recommendation: bool = False,
) -> RouteDecision:
    """
    Classify a user query before the agent chooses tools.

    Recommendation request/confirmation phrases use small deterministic guards.
    Other routes are LLM-based and validated with Pydantic.
    """
    recommendation_route = _rule_based_recommendation_route(
        user_query=user_query,
        has_pending_recommendation=has_pending_recommendation,
    )
    if recommendation_route is not None:
        return recommendation_route

    llm = build_llm()

    response = llm.invoke(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            (
                "user",
                json.dumps(
                    {
                        "query": user_query,
                        "has_pending_recommendation": has_pending_recommendation,
                    },
                    ensure_ascii=False,
                ),
            ),
        ]
    )

    try:
        json_text = extract_json_object(str(response.content))
        return RouteDecision.model_validate_json(json_text)

    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        return RouteDecision(
            route="out_of_scope",
            reason=f"Router could not produce a valid route decision: {exc}",
        )


def _rule_based_recommendation_route(
    user_query: str,
    has_pending_recommendation: bool,
) -> RouteDecision | None:
    normalized = " ".join(user_query.lower().strip().split())

    recommendation_requests = {
        "what should i query next",
        "what should i ask next",
        "what should i look at next",
        "suggest a follow-up question",
        "suggest a follow up question",
        "recommend a useful next query",
        "recommend the next query",
    }

    if normalized.rstrip("?.!") in recommendation_requests:
        return RouteDecision(
            route="recommendation_request",
            reason="The user asked for a recommended next dataset query.",
        )

    if "query next" in normalized or "ask next" in normalized:
        return RouteDecision(
            route="recommendation_request",
            reason="The user asked for a recommended next dataset query.",
        )

    if not has_pending_recommendation:
        return None

    confirmations = {
        "yes",
        "yes do it",
        "yes, do it",
        "go ahead",
        "run that",
        "execute it",
        "do it",
        "please do",
    }
    cancellations = {
        "no",
        "no thanks",
        "cancel",
        "cancel that",
        "don't run it",
        "do not run it",
        "never mind",
    }

    stripped = normalized.rstrip(".!")
    if stripped in confirmations:
        return RouteDecision(
            route="recommendation_confirm",
            reason="The user confirmed the pending recommended query.",
        )

    if stripped in cancellations:
        return RouteDecision(
            route="recommendation_cancel",
            reason="The user canceled the pending recommended query.",
        )

    refinement_markers = [
        "rather",
        "instead",
        "change",
        "make it",
        "refine",
        "different",
        "about ",
    ]
    if any(marker in normalized for marker in refinement_markers):
        return RouteDecision(
            route="recommendation_refine",
            reason="The user asked to refine the pending recommended query.",
        )

    return None
