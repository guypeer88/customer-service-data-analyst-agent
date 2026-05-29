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
- If there is a pending recommendation, interpret short natural replies in that
  context. For example, "sounds good", "sure", "run it for me", or "that works"
  usually confirm; "maybe examples instead" or "make it about shipping" refine;
  "skip it" or "not now" cancel.
- If there is a pending recommendation but the user asks a new dataset question,
  classify the new question as structured or unstructured instead of forcing a
  recommendation route.
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
    pending_recommendation: str | None = None,
) -> RouteDecision:
    """
    Classify a user query before the agent chooses tools.

    The classification is LLM-based. Pending recommendation context is included
    when available so confirmation/refinement/cancel decisions are semantic
    rather than keyword-based.
    """
    llm = build_llm()
    pending_summary = pending_recommendation if pending_recommendation else None

    response = llm.invoke(
        [
            ("system", ROUTER_SYSTEM_PROMPT),
            (
                "user",
                json.dumps(
                    {
                        "query": user_query,
                        "has_pending_recommendation": bool(
                            has_pending_recommendation or pending_summary
                        ),
                        "pending_recommendation": pending_summary,
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
