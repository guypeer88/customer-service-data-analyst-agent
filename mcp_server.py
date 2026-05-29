from typing import Literal

from fastmcp import FastMCP

from app.tools import (
    distribution,
    list_categories,
    list_intents,
    show_examples,
)


mcp = FastMCP("Customer service data analyst tools MCP")


@mcp.tool
def list_categories_tool() -> list[str]:
    """
    List all high-level categories in the customer service dataset.
    """
    return list_categories()


@mcp.tool
def list_intents_tool(category: str | None = None) -> list[str]:
    """
    List all intents in the dataset.

    Optionally filter intents by a high-level category,
    such as ACCOUNT, REFUND, SHIPPING, or FEEDBACK.
    """
    return list_intents(category=category)


@mcp.tool
def distribution_tool(
    group_by: Literal["category", "intent"],
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
) -> dict[str, int]:
    """
    Compute a frequency distribution by category or intent.

    Optionally filter first by category, intent, or text query.
    """
    return distribution(
        group_by=group_by,
        category=category,
        intent=intent,
        text_query=text_query,
    )


@mcp.tool
def show_examples_tool(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
    n: int = 3,
    offset: int = 0,
) -> list[dict]:
    """
    Show example customer instructions and agent responses.

    Supports offset for follow-up pagination.
    """
    return show_examples(
        category=category,
        intent=intent,
        text_query=text_query,
        n=n,
        offset=offset,
    )


if __name__ == "__main__":
    mcp.run()