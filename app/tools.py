from typing import Any, Literal

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.data_loader import load_customer_service_df


OUTPUT_COLUMNS = ["instruction", "category", "intent", "response"]


def _filter_df(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
) -> pd.DataFrame:
    """
    Filter the customer service dataset as a pandas DataFrame.

    This is the central filtering function used by the public tools.
    Filters are combined with AND logic:
    - category must match, if provided
    - intent must match, if provided
    - text_query must appear in at least one searchable text column, if provided

    The text query searches over instruction, response, category, and intent.
    """
    df = load_customer_service_df()

    if category is not None:
        normalized_category = category.strip().upper()
        df = df[df["category"] == normalized_category]

    if intent is not None:
        normalized_intent = intent.strip().lower()
        df = df[df["intent"].str.lower() == normalized_intent]

    if text_query is not None:
        query = text_query.strip().lower()

        if query:
            mask = (
                df["instruction"].str.lower().str.contains(query, na=False, regex=False)
                | df["response"].str.lower().str.contains(query, na=False, regex=False)
                | df["category"].str.lower().str.contains(query, na=False, regex=False)
                | df["intent"].str.lower().str.contains(query, na=False, regex=False)
            )

            df = df[mask]

    return df


def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert a filtered DataFrame into JSON-friendly row dictionaries.
    """
    return df[OUTPUT_COLUMNS].to_dict("records")


def list_categories() -> list[str]:
    """
    Return all unique high-level categories in the dataset.
    """
    df = load_customer_service_df()
    return sorted(df["category"].dropna().unique().tolist())


def list_intents(category: str | None = None) -> list[str]:
    """
    Return all unique intents in the dataset.

    If category is provided, return only intents that belong to that category.
    """
    df = _filter_df(category=category)
    return sorted(df["intent"].dropna().unique().tolist())


def filter_rows(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
) -> list[dict[str, Any]]:
    """
    Filter dataset rows by category, intent, and/or free-text query.

    Use this tool when another operation needs a subset of rows first.
    For example:
    - filter REFUND rows, then count them
    - filter complaint rows, then summarize them
    - search for rows mentioning "money back"
    """
    df = _filter_df(
        category=category,
        intent=intent,
        text_query=text_query,
    )

    return _to_records(df)


def count_rows(
    rows: list[dict[str, Any]] | None = None,
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
) -> int:
    """
    Count rows in the dataset, optionally after filtering.
    """
    if rows is not None:
        return len(rows)

    df = _filter_df(
        category=category,
        intent=intent,
        text_query=text_query,
    )
    return len(df)


def show_examples(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
    n: int = 3,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    Show representative examples from the dataset.

    Use this for requests like:
    - show me 3 examples from SHIPPING
    - show me examples of refund requests
    - show me examples of people wanting their money back

    Use offset for follow-up requests like:
    - show me 3 more
    - show me more examples
    - give me the next 5
    """
    df = _filter_df(
        category=category,
        intent=intent,
        text_query=text_query,
    )

    return _to_records(df.iloc[offset : offset + n])


def distribution(
    group_by: Literal["category", "intent"],
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
) -> dict[str, int]:
    """
    Compute a frequency distribution by category or intent.

    Optionally filter first by category, intent, or text query.
    """
    df = _filter_df(
        category=category,
        intent=intent,
        text_query=text_query,
    )

    return df[group_by].value_counts().to_dict()


def get_rows_for_summary(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
    n: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    Return rows for open-ended summarization.

    Use this before answering qualitative questions such as:
    - summarize the FEEDBACK category
    - how do agents respond to cancellation requests?
    - summarize complaint intents
    """
    return show_examples(
        category=category,
        intent=intent,
        text_query=text_query,
        n=n,
        offset=offset,
    )


class ListCategoriesInput(BaseModel):
    """
    Input schema for list_categories.

    No parameters are required.
    """


class ListIntentsInput(BaseModel):
    category: str | None = Field(
        default=None,
        description=(
            "Optional high-level category to filter by, such as ACCOUNT, REFUND, "
            "SHIPPING, FEEDBACK, or CANCELLATION."
        ),
    )


class FilterRowsInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Optional high-level category filter, such as ACCOUNT, REFUND, SHIPPING, or FEEDBACK.",
    )
    intent: str | None = Field(
        default=None,
        description="Optional specific intent filter, such as get_refund or cancel_order.",
    )
    text_query: str | None = Field(
        default=None,
        description="Optional plain-text search over instructions, responses, categories, and intents.",
    )


class CountRowsInput(BaseModel):
    rows: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional rows returned by filter_rows. If provided, category, intent, "
            "and text_query are ignored."
        ),
    )
    category: str | None = Field(
        default=None,
        description="Optional high-level category filter, such as ACCOUNT, REFUND, SHIPPING, or FEEDBACK.",
    )
    intent: str | None = Field(
        default=None,
        description="Optional specific intent filter, such as get_refund or cancel_order.",
    )
    text_query: str | None = Field(
        default=None,
        description="Optional plain-text search over instructions, responses, categories, and intents.",
    )


class ShowExamplesInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Optional high-level category filter.",
    )
    intent: str | None = Field(
        default=None,
        description="Optional specific intent filter.",
    )
    text_query: str | None = Field(
        default=None,
        description="Optional text search query.",
    )
    n: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of examples to return. Must be between 1 and 20.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of matching examples to skip before returning results. "
            "Use this for follow-up requests like 'show me 3 more'. "
            "For example, if the previous call returned 3 examples, use offset=3."
        ),
    )


class DistributionInput(BaseModel):
    group_by: Literal["category", "intent"] = Field(
        description="Column to group by. Use 'category' or 'intent'.",
    )
    category: str | None = Field(
        default=None,
        description="Optional high-level category filter before calculating distribution.",
    )
    intent: str | None = Field(
        default=None,
        description="Optional intent filter before calculating distribution.",
    )
    text_query: str | None = Field(
        default=None,
        description="Optional text query filter before calculating distribution.",
    )


class GetRowsForSummaryInput(ShowExamplesInput):
    n: int = Field(
        default=50,
        ge=5,
        le=200,
        description="Maximum number of rows to return for summarization.",
    )


LANGCHAIN_TOOLS = [
    StructuredTool.from_function(
        func=list_categories,
        name="list_categories",
        description=(
            "List all high-level categories that exist in the customer service dataset. "
            "Use this when the user asks what categories exist."
        ),
        args_schema=ListCategoriesInput,
    ),
    StructuredTool.from_function(
        func=list_intents,
        name="list_intents",
        description=(
            "List all specific intents in the dataset. Optionally filter by a high-level category. "
            "Use this when the user asks what intents exist or what intents belong to a category."
        ),
        args_schema=ListIntentsInput,
    ),
    # StructuredTool.from_function(
    #     func=filter_rows,
    #     name="filter_rows",
    #     description=(
    #         "Filter dataset rows by category, intent, and/or text query. "
    #         "Use this as a first step before counting, comparing, or inspecting matching rows."
    #     ),
    #     args_schema=FilterRowsInput,
    # ),
    StructuredTool.from_function(
        func=count_rows,
        name="count_rows",
        description=(
            "Count rows in the dataset, optionally filtered by category, intent, or text query. "
            "Use this directly for any 'how many' or count question. "
            "For example, for 'How many refund requests did we get?', call count_rows with category='REFUND'. "
            "Do not use distribution when the user only asks for a total count."
        ),
        args_schema=CountRowsInput,
    ),
    StructuredTool.from_function(
        func=show_examples,
        name="show_examples",
        description=(
            "Show a small number of example customer instructions and agent responses from the dataset. "
            "Use this when the user asks for examples or samples. "
            "For follow-up requests like 'show me 3 more', reuse the previous filters "
            "and set offset to the number of examples already shown."
        ),
        args_schema=ShowExamplesInput,
    ),
    StructuredTool.from_function(
        func=distribution,
        name="distribution",
        description=(
            "Compute a frequency distribution by category or intent. "
            "Use this for questions like 'What is the distribution of intents in ACCOUNT?'."
        ),
        args_schema=DistributionInput,
    ),
    StructuredTool.from_function(
        func=get_rows_for_summary,
        name="get_rows_for_summary",
        description=(
            "Fetch relevant dataset rows for open-ended summarization. "
            "Use this before summarizing a category, intent, or customer-service behavior pattern."
        ),
        args_schema=GetRowsForSummaryInput,
    ),
]
