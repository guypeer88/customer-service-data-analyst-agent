from app.tools import LANGCHAIN_TOOLS


def main() -> None:
    print("\nAvailable LangChain tools:")
    for tool in LANGCHAIN_TOOLS:
        print(f"- {tool.name}")
        print(f"  Description: {tool.description[:120]}...")

    print("\nInvoke list_categories:")
    list_categories_tool = next(
        tool for tool in LANGCHAIN_TOOLS if tool.name == "list_categories"
    )
    print(list_categories_tool.invoke({}))

    print("\nInvoke show_examples:")
    show_examples_tool = next(
        tool for tool in LANGCHAIN_TOOLS if tool.name == "show_examples"
    )
    result = show_examples_tool.invoke(
        {
            "category": "REFUND",
            "n": 2,
        }
    )
    print(result)

    print("\nInvoke count_rows with category filter:")
    count_rows_tool = next(tool for tool in LANGCHAIN_TOOLS if tool.name == "count_rows")
    print(count_rows_tool.invoke({"category": "REFUND"}))

    print("\nInvoke distribution:")
    distribution_tool = next(
        tool for tool in LANGCHAIN_TOOLS if tool.name == "distribution")
    result = distribution_tool.invoke(
        {
            "group_by": "intent",
            "category": "ACCOUNT",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
