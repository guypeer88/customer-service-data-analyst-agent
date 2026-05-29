from app.tools import (
    count_rows,
    distribution,
    filter_rows,
    get_rows_for_summary,
    list_categories,
    list_intents,
    show_examples,
)


def print_examples(examples: list[dict], max_response_chars: int = 250) -> None:
    for index, example in enumerate(examples, start=1):
        print(f"\nExample {index}")
        print("Category:", example["category"])
        print("Intent:", example["intent"])
        print("Instruction:", example["instruction"])
        print("Response:", example["response"][:max_response_chars])


def main() -> None:
    print("\n1. Categories:")
    categories = list_categories()
    print(categories)

    print("\n2. First category intents:")
    first_category = categories[0]
    print("Category:", first_category)
    print(list_intents(category=first_category)[:10])

    print("\n3. Count all rows:")
    print(count_rows())

    print("\n4. Filter REFUND rows and count:")
    refund_rows = filter_rows(category="REFUND")
    print(count_rows(refund_rows))

    print("\n5. Show 3 REFUND examples:")
    print_examples(show_examples(category="REFUND", n=3))

    print("\n6. Show 3 examples matching text query 'money back':")
    print_examples(show_examples(text_query="money back", n=3))

    print("\n7. Intent distribution in ACCOUNT:")
    print(distribution(group_by="intent", category="ACCOUNT"))

    print("\n8. Rows for summary from FEEDBACK:")
    summary_rows = get_rows_for_summary(category="FEEDBACK", n=5)
    print_examples(summary_rows)


if __name__ == "__main__":
    main()