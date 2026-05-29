from app.router import route_query


TEST_QUERIES = [
    "What categories exist in the dataset?",
    "How many refund requests did we get?",
    "Show me 5 examples from the SHIPPING category.",
    "What is the distribution of intents in the ACCOUNT category?",
    "Summarize the FEEDBACK category.",
    "How do customer service representatives typically respond to cancellation requests?",
    "Show me examples of people wanting their money back.",
    "What's the best CRM software for handling complaints?",
    "Who is the president of France?",
    "Write me a poem about customer service.",
    "What do you remember about me?",
    "What should I query next?",
    "Show me 3 more.",
    "What about refunds?",
    "What is the total count of the last two?",
]


def main() -> None:
    for query in TEST_QUERIES:
        decision = route_query(query)

        print("=" * 100)
        print("Query:", query)
        print("Route:", decision.route)
        print("Reason:", decision.reason)


if __name__ == "__main__":
    main()
