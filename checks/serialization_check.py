import json

from langchain_core.messages import AIMessage, HumanMessage, messages_from_dict, messages_to_dict


def main() -> None:
    original_messages = [
        HumanMessage(content="How many refund requests?"),
        AIMessage(content="I will check the dataset."),
    ]

    print("\n1. Original Python objects:")
    print(original_messages)
    print(type(original_messages[0]))

    as_dict = messages_to_dict(original_messages)

    print("\n2. Serialized to dict/list:")
    print(as_dict)
    print(type(as_dict))

    as_json = json.dumps(as_dict)

    print("\n3. Serialized to JSON string:")
    print(as_json)
    print(type(as_json))

    as_bytes = as_json.encode("utf-8")

    print("\n4. Serialized to bytes, similar to BYTEA:")
    print(as_bytes[:120], b"...")
    print(type(as_bytes))

    json_back = as_bytes.decode("utf-8")
    dict_back = json.loads(json_back)
    restored_messages = messages_from_dict(dict_back)

    print("\n5. Deserialized back to Python message objects:")
    print(restored_messages)
    print(type(restored_messages[0]))

    print("\n6. Same content?")
    print(restored_messages[0].content == original_messages[0].content)


if __name__ == "__main__":
    main()