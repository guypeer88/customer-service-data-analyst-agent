import asyncio

from fastmcp import Client


async def main() -> None:
    client = Client("mcp_server.py")

    async with client:
        print("\nAvailable tools:")
        tools = await client.list_tools()

        for tool in tools:
            print("-", tool.name)

        print("\nCalling list_categories_tool:")
        result = await client.call_tool("list_categories_tool", {})
        print(result.data)

        print("\nCalling list_intents_tool:")
        result = await client.call_tool(
            "list_intents_tool",
            {
                "category": "REFUND",
            },
        )
        print(result.data)

        print("\nCalling distribution_tool:")
        result = await client.call_tool(
            "distribution_tool",
            {
                "group_by": "intent",
                "category": "ACCOUNT",
            },
        )
        print(result.data)

        print("\nCalling show_examples_tool:")
        result = await client.call_tool(
            "show_examples_tool",
            {
                "category": "SHIPPING",
                "n": 2,
                "offset": 0,
            },
        )
        print(result.data)


if __name__ == "__main__":
    asyncio.run(main())