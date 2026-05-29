# Customer Service Data Analyst Agent

Assignment 3 - Customer Service Data Analyst Agent  
Student: **Guy Peer**

## What This Project Does

This repository contains a LangGraph ReAct agent that analyzes the Bitext Customer Service dataset. It supports:

- structured questions, such as counts, categories, examples, and distributions
- open-ended summarization questions about the dataset
- out-of-scope refusal for unrelated questions
- persistent conversation memory with Postgres-backed LangGraph checkpoints
- separate persistent user profile memory
- FastMCP access to dataset tools
- bonus Streamlit chat UI
- bonus query recommender

## Model Choice

The project uses this Nebius Token Factory model through an OpenAI-compatible API:

```text
meta-llama/Llama-3.3-70B-Instruct-fast
```

It is used for routing, agent responses, profile updates, and query recommendations. I chose it because it is strong enough for tool use and summarization while still being fast enough for local CLI/UI testing. The code builds the model in `app/utils.py`, so different Nebius models could be assigned to different roles later if needed.

## Setup

Requirements:

- Python 3.11 or 3.12
- `uv`
- Docker and Docker Compose
- Nebius Token Factory API key

Install dependencies:

```bash
uv sync
```

Create `.env`:

```bash
cp .env.example .env
```

Fill in your Nebius key:

```env
NEBIUS_API_KEY=your_nebius_api_key_here
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1
NEBIUS_MODEL=meta-llama/Llama-3.3-70B-Instruct-fast

DATASET_NAME=bitext/Bitext-customer-support-llm-chatbot-training-dataset
DATABASE_URL=postgresql://csda_langgraph:csda_langgraph_password@localhost:5432/csda_checkpoints
```

Start Postgres:

```bash
docker compose up -d
```

The dataset is loaded from Hugging Face on first use and cached locally in `data/bitext.csv`.

## Run the CLI

```bash
uv run python main.py --session demo --user-id guy
```

CLI arguments:

- `--session`: conversation memory ID. Reuse it to resume the same checkpointed conversation.
- `--user-id`: persistent user profile ID. If omitted, it defaults to the session ID.
- `--hide-reasoning`: hides printed tool calls and observations.

Try these assignment-style queries:

```text
What categories exist in the dataset?
How many refund requests did we get?
Show me 5 examples of the SHIPPING category.
Summarize how agents respond to complaint intents.
Show me examples of people wanting their money back.
What is the distribution of intents in the ACCOUNT category?
What's the best CRM software for handling complaints?
Who is the president of France?
```

The CLI prints the route, route reason, visible tool calls, tool observations, and final answer.

## Run the Streamlit UI

```bash
uv run streamlit run streamlit_app.py
```

The sidebar includes:

- session ID input for switching or resuming conversations
- user ID input for switching profile memory
- reasoning toggle for showing tool calls and results

## Query Recommender Bonus

Ask:

```text
What should I query next?
```

The agent uses recent conversation memory and the saved user profile to suggest one next dataset query. It does not execute the suggestion until the user confirms.

Example:

```text
User: What should I query next?
Agent: I suggest: What is the distribution of intents in the REFUND category?

User: I'd rather see examples instead.
Agent: I suggest: Show me 5 examples from the REFUND category.

User: Yes, do it.
Agent: Executes the saved query and shows the result.
```

## Architecture

```text
User query
  |
  v
Query router
  |
  |-- out_of_scope ------------> polite refusal
  |-- profile_read ------------> read saved user profile
  |-- profile_update ----------> update saved user profile
  |-- recommendation_request --> suggest a next query
  |-- recommendation_refine ---> revise pending suggestion
  |-- recommendation_cancel ---> cancel pending suggestion
  |-- recommendation_confirm --> execute pending suggestion
  |-- structured/unstructured -> LangGraph ReAct agent
                                      |
                                      v
                                 Dataset tools
                                      |
                                      v
                                  Final answer
```

Memory:

- conversation memory is stored with LangGraph Postgres checkpoints and keyed by `--session`
- user profile memory is stored separately under `.profiles/` and keyed by `--user-id`
- pending query recommendations are stored under `.recommendations/`

## Tools Defined

The agent tools are intentionally bounded so the model does not receive large raw dataset dumps.

- `list_categories`: returns all dataset categories
- `list_intents`: returns all intents, optionally filtered by category
- `count_rows`: counts rows, optionally filtered by category, intent, or text query
- `show_examples`: returns a small set of examples with optional pagination
- `distribution`: returns a frequency distribution by category or intent
- `get_rows_for_summary`: returns a bounded sample for summarization

All tools have Pydantic input schemas in `app/tools.py`.

## MCP Server

Start the FastMCP server:

```bash
uv run python mcp_server.py
```

Exposed MCP tools:

- `list_categories_tool`
- `list_intents_tool`
- `distribution_tool`
- `show_examples_tool`

Client example:

```python
import asyncio
from fastmcp import Client

async def main() -> None:
    client = Client("mcp_server.py")

    async with client:
        tools = await client.list_tools()
        print([tool.name for tool in tools])

        result = await client.call_tool("list_categories_tool", {})
        print(result.data)

asyncio.run(main())
```

Run the included client check:

```bash
uv run python -m checks.check_fastmcp
```

## Checks

Basic local checks:

```bash
uv run python -m compileall app checks main.py mcp_server.py streamlit_app.py
uv run python -m checks.check_tools
uv run python -m checks.check_langchain_tools
uv run python -m checks.check_fastmcp
```

LLM and memory checks:

```bash
docker compose up -d
uv run python -m checks.check_router
uv run python -m checks.check_agent
uv run python -m checks.check_memory
uv run python -m checks.check_profile
uv run python -m checks.check_recommendations
```

## Assignment Coverage

- [x] LangGraph ReAct agent
- [x] query router before tool selection
- [x] structured dataset questions
- [x] unstructured summarization questions
- [x] out-of-scope refusal
- [x] tools with clear descriptions and Pydantic schemas
- [x] multi-step tool use
- [x] CLI interface with visible tool calls and observations
- [x] max iteration limit with graceful fallback
- [x] persistent conversation memory with session ID support
- [x] separate persistent user profile memory
- [x] FastMCP server with at least 3 exposed tools
- [x] MCP client example
- [x] bonus Streamlit UI
- [x] bonus query recommender

## Submission Notes

- Make sure Docker is running before testing memory-related features.
