"""High-fidelity reproduction of the agent's LLM tool-calling context.

Uses the FULL system prompt from src/agent.py and the complete tool list
(5 MCP tools + lookup_weather + play_audio_file), then asks the LLM the
user's query several times to observe what arguments it actually generates,
and feeds each attempt into the real MCP server.
"""
import asyncio
import json
import os

import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"))

MCP_URL = "http://192.168.1.51:5080/mcp"
# defaults: the litellm proxy the agent uses; override via env to test other models
LLM_URL = os.getenv("TEST_LLM_URL", "http://192.168.1.51:4000/v1/chat/completions")
LLM_KEY = os.getenv("TEST_LLM_KEY", "sk-local-litellm-change-this")
LLM_MODEL = os.getenv("TEST_LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct")
USER_QUERY = os.getenv("TEST_USER_QUERY", "T14中的USB")
TRIALS = int(os.getenv("TEST_TRIALS", "3"))

# load the real instructions from src/agent.py so the test never drifts
def load_agent_prompt() -> str:
    import re
    import textwrap
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "agent.py"
    text = src.read_text(encoding="utf-8")
    start = text.index('"""', text.index("instructions=textwrap.dedent("))
    start = text.index("\n", start) + 1
    end = text.index('"""', start)
    return textwrap.dedent(text[start:end]).strip()

LOCAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Use this tool to look up current weather information in the given location. If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to look up weather information for (e.g. city name)",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_audio_file",
            "description": "Play the built-in audio clip to the user. This tool takes no arguments: the audio file is fixed and bundled with the application, so never ask the user for a file name or path. Call this tool immediately whenever the user asks to play audio.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


async def get_mcp_tools() -> list[dict]:
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                }
                for t in (await session.list_tools()).tools
            ]


async def ask_llm(tools: list[dict], system_prompt: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            LLM_URL,
            headers={"Authorization": f"Bearer {LLM_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": USER_QUERY},
                ],
                "tools": tools,
                "tool_choice": "auto",
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]


async def call_mcp(arguments: dict) -> tuple[bool, str]:
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("search_knowledge", arguments)
            first = result.content[0].text[:80] if result.content else ""
            return result.isError, first


async def main() -> None:
    system_prompt = load_agent_prompt()
    tools = await get_mcp_tools() + LOCAL_TOOLS
    print(f"endpoint: {LLM_URL}  model: {LLM_MODEL}  query: {USER_QUERY!r}")
    print(f"total tools sent to LLM: {len(tools)}")

    for i in range(1, TRIALS + 1):
        message = await ask_llm(tools, system_prompt)
        tool_calls = message.get("tool_calls") or []
        print(f"\n=== trial {i} ===")
        if not tool_calls:
            print(f"NO tool call, plain text reply: {(message.get('content') or '')[:120]!r}")
            continue
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                print(f"{name}: INVALID JSON arguments: {tc['function']['arguments']!r}")
                continue
            print(f"{name} args: {json.dumps(args, ensure_ascii=False)}")
            if name == "search_knowledge":
                is_error, first = await call_mcp(args)
                print(f"  -> MCP isError={is_error}, first={first!r}")


if __name__ == "__main__":
    asyncio.run(main())
