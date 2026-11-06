"""Measure search_knowledge latency and test argument-omission failure mode.

Reproduces the agent's exact client config: MCPServerHTTP defaults give
ClientSession a 5-second read timeout (client_session_timeout_seconds=5).
"""
import asyncio
import time
from datetime import timedelta

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = "http://192.168.1.51:5080/mcp"


async def timed_call(session: ClientSession, label: str, arguments: dict) -> None:
    started = time.perf_counter()
    try:
        result = await session.call_tool("search_knowledge", arguments)
        elapsed = (time.perf_counter() - started) * 1000
        first = result.content[0].text[:60] if result.content else ""
        print(f"{label}: {elapsed:.0f} ms, isError={result.isError}, first={first!r}")
    except Exception as e:
        elapsed = (time.perf_counter() - started) * 1000
        print(f"{label}: {elapsed:.0f} ms, EXCEPTION {type(e).__name__}: {e}")


async def main() -> None:
    # agent-equivalent config: 5 s read timeout (MCPServerHTTP default)
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=5)
        ) as session:
            await session.initialize()

            # 1) what the LLM is most likely to send: only the query
            await timed_call(session, "query only          ", {"query": "T14中的USB"})
            # 2) full args, first call (may hit cold embedding)
            await timed_call(
                session,
                "full args           ",
                {"query": "T14中的USB", "knowledgeBaseIds": None, "topK": 3},
            )
            # 3) repeat to see warm latency
            await timed_call(
                session,
                "full args (warm)    ",
                {"query": "T14中的USB", "knowledgeBaseIds": None, "topK": 3},
            )


if __name__ == "__main__":
    asyncio.run(main())
