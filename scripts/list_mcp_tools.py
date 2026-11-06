"""Query the MCP server (Streamable HTTP): list tools, or call a tool.

Usage:
    list_mcp_tools.py                      # list tools
    list_mcp_tools.py search_knowledge '{"query": "T14中的USB"}'   # call a tool
"""
import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = "http://192.168.1.51:5080/mcp"


async def main() -> None:
    tool_name = sys.argv[1] if len(sys.argv) > 1 else None
    tool_args = {}
    if len(sys.argv) > 2:
        try:
            tool_args = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            # PowerShell strips quotes from inline JSON; treat raw text as the query
            tool_args = {"query": sys.argv[2]}

    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            init_result = await session.initialize()
            print(f"server: {init_result.serverInfo.name} {init_result.serverInfo.version}")

            if tool_name is None:
                result = await session.list_tools()
                print(f"\ntools ({len(result.tools)}):")
                for tool in result.tools:
                    print(f"\n- {tool.name}")
                    if tool.description:
                        print(f"  description: {tool.description}")
                    print(f"  inputSchema: {json.dumps(tool.inputSchema, ensure_ascii=False)}")
                return

            print(f"\ncalling {tool_name} with {json.dumps(tool_args, ensure_ascii=False)}")
            result = await session.call_tool(tool_name, tool_args)
            print(f"isError: {result.isError}")
            for part in result.content:
                if part.type == "text":
                    print(part.text)
                else:
                    print(f"[{part.type}] {part}")


if __name__ == "__main__":
    asyncio.run(main())
