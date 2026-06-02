# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
import sys
import traceback
import asyncio
from mcp.server.stdio import stdio_server

async def mcp_stdio_cli_runner(instance):
    try:
        # Run the server using stdio transport
        async with stdio_server() as (read_stream, write_stream):
            await instance.server.run(
                read_stream,
                write_stream,
                instance.server.create_initialization_options()
            )

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\nExiting stdio MCP server...")
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        sys.exit(1)


async def agent_cli_runner(agent):
    try:
        await agent.setup()
        await agent.chat_loop()

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\nExiting agent...")
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        sys.exit(1)