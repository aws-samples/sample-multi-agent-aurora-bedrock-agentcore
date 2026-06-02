# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
DataViz Agent - Calls DataViz MCP server via HTTPS gateway.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import asyncio
import logging
import json
from datetime import datetime
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage, AIMessage
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from common.agent import BaseAgent
from common.types import AgentConfig, HttpsServerConfig
from common.cli import agent_cli_runner
from common.prompts import dataviz_prompt
from common.sanitize import sanitize_messages_middleware, sanitize_tool_output


@tool(parse_docstring=True)
def getDateTime() -> str:
    """Get the date and time in the local timezone.
    Returns:
        str: The current date and time in ISO format
    """
    return str(datetime.now().astimezone().isoformat())


class DataVizAgent(BaseAgent):
    async def setup(self):
        self.logger.info("Setting up DataViz agent...")

        self.llm = init_chat_model(
            model=self.config.model,
            model_provider='bedrock_converse',
            region_name=os.getenv('MODEL_REGION', os.getenv('AWS_REGION', 'us-east-1'))
        )
        self.logger.info("LLM initialized")

        # Initialize MCP client with HTTPS servers
        mcp_config = {}
        for s in self.config.https_servers:
            headers = dict(s.headers) if s.headers else {}
            if s.propagate_identity and self.config.identity_context and self.config.identity_context.jwt_token:
                headers["Authorization"] = f"Bearer {self.config.identity_context.jwt_token}"
                headers["X-User-Id"] = self.config.identity_context.sub
                headers["X-Username"] = self.config.identity_context.username
            mcp_config[s.name] = {
                "url": s.url,
                "transport": "streamable_http",
                "headers": headers,
                "timeout": s.timeout
            }
            self.logger.info(f"Configured HTTPS MCP server: {s.name} at {s.url}")

        if mcp_config:
            self.mcp = MultiServerMCPClient(mcp_config)
            mcp_tools = await self.mcp.get_tools()
            self.logger.info(f"Retrieved {len(mcp_tools)} MCP tools")
            # MCP tools used directly — sanitize_tool_output middleware handles formatting
            self.all_tools = list(mcp_tools)
        else:
            self.logger.warning("No MCP servers configured")
            self.all_tools = []

        self.all_tools.append(getDateTime)

        checkpointer = InMemorySaver()
        self.logger.info("Using InMemorySaver for conversation state")

        self.agent = create_agent(
            model=self.llm,
            tools=self.all_tools,
            system_prompt=self.system_prompt,
            checkpointer=checkpointer,
            middleware=[sanitize_messages_middleware, sanitize_tool_output]
        )
        self.logger.info("DataViz agent setup complete")

    async def visualize_data(self, data: str, description: str) -> str:
        """Create a visualization from data and description.

        Returns the LLM's text response with any captured chart images appended
        as `<chart>data:image/svg+xml;base64,...</chart>` blocks. The orchestrator
        adapter's stream loop matches that pattern on `on_tool_end` events and
        forwards the SVG payload to the runtime — without this append, the
        DataViz tool's chart never reaches the response and the UI sees an
        empty `images` field.
        """
        query = f"Here's my data:\n{data}\n\n{description}"
        result = await self.invoke(query, use_fresh_thread=True)
        if isinstance(result, dict):
            text = result.get('text', '')
            images = result.get('images', [])
            if images:
                chart_blocks = ''.join(f'<chart>{img}</chart>' for img in images)
                return f"{text}\n{chart_blocks}".strip()
            return text
        return str(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user', default=os.getenv('USER', 'unknown'))
    parser.add_argument('-t', '--thread', default=None)
    parser.add_argument('--https-url')
    parser.add_argument('-m', '--model', default="global.anthropic.claude-sonnet-4-6")
    args = parser.parse_args()

    https_servers = []
    if args.https_url:
        https_servers.append(HttpsServerConfig(name="dataviz_gateway", url=args.https_url))

    config = AgentConfig(
        name="dataviz_agent",
        description="This agent creates charts and visualizations from CSV data.",
        identity=args.user,
        thread=args.thread or args.user,
        system_prompt=dataviz_prompt(),
        log_file="dataviz_agent.log",
        model=args.model,
        https_servers=https_servers
    )
    asyncio.run(agent_cli_runner(DataVizAgent(config)))


if __name__ == "__main__":
    main()