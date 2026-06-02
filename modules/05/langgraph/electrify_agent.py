# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage, AIMessage
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from common.agent import BaseAgent
from common.types import AgentConfig, StdioServerConfig, HttpsServerConfig
from common.cli import agent_cli_runner
from common.sanitize import sanitize_messages_middleware, sanitize_tool_output


@tool(parse_docstring=True)
def getDateTime() -> str:
    """Get the date and time in the local timezone.

    Returns:
        str: The current date and time in ISO format
    """
    return str(datetime.now().astimezone().isoformat())


class ElectrifyAgent(BaseAgent):
    async def setup(self):
        self.logger.info("Setting up agent...")

        # Initialize LLM
        self.llm = init_chat_model(
            model=self.config.model,
            model_provider='bedrock_converse',
            region_name=os.getenv('MODEL_REGION', os.getenv('AWS_REGION', 'us-east-1'))
        )
        self.logger.info("LLM initialized")

        # Initialize MCP client with both stdio and HTTPS servers
        mcp_config = {}

        # Add stdio servers
        for s in self.config.stdio_servers:
            mcp_config[s.name] = {
                "command": s.script,
                "args": s.args,
                "transport": "stdio"
            }
            self.logger.info(f"Configured stdio MCP server: {s.name}")

        # Add HTTPS servers (e.g., AgentCore Gateway) with identity propagation
        for s in self.config.https_servers:
            headers = dict(s.headers) if s.headers else {}

            if s.propagate_identity and self.config.identity_context:
                if self.config.identity_context.jwt_token:
                    headers["Authorization"] = f"Bearer {self.config.identity_context.jwt_token}"
                headers["X-User-Id"] = self.config.identity_context.sub
                headers["X-Username"] = self.config.identity_context.username
                if self.config.identity_context.email:
                    headers["X-User-Email"] = self.config.identity_context.email
                if self.config.identity_context.groups:
                    headers["X-User-Groups"] = ','.join(self.config.identity_context.groups)
                self.logger.info(f"Identity propagated to {s.name}: {self.config.identity_context.username}")

            mcp_config[s.name] = {
                "url": s.url,
                "transport": "streamable_http",
                "headers": headers,
                "timeout": s.timeout
            }
            self.logger.info(f"Configured HTTPS MCP server: {s.name} at {s.url}")

        if not mcp_config:
            self.logger.warning("No MCP servers configured")
            mcp_tools = []
        else:
            self.mcp = MultiServerMCPClient(mcp_config)
            mcp_tools = await self.mcp.get_tools()
            self.logger.info(f"Retrieved {len(mcp_tools)} MCP tools from {len(mcp_config)} server(s)")

        # MCP tools are used directly — sanitize_tool_output middleware handles
        # content format normalization, so wrap_tool_for_bedrock is not needed
        self.all_tools = list(mcp_tools)
        self.all_tools.append(getDateTime)

        # Child agents called as tools don't need persistent memory
        checkpointer = InMemorySaver()
        self.logger.info("Using InMemorySaver for stateless tool invocations")

        # Inject user identity into system prompt
        identity_info = ""
        if self.config.identity_context:
            identity_info = f"\n\nCurrent user identity: {self.config.identity_context.username}\nWhen calling tools that require customer_username, always use: {self.config.identity_context.username}"
        elif self.config.identity:
            identity_info = f"\n\nCurrent user identity: {self.config.identity}\nWhen calling tools that require customer_username, always use: {self.config.identity}"

        enhanced_prompt = self.system_prompt + identity_info if self.system_prompt else identity_info

        # Create agent with message sanitization middleware
        self.agent = create_agent(
            model=self.llm,
            tools=self.all_tools,
            system_prompt=enhanced_prompt,
            checkpointer=checkpointer,
            middleware=[sanitize_messages_middleware, sanitize_tool_output]
        )
        self.logger.info("Agent setup complete")


# CLI interface
def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user', default=os.getenv('USER', 'unknown'))
    parser.add_argument('-t', '--thread', default=None)
    parser.add_argument('-p', '--system-prompt', default="./system_prompt.md")
    parser.add_argument('-s', '--server-script')
    parser.add_argument('-a', '--server-args', nargs='*')
    parser.add_argument('--https-url')
    parser.add_argument('--https-headers')
    parser.add_argument('-r', '--region', default=os.getenv("AWS_REGION", "us-east-1"))
    parser.add_argument('-m', '--model', default="global.anthropic.claude-sonnet-4-6")
    args = parser.parse_args()

    # Configure stdio servers
    stdio_servers = []
    if args.server_script or args.server_args:
        import shlex
        if args.server_args and len(args.server_args) == 1:
            params = shlex.split(args.server_args[0])
        elif args.server_args and len(args.server_args) > 1:
            params = args.server_args
        else:
            params = []
        stdio_servers.append(StdioServerConfig(
            name="electrify_mcp_server", script="uv", args=params
        ))

    # Configure HTTPS servers
    https_servers = []
    if args.https_url:
        import json
        headers = {}
        if args.https_headers:
            try:
                headers = json.loads(args.https_headers)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON for headers: {args.https_headers}")
        https_servers.append(HttpsServerConfig(
            name="electrify_gateway", url=args.https_url, headers=headers
        ))

    config = AgentConfig(
        name="electrify_agent",
        description="This agent enables users to explore their electricity usage, rate plans and programs.",
        identity=args.user,
        thread=args.thread if args.thread else args.user,
        log_file="electrify_agent.log",
        model=args.model,
        region=args.region,
        stdio_servers=stdio_servers,
        https_servers=https_servers
    )

    asyncio.run(agent_cli_runner(ElectrifyAgent(config)))


if __name__ == "__main__":
    main()