# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
import os
import sys
import re
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import asyncio
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage, ToolMessage, AIMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_checkpoint_aws import AgentCoreMemorySaver

from common.agent import BaseAgent
from common.types import AgentConfig, HttpsServerConfig
from common.cli import agent_cli_runner
from common.prompts import electrify_prompt, dataviz_prompt
from common.sanitize import sanitize_messages_middleware, sanitize_tool_output


class OrchestratorAgent(BaseAgent):
    async def setup(self):
        self.logger.info("Setting up agent...")

        # Initialize LLM
        self.llm = init_chat_model(
            model=self.config.model,
            model_provider='bedrock_converse',
            region_name=self.config.region if self.config.region else os.getenv('MODEL_REGION', os.getenv('AWS_REGION', 'us-east-1'))
        )
        self.logger.info("LLM initialized")

        # Initialize child agents
        self.dataviz_agent = None
        self.electrify_agent = None

        from dataviz_agent import DataVizAgent
        from electrify_agent import ElectrifyAgent

        # Build auth headers for gateway calls
        headers = {}
        if self.config.identity_context and self.config.identity_context.jwt_token:
            headers["Authorization"] = f"Bearer {self.config.identity_context.jwt_token}"

        # Get gateway URL - supports single unified gateway or separate gateways
        mcp_gateway_url = os.getenv("MCP_GATEWAY_URL")
        dataviz_gateway_url = os.getenv("DATAVIZ_MCP_SERVER_URL") or mcp_gateway_url
        electrify_gateway_url = os.getenv("ELECTRIFY_MCP_SERVER_URL") or mcp_gateway_url

        # Initialize DataViz Agent via HTTPS gateway
        if dataviz_gateway_url:
            self.logger.info(f"Using DataViz Gateway at {dataviz_gateway_url}")
            self.dataviz_agent = DataVizAgent(AgentConfig(
                name="dataviz_agent",
                description="This agent creates charts and visualizations from CSV data.",
                identity=self.config.identity,
                thread=f"{self.config.thread if self.config.thread else self.config.identity}_dataviz",
                system_prompt=dataviz_prompt(),
                model=self.config.model,
                identity_context=self.config.identity_context,
                https_servers=[HttpsServerConfig(
                    name="dataviz_gateway",
                    url=dataviz_gateway_url,
                    headers=headers,
                    propagate_identity=False
                )]
            ))
        else:
            self.logger.warning("No gateway URL for DataViz - DataViz will not be available")

        if self.dataviz_agent:
            await self.dataviz_agent.setup()
            self.logger.info("DataViz agent initialized via gateway")

        # Initialize Electrify Agent via HTTPS gateway
        if electrify_gateway_url:
            self.logger.info(f"Using Electrify Gateway at {electrify_gateway_url}")
            self.electrify_agent = ElectrifyAgent(AgentConfig(
                name="electrify_agent",
                description="This agent enables users to explore their electricity usage, rate plans and programs.",
                identity=self.config.identity,
                thread=f"{self.config.thread if self.config.thread else self.config.identity}_electrify",
                system_prompt=electrify_prompt(),
                model=self.config.model,
                identity_context=self.config.identity_context,
                https_servers=[HttpsServerConfig(
                    name="electrify_gateway",
                    url=electrify_gateway_url,
                    headers=headers,
                    propagate_identity=False
                )]
            ))
        else:
            self.logger.warning("No gateway URL for Electrify - Electrify will not be available")

        if self.electrify_agent:
            await self.electrify_agent.setup()
            self.logger.info("Electrify agent initialized via gateway")

        # Define orchestrator tools (only add tools for available agents)
        self.all_tools = [self.create_datetime_tool(), self.create_json_to_csv_tool()]
        if self.dataviz_agent:
            self.all_tools.append(self.create_dataviz_tool())
        if self.electrify_agent:
            self.all_tools.append(self.create_electrify_tool())

        if self.config.memory:
            # Use AgentCore Memory for persistent checkpointing
            checkpointer = AgentCoreMemorySaver(
                memory_id=self.config.memory,
                region_name=self.config.region if self.config.region else os.getenv('AWS_REGION', 'us-east-1')
            )
            self.logger.info(f"Using AgentCore Memory checkpointer (Memory ID: {self.config.memory})")
        else:
            checkpointer = InMemorySaver()
            self.logger.warning("Memory ID not provided, using InMemorySaver")

        # Inject user identity into system prompt
        identity_info = ""
        if self.config.identity_context:
            identity_info = f"\n\nCurrent user identity: {self.config.identity_context.username}\nWhen calling tools that require customer_username, use: {self.config.identity_context.username}"
        elif self.config.identity:
            identity_info = f"\n\nCurrent user identity: {self.config.identity}\nWhen calling tools that require customer_username, use: {self.config.identity}"

        enhanced_prompt = self.system_prompt + identity_info if self.system_prompt else identity_info

        # Create agent with message sanitization middleware
        self.agent = create_agent(
            model=self.llm,
            tools=self.all_tools,
            system_prompt=enhanced_prompt,
            checkpointer=checkpointer,
            middleware=[sanitize_messages_middleware, sanitize_tool_output]
        )

        self.logger.info("Orchestrator agent setup complete")

    def create_dataviz_tool(self):
        """Create a tool that wraps the DataViz agent."""
        call_count = {"n": 0}
        _dataviz_lock = asyncio.Lock()

        @tool(parse_docstring=True)
        async def use_dataviz_agent(data: str, description: str) -> str:
            """Create charts and visualizations from CSV formatted data. A
            variety of charts are supported including, bar, line, pie charts and more

            Args:
                data (str): The CSV formatted data as text
                description (str): Describe how to visualize the data

            Returns:
                str: The base64 data url encoded image

            Raises:
                Exception: If the underlying agent is not available
            """
            call_count["n"] += 1
            if call_count["n"] > 1:
                return "Chart already created"
            try:
                async with _dataviz_lock:
                    result = await self.dataviz_agent.visualize_data(data, description)
                return result
            except Exception as e:
                return f"Error using DataViz agent: {str(e)}"

        return use_dataviz_agent

    def create_electrify_tool(self):
        """Create a tool that wraps the Electrify agent."""
        _electrify_lock = asyncio.Lock()

        @tool(parse_docstring=True)
        async def use_electrify_agent(query: str) -> str:
            """Retrieve customer data about utilization, billing,
            rates and other data associated with the electricity
            utility account of a consumer customer.

            Args:
                query (str): The natural language data query

            Returns:
                str: The dataset requested, if available.

            Raises:
                Exception: If the underlying agent is not available
            """
            try:
                async with _electrify_lock:
                    result = await self.electrify_agent.invoke(query, use_fresh_thread=True)
                if isinstance(result, dict):
                    return result.get('text', str(result))
                return str(result)
            except Exception as e:
                return f"Error using Electrify agent: {str(e)}"

        return use_electrify_agent

    def create_datetime_tool(self):
        """Create a tool for getting current date/time."""
        @tool(parse_docstring=True)
        def get_current_datetime() -> str:
            """Get the date and time in the local timezone.

            Returns:
                str: The current date and time in ISO format
            """
            return str(datetime.now().astimezone().isoformat())

        return get_current_datetime

    def create_json_to_csv_tool(self):
        """Create a tool for converting JSON data to CSV format."""
        @tool(parse_docstring=True)
        def convert_json_to_csv(json_data: str) -> str:
            """Converts a JSON data structure supplied as text, into
            a tabular CSV format, represented as text with row values
            separated by commas.

            Args:
                json_data (str): The text of the input JSON data structure

            Returns:
                str: The CSV formatted text output

            Raises:
                Exception: If the data is not suitable for conversion.
            """
            try:
                import json
                import pandas as pd
                from io import StringIO

                data = json.loads(json_data)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    return f"Error: Unsupported data format for conversion"

                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                return csv_buffer.getvalue()

            except Exception as e:
                return f"Error converting JSON to CSV: {str(e)}"

        return convert_json_to_csv


# CLI interface
def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user', default=os.getenv('USER', 'unknown'))
    parser.add_argument('-t', '--thread', default=None)
    parser.add_argument('-p', '--system-prompt', default="./system_prompt.md")
    parser.add_argument('-m', '--model', default="global.anthropic.claude-sonnet-4-6")
    parser.add_argument('-r', '--region', default=os.getenv("AWS_REGION", "us-east-1"))
    parser.add_argument('--memory-id', default=None)
    args = parser.parse_args()

    config = AgentConfig(
        name="orchestrator_agent",
        description="This agent reasons and decides what downstream tools or agents to invoke to complete the user request.",
        identity=args.user,
        thread=args.thread if args.thread else args.user,
        log_file="orchestrator_agent.log",
        model=args.model,
        region=args.region,
        memory=args.memory_id,
        stdio_servers=[]
    )

    asyncio.run(agent_cli_runner(OrchestratorAgent(config)))


if __name__ == "__main__":
    main()