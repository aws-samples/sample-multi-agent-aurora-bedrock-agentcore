#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
Orchestrator Agent - Strands SDK equivalent of the LangGraph orchestrator_agent.py.
Coordinates between DataViz and Electrify agents via HTTPS gateways.
"""

import os
import sys
import json
import argparse
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands.tools.mcp import MCPClient
from dataviz_agent import getDateTime as dataviz_getDateTime

# Configure logging
logger = logging.getLogger("orchestrator-agent")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('/tmp/orchestrator_agent.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

_ = load_dotenv('.env')


ORCHESTRATOR_SYSTEM_PROMPT = """You are an intelligent orchestrator agent that coordinates between two specialized agents:

1. **DataViz Agent**: Creates charts and visualizations from CSV data
   - Use when users want to create charts, graphs, or visualizations
   - Can create bar charts, line charts, scatter plots, pie charts
   - Requires CSV data as input

2. **Electrify Agent**: Retrieves data from an electricity company database
   - Use when users want to query customer information, bills, or rate plans
   - Use when users ask about "available plans", "rate plans", "pricing", "billing"
   - Can get customer profiles, billing history, and rate information
   - Returns data in JSON format that can be converted to CSV for visualization

Your role is to:
- Analyze user requests and determine which agent(s) to use
- Route simple requests to the appropriate single agent
- Chain operations when needed (e.g., get data from Electrify, then visualize with DataViz)
- Handle requests that don't require either agent with general assistance
- Provide clear, helpful responses

When chaining operations:
1. First use the Electrify agent to retrieve data
2. Convert the JSON response to CSV format if needed
3. Then use the DataViz agent to create visualizations

## SEQUENTIAL TOOL USE:
- Always call tools ONE AT A TIME. Never issue multiple tool calls in parallel.
- Wait for each tool result before deciding the next step.

## SINGLE CHART RULE:
- You MUST call use_dataviz_agent at most ONCE per user request.
- Never call use_dataviz_agent a second time to "refine" or "improve" a chart.
- If the first call succeeds, use that result as-is.

## CRITICAL RESPONSE GUIDELINES:

**Be Concise:**
- Give direct answers without narrating your process
- Do NOT say "Let me...", "I'll now...", "First I will..." - just do it
- Summarize data in 2-3 sentences, not lengthy tables
- For recommendations: recommend ONE best option with estimated savings

**Avoid Redundancy:**
- Create only ONE chart per request unless explicitly asked for multiple
- Do NOT create charts just to "analyze" data - only for final presentation
- Do NOT repeat data the user can already see in the UI
- Do NOT list every single item - summarize totals and trends

**Response Format:**
- Keep responses under 200 words unless complex analysis is requested
- Use bullet points for key facts
- End with a single clear recommendation or next step
"""


@dataclass
class OrchestratorConfig:
    """Configuration for the Orchestrator Agent."""
    model: str = "global.anthropic.claude-sonnet-4-6"
    user: str = "unknown"
    thread_id: Optional[str] = None
    region: str = "us-east-1"
    memory_id: Optional[str] = None
    identity: Optional[str] = None
    jwt_token: Optional[str] = None


class OrchestratorAgent:
    """Orchestrator Agent using Strands SDK with HTTPS gateway support."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.agent = None
        self.model = None
        self.dataviz_agent = None
        self.electrify_agent = None

        logger.info("Initializing Orchestrator Agent...")

    def setup(self):
        """Set up the orchestrator and initialize sub-agents."""
        logger.info("Setting up Orchestrator agent...")

        username = self.config.user
        model_id = self.config.model

        # Initialize Amazon Bedrock model
        self.model = BedrockModel(
            model_id=model_id,
            region_name=os.getenv('MODEL_REGION', os.getenv('AWS_REGION', 'us-east-1'))
        )
        logger.info("Amazon Bedrock model initialized")

        # Build auth headers for gateway calls
        headers = {}
        if self.config.jwt_token:
            headers["Authorization"] = f"Bearer {self.config.jwt_token}"

        # Get gateway URLs
        mcp_gateway_url = os.getenv("MCP_GATEWAY_URL")
        dataviz_gateway_url = os.getenv("DATAVIZ_MCP_SERVER_URL") or mcp_gateway_url
        electrify_gateway_url = os.getenv("ELECTRIFY_MCP_SERVER_URL") or mcp_gateway_url

        # Initialize DataViz Agent via HTTPS gateway
        if dataviz_gateway_url:
            from dataviz_agent import DataVizAgent, DataVizConfig
            dataviz_config = DataVizConfig(
                model=model_id,
                user=username,
                thread_id=f"{self.config.thread_id or username}_dataviz",
                https_url=dataviz_gateway_url,
                https_headers=headers
            )
            self.dataviz_agent = DataVizAgent(dataviz_config)
            self.dataviz_agent.setup()
            logger.info(f"DataViz agent initialized via gateway at {dataviz_gateway_url}")
        else:
            logger.warning("No gateway URL for DataViz - DataViz will not be available")

        # Initialize Electrify Agent via HTTPS gateway
        if electrify_gateway_url:
            from electrify_agent import Application as ElectrifyApp
            electrify_args = argparse.Namespace(
                user=username,
                thread=f"{self.config.thread_id or username}_electrify",
                system_prompt="./electrify_prompt.md",
                server_script=None,
                server_args=None,
                https_url=electrify_gateway_url,
                https_headers=json.dumps(headers) if headers else None,
                region=self.config.region,
                model=model_id
            )
            self.electrify_agent = ElectrifyApp(electrify_args)
            self.electrify_agent.setup()
            logger.info(f"Electrify agent initialized via gateway at {electrify_gateway_url}")
        else:
            logger.warning("No gateway URL for Electrify - Electrify will not be available")

        # Inject user identity into system prompt
        identity_info = ""
        identity = self.config.identity or self.config.user
        if identity:
            identity_info = f"\n\nCurrent user identity: {identity}\nWhen calling tools that require customer_username, use: {identity}"

        self.system_prompt = ORCHESTRATOR_SYSTEM_PROMPT + identity_info

        logger.info("Orchestrator agent setup complete")

    def _create_tools(self):
        """Create orchestrator tools that wrap sub-agents."""
        dataviz = self.dataviz_agent
        electrify = self.electrify_agent
        call_count = {"n": 0}
        _electrify_lock = threading.Lock()
        _dataviz_lock = threading.Lock()

        tools = []

        @tool
        def get_current_datetime() -> str:
            """Get the current date and time in ISO format."""
            return str(datetime.now().astimezone().isoformat())
        tools.append(get_current_datetime)

        @tool
        def convert_json_to_csv(json_data: str) -> str:
            """Convert JSON data to CSV format for visualization.

            Args:
                json_data: The text of the input JSON data structure
            """
            try:
                import pandas as pd
                from io import StringIO

                data = json.loads(json_data)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    return "Error: Unsupported data format for conversion"

                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                return csv_buffer.getvalue()
            except Exception as e:
                return f"Error converting JSON to CSV: {str(e)}"
        tools.append(convert_json_to_csv)

        if dataviz:
            @tool
            def use_dataviz_agent(data: str, description: str) -> str:
                """Create charts and visualizations from CSV formatted data.

                Args:
                    data: The CSV formatted data as text
                    description: Describe how to visualize the data
                """
                call_count["n"] += 1
                if call_count["n"] > 1:
                    return "Chart already created"
                try:
                    with _dataviz_lock:
                        result = dataviz.visualize_data(data, description)
                    return result or "Error: DataViz agent returned empty result."
                except Exception as e:
                    return f"Error using DataViz agent: {str(e)}"
            tools.append(use_dataviz_agent)

        if electrify:
            @tool
            def use_electrify_agent(query: str) -> str:
                """Retrieve customer data about utilization, billing, rates and other data associated with the electricity utility account.

                Args:
                    query: The natural language data query
                """
                try:
                    with _electrify_lock:
                        result = electrify.invoke_agent(query)
                    return result or "Error: Electrify agent returned empty result."
                except Exception as e:
                    return f"Error using Electrify agent: {str(e)}"
            tools.append(use_electrify_agent)

        return tools

    def invoke_agent(self, query: str) -> str:
        """Invoke the orchestrator agent with a query."""
        if not self.agent:
            raise RuntimeError("Agent not set up. Call setup() and start chat_loop() first.")
        if not query or not query.strip():
            return "Please provide a valid query."
        logger.info(f"Orchestrator query: {query}")
        try:
            response = self.agent(query.strip())
            return str(response)
        except Exception as e:
            logger.error(f"Error invoking orchestrator: {str(e)}")
            return f"Error: {str(e)}"

    def chat_loop(self):
        """Interactive chat loop for CLI usage."""
        print("\n==================== Orchestrator Agent (Strands) ====================")
        print("This orchestrator coordinates between two specialized agents:")
        print("• DataViz Agent: Creates charts and visualizations")
        print("• Electrify Agent: Queries electricity company database")
        print("")
        print("Type 'quit' to exit.")
        print("=====================================================================")

        # Open MCP contexts for sub-agents that use HTTPS
        mcp_contexts = []

        if self.dataviz_agent and self.dataviz_agent.mcp_client:
            self.dataviz_agent.mcp_client.__enter__()
            mcp_contexts.append(self.dataviz_agent.mcp_client)
            mcp_tools = self.dataviz_agent.mcp_client.list_tools_sync()
            # Use ONLY gateway tools (+ getDateTime) so policy enforcement applies.
            # Do NOT include BUILTIN_TOOLS — they duplicate gateway tools and bypass policies.
            self.dataviz_agent.agent = Agent(
                model=self.dataviz_agent.model,
                system_prompt=self.dataviz_agent._system_prompt,
                tools=[dataviz_getDateTime] + mcp_tools
            )
            logger.info(f"DataViz agent created with {len(mcp_tools)} gateway tools (policy-enforced)")

        if self.electrify_agent and self.electrify_agent.https_mcp_client:
            self.electrify_agent.https_mcp_client.__enter__()
            mcp_contexts.append(self.electrify_agent.https_mcp_client)
            from electrify_agent import getDateTime as electrify_getDateTime, calculateSavings as electrify_calculateSavings
            https_tools = self.electrify_agent.https_mcp_client.list_tools_sync()
            self.electrify_agent.agent = Agent(
                model=self.electrify_agent.model,
                system_prompt=self.electrify_agent.system_prompt,
                tools=[electrify_getDateTime, electrify_calculateSavings] + https_tools
            )
            logger.info(f"Electrify agent created with {len(https_tools)} gateway tools")

        try:
            # Create orchestrator tools and agent
            tools = self._create_tools()
            self.agent = Agent(
                model=self.model,
                system_prompt=self.system_prompt,
                tools=tools
            )
            logger.info("Orchestrator agent created")

            while True:
                try:
                    query = input("\n>>> Your query: ").strip()
                    if query.lower() == 'quit':
                        break
                    if not query:
                        continue
                    response = self.invoke_agent(query)
                    print(f"\n{response}")
                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    break
                except Exception as e:
                    logger.error(f"Error in chat loop: {str(e)}")
                    print(f"\nError in chat loop: {str(e)}")
        finally:
            for ctx in mcp_contexts:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass


def main():
    """Main function for CLI usage."""
    _ = load_dotenv('.env')

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user', default=os.getenv('USER', 'unknown'))
    parser.add_argument('-t', '--thread', default=None)
    parser.add_argument('-p', '--system-prompt', default="./system_prompt.md")
    parser.add_argument('-m', '--model', default="global.anthropic.claude-sonnet-4-6")
    parser.add_argument('-r', '--region', default=os.getenv("AWS_REGION", "us-east-1"))
    parser.add_argument('--memory-id', default=None)
    args = parser.parse_args()

    config = OrchestratorConfig(
        model=args.model,
        user=args.user,
        thread_id=args.thread or args.user,
        region=args.region,
        memory_id=args.memory_id,
        identity=args.user
    )

    try:
        agent = OrchestratorAgent(config)
        agent.setup()
        agent.chat_loop()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        logger.error(f"Main error: {str(e)}")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
