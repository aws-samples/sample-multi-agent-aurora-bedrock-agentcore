#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
Orchestrator Agent Module (Strands SDK)

Strands equivalent of the LangGraph orchestrator_agent.py.
Coordinates between the DataViz Agent and Electrify Agent.

Architecture
------------
    User query
        |
        v
    OrchestratorAgent (this module)
        |  Amazon Bedrock — Claude Sonnet 4.6 (BedrockModel)
        |
        +-- use_electrify_agent(query)  -- ElectrifyAgent
        |   (M2 stdio MCP server: get_rates, get_customer, get_bills)
        |       PostgreSQL via Aurora (sslmode=require)
        |
        +-- use_dataviz_agent(data, description) -- DataVizAgent
            (M3 BUILTIN_TOOLS: bar/line/scatter/pie/histogram via matplotlib)

Security boundaries (see THREAT_MODEL.md for full analysis):
- Authentication: deferred to deployer; this orchestrator runs locally as a
  single-user CLI in the workshop. The M5 deployment wraps this in
  AgentCore Runtime fronted by a Cognito-JWT-verified Gateway.
- Authorization: passes the user's identity (-u CLI arg) into sub-agent
  calls. M5's electrify_server enforces that the authenticated principal
  matches the requested customer_username.
- Tool isolation: each sub-agent runs in its own MCPClient context; tool
  outputs are sanitized through Strands SDK's built-in normalization.
- Secrets: agents do not load credentials directly; database access is
  brokered by the Electrify MCP server.
"""

import os
import sys
import argparse
import json
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands.tools.mcp import MCPClient
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

# Import the Strands agent modules
from dataviz_agent import DataVizAgent, DataVizConfig
from electrify_agent import Application as ElectrifyAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator-agent")


@dataclass
class OrchestratorConfig:
    """Configuration for the Orchestrator Agent."""
    model: str = "global.anthropic.claude-sonnet-4-6"
    user: str = "unknown"
    thread_id: Optional[str] = None
    log_level: str = "INFO"
    electrify_server_script: str = "python"
    electrify_server_args: list = None
    electrify_system_prompt: str = "modules/04/strands/electrify_prompt.md"


class OrchestratorAgent:
    """Orchestrator Agent that coordinates DataViz and Electrify agents (Strands SDK)."""

    def __init__(self, config: OrchestratorConfig):
        """Initialize the Orchestrator Agent with configuration."""
        self.config = config
        self.agent = None
        self.dataviz_agent = None
        self.electrify_agent = None
        self.mcp_client = None

        # Configure logging
        logger.setLevel(getattr(logging, config.log_level.upper()))
        logger.info("Initializing Orchestrator Agent...")

        # System prompt for orchestration
        base_prompt = """You are an intelligent orchestrator agent that coordinates between two specialized agents:

1. **DataViz Agent**: Creates charts and visualizations from CSV data
   - Use when users want to create charts, graphs, or visualizations
   - Can create bar charts, line charts, scatter plots, pie charts
   - Requires CSV data as input

2. **Electrify Agent**: Retrieves data from an electricity company database
   - Use when users want to query customer information, bills, or rate plans
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

IMPORTANT: Always call tools ONE AT A TIME. Never issue multiple tool calls in parallel. Wait for each tool result before deciding the next step.

Always explain what you're doing and why you're using specific agents."""

        identity_info = ""
        if config.user and config.user != "unknown":
            identity_info = f"\n\nCurrent user identity: {config.user}\nWhen the user says 'my', 'I', or 'me', they refer to {config.user}. Pass this identity to the Electrify agent when querying customer-specific data; do not ask the user for their customer ID."
        self.system_prompt = base_prompt + identity_info

    def setup(self):
        """Set up the orchestrator and initialize sub-agents."""
        try:
            logger.info("Setting up Orchestrator agent...")

            username = self.config.user
            model_id = self.config.model

            # Initialize Amazon Bedrock model
            self.model = BedrockModel(
                model_id=model_id,
                region_name=os.getenv('MODEL_REGION', 'us-west-2')
            )
            logger.info("Amazon Bedrock model initialized")

            # Initialize DataViz Agent
            dataviz_config = DataVizConfig(
                model=model_id,
                user=username,
                thread_id=f"{self.config.thread_id or username}_dataviz"
            )
            self.dataviz_agent = DataVizAgent(dataviz_config)
            self.dataviz_agent.setup()
            # Create the dataviz agent instance (no MCP needed for dataviz built-in tools)
            self.dataviz_agent.agent = self.dataviz_agent._create_agent_with_tools()
            logger.info("DataViz agent initialized")

            # Initialize Electrify Agent
            db_args = [
                "-e", os.getenv('PGHOST', 'localhost'),
                "-p", os.getenv('PGPORT', '5432'),
                "-d", os.getenv('PGDBNAME', 'postgres'),
                "-u", os.getenv('PGUSER', 'postgres'),
                "--password", os.getenv('PGPASSWORD', '')
            ]

            electrify_args = argparse.Namespace(
                user=username,
                thread=f"{self.config.thread_id or username}_electrify",
                system_prompt=self.config.electrify_system_prompt,
                server_script=self.config.electrify_server_script,
                server_args=self.config.electrify_server_args + db_args if self.config.electrify_server_args else db_args,
                model=model_id
            )
            self.electrify_agent = ElectrifyAgent(electrify_args)
            self.electrify_agent.setup()
            logger.info("Electrify agent initialized")

            # Set up MCP client for the electrify server (used by orchestrator tools)
            server_args = (self.config.electrify_server_args or []) + db_args
            self.mcp_client = MCPClient(
                lambda: stdio_client(StdioServerParameters(
                    command=self.config.electrify_server_script,
                    args=server_args
                ))
            )

            logger.info("Orchestrator agent setup complete")

        except Exception as e:
            logger.error(f"Error setting up orchestrator: {str(e)}")
            raise

    def _create_tools(self):
        """Create orchestrator tools that wrap sub-agents."""
        dataviz = self.dataviz_agent
        electrify = self.electrify_agent
        _dataviz_lock = threading.Lock()
        _electrify_lock = threading.Lock()

        @tool
        def use_dataviz_agent(data: str, description: str) -> str:
            """Create data visualizations and charts from CSV data.

            Args:
                data: CSV formatted data as string
                description: Description of what visualization to create
            """
            try:
                with _dataviz_lock:
                    result = dataviz.visualize_data(data, description)
                return result or "Error: DataViz agent returned empty result."
            except Exception as e:
                return f"Error using DataViz agent: {str(e)}"

        @tool
        def use_electrify_agent(query: str) -> str:
            """Query the electricity company database for customer information, bills, and rates.

            Args:
                query: Natural language query about customers, bills, or rate plans
            """
            try:
                with _electrify_lock:
                    result = electrify.invoke_agent(query)
                return result or "Error: Electrify agent returned empty result."
            except Exception as e:
                return f"Error using Electrify agent: {str(e)}"

        @tool
        def get_current_datetime() -> str:
            """Get the current date and time in ISO format."""
            return str(datetime.now().astimezone().isoformat())

        @tool
        def convert_json_to_csv(json_data: str) -> str:
            """Convert JSON data to CSV format for visualization.

            Args:
                json_data: JSON formatted data as string
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

        return [use_dataviz_agent, use_electrify_agent, get_current_datetime, convert_json_to_csv]

    def invoke_agent(self, query: str) -> str:
        """Invoke the orchestrator agent with a query."""
        if not self.agent:
            raise RuntimeError("Agent not set up. Call setup() first.")

        logger.info(f"Orchestrator query: {query}")

        try:
            if not query or not query.strip():
                return "Please provide a valid query."

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
        print("I can help you:")
        print("• Retrieve customer data, bills, and rate information")
        print("• Create visualizations from your data")
        print("• Chain operations (get data, then visualize it)")
        print("• Answer general questions about energy management")
        print("")
        print("Examples:")
        print("• 'Show me customer Richard_doe's information'")
        print("• 'Get the latest bills for customer jane_smith and create a chart'")
        print("• 'Create a bar chart from this CSV data: ...'")
        print("• 'What are the available rate plans?'")
        print("")
        print("Type 'quit' to exit.")
        print("=====================================================================")

        # Keep MCP client open for the entire chat session
        with self.mcp_client:
            mcp_tools = self.mcp_client.list_tools_sync()
            logger.info(f"Retrieved {len(mcp_tools)} MCP tools for electrify agent")

            # Create the electrify agent with MCP tools
            from electrify_agent import getDateTime as electrify_getDateTime, calculateSavings as electrify_calculateSavings
            self.electrify_agent.agent = Agent(
                model=self.electrify_agent.model,
                system_prompt=self.electrify_agent.system_prompt,
                tools=[electrify_getDateTime, electrify_calculateSavings] + mcp_tools
            )

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

                    if query:
                        response = self.invoke_agent(query)
                        print(f"\n{response}")
                    else:
                        print("\nPlease enter a valid query or 'quit' to exit.")

                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    break
                except Exception as e:
                    error_msg = f"Error in chat loop: {str(e)}"
                    logger.error(error_msg)
                    print(f"\n{error_msg}")


# Convenience functions
def create_orchestrator_agent(config: Optional[OrchestratorConfig] = None) -> OrchestratorAgent:
    """Create and set up an Orchestrator agent with default or custom configuration."""
    if config is None:
        config = OrchestratorConfig()
    agent = OrchestratorAgent(config)
    agent.setup()
    return agent


# CLI interface
def main():
    """Main function for CLI usage."""
    load_dotenv('.env')

    parser = argparse.ArgumentParser(description="Orchestrator Agent (Strands)")
    parser.add_argument('-u', '--user', help="Username", default=os.getenv('USER', 'unknown'))
    parser.add_argument('-t', '--thread', help="Thread ID", default=None)
    parser.add_argument('-m', '--model', help="LLM model ID", default="global.anthropic.claude-sonnet-4-6")
    parser.add_argument('--electrify-server', help="Command to run the Electrify server",
                       default="python")
    parser.add_argument('--electrify-args', nargs='*', help="Arguments for the Electrify server command",
                       default=["modules/04/strands/electrify_server.py"])
    parser.add_argument('--electrify-prompt', help="Path to Electrify system prompt",
                       default="modules/04/strands/electrify_prompt.md")
    parser.add_argument('--log-level', help="Log level", default="INFO")

    args = parser.parse_args()

    config = OrchestratorConfig(
        model=args.model,
        user=args.user,
        thread_id=args.thread,
        log_level=args.log_level,
        electrify_server_script=args.electrify_server,
        electrify_server_args=args.electrify_args,
        electrify_system_prompt=args.electrify_prompt
    )

    try:
        logger.info("Starting Orchestrator application")

        agent = OrchestratorAgent(config)
        agent.setup()
        agent.chat_loop()

    except KeyboardInterrupt:
        print("\n\nExiting...")
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Main error: {str(e)}")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
