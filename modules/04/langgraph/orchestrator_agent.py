#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
Orchestrator Agent Module

Coordinates between the DataViz Agent and Electrify Agent using
LangGraph's create_react_agent.

Architecture
------------
    User query
        |
        v
    OrchestratorAgent (this module)
        |  Amazon Bedrock — Claude Sonnet 4.6 (init_chat_model + bedrock_converse)
        |
        +-- use_electrify_agent(query)  -- ElectrifyAgent
        |   (M2 stdio MCP server: get_rates, get_customer, get_bills)
        |       PostgreSQL via Aurora (sslmode=require)
        |
        +-- use_dataviz_agent(data, description) -- DataVizAgent
            (M3 tools: bar/line/scatter/pie/histogram via matplotlib)

Security boundaries (see THREAT_MODEL.md for full analysis):
- Authentication: deferred to deployer; this orchestrator runs locally as a
  single-user CLI in the workshop. The M5 deployment wraps this in
  AgentCore Runtime fronted by a Cognito-JWT-verified Gateway.
- Authorization: passes the user's identity (-u CLI arg) into sub-agent
  calls. M5's electrify_server enforces that the authenticated principal
  matches the requested customer_username.
- Tool isolation: tool outputs flow through sanitize_tool_output middleware
  to ensure Amazon Bedrock-compatible plain strings.
- Secrets: agents do not load credentials directly; database access is
  brokered by the Electrify MCP server.
"""

import os
import sys
import argparse
import asyncio
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool

# Import the agent modules
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
    electrify_server_path: str = "modules/04/langgraph/electrify_server.py"
    electrify_system_prompt: str = "modules/04/langgraph/electrify_prompt.md"


class OrchestratorAgent:
    """Orchestrator Agent that coordinates DataViz and Electrify agents."""
    
    def __init__(self, config: OrchestratorConfig):
        """Initialize the Orchestrator Agent with configuration."""
        self.config = config
        self.agent = None
        self.dataviz_agent = None
        self.electrify_agent = None
        
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

    async def setup(self):
        """Set up the orchestrator and initialize sub-agents."""
        try:
            logger.info("Setting up Orchestrator agent...")
            
            # Basic config
            username = self.config.user
            thread_id = self.config.thread_id or username
            model_id = self.config.model

            # Initialize LLM
            self.llm = init_chat_model(model=model_id, model_provider='bedrock_converse')
            logger.info("LLM initialized")

            # Agent state configuration
            self.agent_config = {
                "configurable": {"thread_id": thread_id},
                "identity": {"username": username}
            }

            # Initialize DataViz Agent
            dataviz_config = DataVizConfig(
                model=model_id,
                user=username,
                thread_id=f"{thread_id}_dataviz"
            )
            self.dataviz_agent = DataVizAgent(dataviz_config)
            await self.dataviz_agent.setup()
            logger.info("DataViz agent initialized")

            # Initialize Electrify Agent
            # Pass Aurora PostgreSQL connection parameters explicitly
            db_args = [
                "-e", os.getenv('PGHOST', 'localhost'),
                "-p", os.getenv('PGPORT', '5432'),
                "-d", os.getenv('PGDATABASE', 'postgres'),
                "-u", os.getenv('PGUSER', 'postgres'),
                "--password", os.getenv('PGPASSWORD', '')
            ]
            
            electrify_args = argparse.Namespace(
                user=username,
                thread=f"{thread_id}_electrify",
                system_prompt=self.config.electrify_system_prompt,
                server_script=self.config.electrify_server_script,
                server_args=[self.config.electrify_server_path] + db_args,
                model=model_id
            )
            self.electrify_agent = ElectrifyAgent(electrify_args)
            await self.electrify_agent.setup()
            logger.info("Electrify agent initialized")

            # Define orchestrator tools
            tools = [
                self.create_dataviz_tool(),
                self.create_electrify_tool(),
                self.create_datetime_tool(),
                self.create_json_to_csv_tool()
            ]

            # Set up checkpointer
            checkpointer = InMemorySaver()
            logger.info("Using InMemorySaver for conversation state")

            # Create orchestrator agent
            self.agent = create_agent(
                model=self.llm,
                tools=tools,
                system_prompt=self.system_prompt,
                checkpointer=checkpointer
            )
            
            # Configure recursion limit
            self.agent_config["configurable"]["recursion_limit"] = 50
            logger.info("Orchestrator agent setup complete")

        except Exception as e:
            logger.error(f"Error setting up orchestrator: {str(e)}")
            raise

    def create_dataviz_tool(self):
        """Create a tool that wraps the DataViz agent."""
        _dataviz_lock = asyncio.Lock()

        @tool(parse_docstring=True)
        async def use_dataviz_agent(data: str, description: str) -> str:
            """Create data visualizations and charts from CSV data.
            
            Args:
                data: CSV formatted data as string
                description: Description of what visualization to create
            """
            try:
                async with _dataviz_lock:
                    result = await self.dataviz_agent.visualize_data(data, description)
                return result or "Error: DataViz agent returned empty result."
            except Exception as e:
                return f"Error using DataViz agent: {str(e)}"
        
        return use_dataviz_agent

    def create_electrify_tool(self):
        """Create a tool that wraps the Electrify agent."""
        _electrify_lock = asyncio.Lock()

        @tool(parse_docstring=True)
        async def use_electrify_agent(query: str) -> str:
            """Query the electricity company database for customer information, bills, and rates.
            
            Args:
                query: Natural language query about customers, bills, or rate plans
            """
            try:
                async with _electrify_lock:
                    result = await self.electrify_agent.invoke_agent(query)
                return result or "Error: Electrify agent returned empty result."
            except Exception as e:
                return f"Error using Electrify agent: {str(e)}"
        
        return use_electrify_agent

    def create_datetime_tool(self):
        """Create a tool for getting current date/time."""
        @tool(parse_docstring=True)
        def get_current_datetime() -> str:
            """Get the current date and time in ISO format.
            """
            return str(datetime.now().astimezone().isoformat())
        
        return get_current_datetime

    def create_json_to_csv_tool(self):
        """Create a tool for converting JSON data to CSV format."""
        @tool(parse_docstring=True)
        def convert_json_to_csv(json_data: str) -> str:
            """Convert JSON data to CSV format for visualization.
            
            Args:
                json_data: JSON formatted data as string
            """
            try:
                import json
                import pandas as pd
                from io import StringIO
                
                # Parse JSON data
                data = json.loads(json_data)
                
                # Convert to DataFrame
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    return f"Error: Unsupported data format for conversion"
                
                # Convert to CSV
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                return csv_data
                
            except Exception as e:
                return f"Error converting JSON to CSV: {str(e)}"
        
        return convert_json_to_csv

    async def invoke_agent(self, query: str) -> str:
        """Invoke the orchestrator agent with a query.
        
        Args:
            query: The query/request for the orchestrator
            
        Returns:
            str: Orchestrator response
        """
        if not self.agent:
            raise RuntimeError("Agent not set up. Call setup() first.")
            
        logger.info(f"Orchestrator query: {query}")
        
        try:
            if not query or not query.strip():
                return "Please provide a valid query."
            
            logger.info(f"Processing query: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            result = await self.agent.ainvoke({
                "messages": [HumanMessage(content=query.strip())]
            }, config=self.agent_config)
            
            # Get the final state and return the last assistant message
            after_state = await self.agent.aget_state(self.agent_config)
            after_messages = after_state.values.get("messages", []) if after_state else []
            
            if after_messages:
                last_message = after_messages[-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
                return str(last_message)
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error invoking orchestrator: {str(e)}")
            return f"Error: {str(e)}"

    async def chat_loop(self):
        """Interactive chat loop for CLI usage."""
        print("\n==================== Orchestrator Agent ====================")
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
        print("============================================================")

        while True:
            try:
                query = input("\n>>> Your query: ").strip()

                if query.lower() == 'quit':
                    break
                
                # Only process non-empty queries
                if query:
                    response = await self.invoke_agent(query)
                    print(f"\n{response}")
                else:
                    print("\nPlease enter a valid query or 'quit' to exit.")

            except (KeyboardInterrupt, asyncio.CancelledError):
                print("\n\nExiting...")
                break
            except Exception as e:
                error_msg = f"Error in chat loop: {str(e)}"
                logger.error(error_msg)
                print(f"\n{error_msg}")


# Convenience functions
async def create_orchestrator_agent(config: Optional[OrchestratorConfig] = None) -> OrchestratorAgent:
    """Create and set up an Orchestrator agent with default or custom configuration.
    
    Args:
        config: Optional OrchestratorConfig. If None, uses defaults.
        
    Returns:
        OrchestratorAgent: Ready-to-use orchestrator instance
    """
    if config is None:
        config = OrchestratorConfig()
    
    agent = OrchestratorAgent(config)
    await agent.setup()
    return agent


# CLI interface
def main():
    """Main function for CLI usage."""
    # Load environment variables
    load_dotenv('.env')
    
    # Define parser
    parser = argparse.ArgumentParser(description="Orchestrator Agent")
    parser.add_argument('-u', '--user', help="Username", default=os.getenv('USER', 'unknown'))
    parser.add_argument('-t', '--thread', help="Thread ID", default=None)
    parser.add_argument('-m', '--model', help="LLM model ID", default="global.anthropic.claude-sonnet-4-6")
    parser.add_argument('--electrify-server', help="Command to run the Electrify server (e.g. python, uv)", 
                       default="python")
    parser.add_argument('--electrify-server-path', help="Path to Electrify server script",
                       default="modules/04/langgraph/electrify_server.py")
    parser.add_argument('--electrify-args', nargs='*', help="Additional arguments for Electrify server",
                       default=None)
    parser.add_argument('--electrify-prompt', help="Path to Electrify system prompt",
                       default="modules/04/langgraph/electrify_prompt.md")
    parser.add_argument('--log-level', help="Log level", default="INFO")
    
    args = parser.parse_args()
    
    # Create config from args
    config = OrchestratorConfig(
        model=args.model,
        user=args.user,
        thread_id=args.thread,
        log_level=args.log_level,
        electrify_server_script=args.electrify_server,
        electrify_server_path=args.electrify_server_path,
        electrify_server_args=args.electrify_args,
        electrify_system_prompt=args.electrify_prompt
    )
    
    async def run_cli():
        try:
            logger.info("Starting Orchestrator application")
            
            agent = OrchestratorAgent(config)
            await agent.setup()
            await agent.chat_loop()

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n\nExiting...")
            logger.info("Application terminated by user")
        except Exception as e:
            logger.error(f"Main error: {str(e)}")
            print(str(e))
            sys.exit(1)
    
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
