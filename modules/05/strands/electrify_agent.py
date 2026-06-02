#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
Electrify Agent - Strands SDK equivalent of the LangGraph electrify_agent.py.
Connects to the Electrify MCP server via stdio or HTTPS gateway.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands.tools.mcp import MCPClient
from strands.session import FileSessionManager
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
file_handler = logging.FileHandler('/tmp/electrify_agent.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger("electrify-agent")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

_ = load_dotenv('.env')

# define parser
parser = argparse.ArgumentParser()
required = parser.add_argument_group('Required Parameters')
required.add_argument('-u', '--user', help="The name of the user using the agent", default=os.getenv('USER', 'unknown'))
required.add_argument('-t', '--thread', help="Thread identifier", default=None)
required.add_argument('-p', '--system-prompt', help="Path to system prompt .md file", default="./system_prompt.md")
required.add_argument('-s', '--server-script', help="Path to the MCP server script")
required.add_argument('-a', '--server-args', nargs='*', help="Arguments for the server script")
required.add_argument('--https-url', help="HTTPS MCP gateway URL")
required.add_argument('--https-headers', help="JSON string of headers for HTTPS gateway")
required.add_argument('-r', '--region', default=os.getenv("AWS_REGION", "us-east-1"))
required.add_argument('-m', '--model', help="Model ID", default="global.anthropic.claude-sonnet-4-6")
args = parser.parse_args()


@tool
def getDateTime() -> str:
    """Get the date and time in the local timezone.

    Returns:
        str: The current date and time in ISO format
    """
    return str(datetime.now().astimezone().isoformat())


@tool
def calculateSavings(current_rate: float, new_rate: float, monthly_kwh: float) -> str:
    """Calculate potential monthly savings when switching electricity rate plans.

    Args:
        current_rate: Current electricity rate in cents per kWh
        new_rate: New electricity rate in cents per kWh
        monthly_kwh: Average monthly electricity usage in kilowatt-hours

    Returns:
        str: A formatted message showing the monthly and annual savings
    """
    current_cost = (current_rate * monthly_kwh) / 100
    new_cost = (new_rate * monthly_kwh) / 100
    monthly_savings = current_cost - new_cost
    annual_savings = monthly_savings * 12

    return f"Switching from {current_rate}¢/kWh to {new_rate}¢/kWh with {monthly_kwh} kWh monthly usage:\n- Monthly savings: ${monthly_savings:.2f}\n- Annual savings: ${annual_savings:.2f}"


class Application:
    def __init__(self, args):
        try:
            logger.info("Initializing Application...")
            self.args = args
            self.mcp_client = None
            self.https_mcp_client = None

            # Determine mode: stdio or HTTPS
            self.use_https = bool(getattr(args, 'https_url', None))
            self.use_stdio = bool(getattr(args, 'server_script', None) and getattr(args, 'server_args', None))

            if not self.use_https and not self.use_stdio:
                raise ValueError("Must provide either --server-script + --server-args (stdio) or --https-url (HTTPS gateway)")

            # Load system prompt if provided as file
            self.system_prompt_template = ""
            prompt_path = getattr(args, 'system_prompt', None)
            if prompt_path and os.path.exists(prompt_path) and prompt_path.lower().endswith('.md'):
                with open(prompt_path, 'r', encoding='utf-8') as file:
                    self.system_prompt_template = file.read()
                logger.info(f"System prompt loaded: {len(self.system_prompt_template)} characters")
            else:
                # Use default electrify prompt
                self.system_prompt_template = _default_electrify_prompt()
                logger.info("Using default electrify system prompt")

        except Exception as e:
            logger.error(f"Error initializing: {str(e)}")
            print(f"[ERROR] initializing the agent: {str(e)}")
            sys.exit(2)

    def setup(self):
        try:
            logger.info("Setting up agent...")

            username = self.args.user if self.args.user else "unknown"
            self.thread_id = self.args.thread if self.args.thread else username
            model_id = self.args.model if self.args.model else "global.anthropic.claude-sonnet-4-6"

            # Inject username into system prompt
            identity_info = f"\n\nCurrent user identity: {username}\nWhen calling tools that require customer_username, always use: {username}"
            self.system_prompt = f"{self.system_prompt_template}{identity_info}"

            # Initialize Amazon Bedrock model
            self.model = BedrockModel(
                model_id=model_id,
                region_name=os.getenv('MODEL_REGION', os.getenv('AWS_REGION', 'us-east-1'))
            )
            logger.info("Amazon Bedrock model initialized")

            # Initialize MCP client based on mode
            if self.use_stdio:
                if len(self.args.server_args) == 1:
                    import shlex
                    params = shlex.split(self.args.server_args[0])
                else:
                    params = self.args.server_args

                self.mcp_client = MCPClient(
                    lambda: stdio_client(StdioServerParameters(
                        command=self.args.server_script,
                        args=params
                    ))
                )
                logger.info("Stdio MCP client configured")

            if self.use_https:
                from strands.tools.mcp import MCPClient as MCPClientHTTPS
                from mcp.client.streamable_http import streamablehttp_client

                headers = {}
                if getattr(self.args, 'https_headers', None):
                    try:
                        headers = json.loads(self.args.https_headers)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON for headers: {self.args.https_headers}")

                self.https_mcp_client = MCPClientHTTPS(
                    lambda: streamablehttp_client(url=self.args.https_url, headers=headers)
                )
                logger.info(f"HTTPS MCP client configured at {self.args.https_url}")

            logger.info("Agent setup complete")

        except Exception as e:
            logger.error(f"Error setting up agent: {str(e)}")
            print(f"[ERROR] setting up the agent: {str(e)}")
            sys.exit(2)

    def invoke_agent(self, query):
        if not query or not query.strip():
            return "Please provide a valid query."
        logger.info(f"Query: {query}")
        try:
            response = self.agent(query)
            return str(response)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"Error invoking agent: {str(e)}")
            return f"Error: {str(e)}"

    def chat_loop(self):
        print("\n====================== Strands Agent ===============================")
        print("This agent is designed to explore connecting a custom MCP server to ")
        print("a Strands Agents SDK agent. Supports stdio and HTTPS transports.\n")
        print("How to use:")
        print("type your queries, or instructions for the agent or  ")
        print("type 'quit' or press CTRL+C to exit.")
        print("====================================================================")

        # Collect all MCP tools from active clients
        mcp_contexts = []
        all_mcp_tools = []

        if self.mcp_client:
            self.mcp_client.__enter__()
            mcp_contexts.append(self.mcp_client)
            all_mcp_tools.extend(self.mcp_client.list_tools_sync())
            logger.info(f"Retrieved {len(all_mcp_tools)} stdio MCP tools")

        if self.https_mcp_client:
            self.https_mcp_client.__enter__()
            mcp_contexts.append(self.https_mcp_client)
            https_tools = self.https_mcp_client.list_tools_sync()
            all_mcp_tools.extend(https_tools)
            logger.info(f"Retrieved {len(https_tools)} HTTPS MCP tools")

        try:
            # Create session manager for conversation persistence
            session_manager = FileSessionManager(
                session_id=self.thread_id,
                storage_dir=os.path.join(os.path.dirname(__file__), ".sessions")
            )

            # Create agent once with all tools
            self.agent = Agent(
                model=self.model,
                system_prompt=self.system_prompt,
                tools=[getDateTime, calculateSavings] + all_mcp_tools,
                session_manager=session_manager
            )
            logger.info("Agent created")

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


def _default_electrify_prompt():
    return """# Energy Assistant System Prompt

You are a helpful, knowledgeable energy assistant for electrify's customer app. Your purpose is to help customers manage their energy accounts, understand their bills, navigate policies, and make informed decisions about their energy plans.

## Your Core Responsibilities:

**Bill Management & Payments:**
- Help customers view, understand, and pay their bills
- Explain charges, usage patterns, and billing cycles

**Rate Plans & Optimization:**
- Compare available rate plans based on customer usage patterns
- Recommend plans that could reduce costs

**Renewable Energy Guidance:**
- Present available renewable energy plans
- Explain environmental benefits and cost implications

Be friendly, patient, and empathetic. Use clear, jargon-free language.
"""


def main():
    try:
        logger.info("Starting application")
        app = Application(args=args)
        app.setup()
        app.chat_loop()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Main error: {str(e)}")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
