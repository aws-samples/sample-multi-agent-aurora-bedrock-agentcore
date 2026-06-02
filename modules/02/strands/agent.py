# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
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
file_handler = logging.FileHandler('agent_debug.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger("electrify-agent")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# load environment variables from .env
_ = load_dotenv('.env')

# define parser
parser = argparse.ArgumentParser()
required = parser.add_argument_group('Required Parameters')
required.add_argument('-u', '--user', help="The name of the user using the agent", default=os.getenv('USER', 'unknown'))
required.add_argument('-t', '--thread', help="The name or identifier of the conversation thread the user started with the agent. Defaults to the value for --user.", default=None)
required.add_argument('-p', '--system-prompt', help="The path to the Markdown file containing the system prompt, defaults to './system_prompt.md'", default="./system_prompt.md")
required.add_argument('-s', '--server-script', help="The path to the MCP server script")
required.add_argument('-a', '--server-args', nargs='*', help="Arguments for the server script")
required.add_argument('-m', '--model', help="The model ID for the LLM to use for the agent", default="global.anthropic.claude-sonnet-4-6")
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
            if self.args.server_script is None or self.args.server_args is None or not (self.args.server_script.endswith('.py') or self.args.server_script in ('python', 'python3', 'uvx', 'uv')):
                raise ValueError("Server script must be a .py file, uvx or uv, and a list of command parameters must be present")

            # Load system prompt
            if not (os.path.exists(args.system_prompt) and args.system_prompt.lower().endswith('.md')):
                raise ValueError(f"The system prompt file '{args.system_prompt}' does not exist, or is not a Markdown file.")

            with open(args.system_prompt, 'r', encoding='utf-8') as file:
                self.system_prompt_template = file.read()

            logger.info(f"System prompt loaded: {len(self.system_prompt_template)} characters")

        except Exception as e:
            logger.error(f"Error initializing: {str(e)}")
            print(f"[ERROR] initializing the agent: {str(e)}")
            sys.exit(2)

    def setup(self):
        try:
            logger.info("Setting up agent...")

            # Basic config
            username = self.args.user if self.args.user else "unknown"
            self.thread_id = self.args.thread if self.args.thread else username
            model_id = self.args.model if self.args.model else "global.anthropic.claude-sonnet-4-6"

            # Inject username into system prompt
            self.system_prompt = f"Current user: {username}\n\n{self.system_prompt_template}"

            # Process server arguments
            if len(self.args.server_args) == 1:
                import shlex
                params = shlex.split(self.args.server_args[0])
            else:
                params = self.args.server_args

            # Initialize Amazon Bedrock model
            self.model = BedrockModel(
                model_id=model_id,
                region_name=os.getenv('MODEL_REGION', 'us-west-2')
            )
            logger.info("Amazon Bedrock model initialized")

            # Initialize MCP client for the server
            self.mcp_client = MCPClient(
                lambda: stdio_client(StdioServerParameters(
                    command=self.args.server_script,
                    args=params
                ))
            )

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
        print("a Strands Agents SDK agent. Only 'stdio' transport is currently     ")
        print("supported.\n")
        print("How to use:")
        print("type your queries, or instructions for the agent or  ")
        print("type 'quit' or press CTRL+C to exit.")
        print("====================================================================")

        # Keep MCP client open for the entire chat session
        with self.mcp_client:
            mcp_tools = self.mcp_client.list_tools_sync()
            logger.info(f"Retrieved {len(mcp_tools)} MCP tools")

            # Create session manager for conversation persistence
            session_manager = FileSessionManager(
                session_id=self.thread_id,
                storage_dir=os.path.join(os.path.dirname(__file__), ".sessions")
            )

            # Create agent once with all tools
            self.agent = Agent(
                model=self.model,
                system_prompt=self.system_prompt,
                tools=[getDateTime, calculateSavings] + mcp_tools,
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
                    error_msg = f"Error in chat loop: {str(e)}"
                    logger.error(error_msg)
                    print(f"\n{error_msg}")


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