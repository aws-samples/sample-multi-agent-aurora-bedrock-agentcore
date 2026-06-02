# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
import os
import sys
import json
import argparse
import traceback
import asyncio
import psycopg
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver 
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents.middleware import wrap_model_call, wrap_tool_call, ModelRequest, ModelResponse
from langchain.agents import AgentState
from typing import Any, Callable


def sanitize_message(msg):
    """Sanitize a message to be Bedrock-compatible."""
    if isinstance(msg, ToolMessage):
        content = msg.content
        if isinstance(content, list):
            text_parts = [block.get('text', str(block)) if isinstance(block, dict) else str(block) for block in content]
            content = '\n'.join(text_parts)
        elif not isinstance(content, str):
            content = str(content)
        return ToolMessage(content=content, tool_call_id=msg.tool_call_id, name=getattr(msg, 'name', None))
    return msg


def _sanitize_checkpoint(checkpoint):
    """Sanitize checkpoint messages for Amazon Bedrock compatibility."""
    if not checkpoint or 'channel_values' not in checkpoint:
        return checkpoint
    messages = checkpoint.get('channel_values', {}).get('messages', [])
    if messages:
        checkpoint['channel_values']['messages'] = [sanitize_message(m) for m in messages]
    return checkpoint


class SanitizedInMemorySaver(InMemorySaver):
    def get(self, config):
        return _sanitize_checkpoint(super().get(config))
    async def aget(self, config):
        return _sanitize_checkpoint(await super().aget(config))


class SanitizedAsyncPostgresSaver(AsyncPostgresSaver):
    async def aget(self, config):
        return _sanitize_checkpoint(await super().aget(config))
    def get_tuple(self, config):
        result = super().get_tuple(config)
        return result
    async def aget_tuple(self, config):
        result = await super().aget_tuple(config)
        return result


def wrap_tool_for_bedrock(original_tool):
    """Wrap a tool to ensure its output is a plain string for Amazon Bedrock."""
    async def wrapped_func(*args, **kwargs):
        result = await original_tool.ainvoke(kwargs)
        return json.dumps(result) if isinstance(result, (dict, list)) else str(result)
    def sync_wrapped_func(*args, **kwargs):
        result = original_tool.invoke(kwargs)
        return json.dumps(result) if isinstance(result, (dict, list)) else str(result)
    return StructuredTool(
        name=original_tool.name,
        description=original_tool.description,
        args_schema=original_tool.args_schema,
        func=sync_wrapped_func,
        coroutine=wrapped_func
    )


@wrap_model_call
async def sanitize_messages_middleware(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Middleware to sanitize and deduplicate messages before sending to LLM (transient)."""
    messages = request.messages
    seen_tool_call_ids = set()
    deduped = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if msg.tool_call_id in seen_tool_call_ids:
                continue  # Skip duplicate tool results
            seen_tool_call_ids.add(msg.tool_call_id)
            deduped.append(sanitize_message(msg))
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            # Deduplicate tool_calls within AIMessage
            seen_ai_tool_ids = set()
            unique_tool_calls = []
            for tc in msg.tool_calls:
                tc_id = tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None)
                if tc_id and tc_id not in seen_ai_tool_ids:
                    seen_ai_tool_ids.add(tc_id)
                    unique_tool_calls.append(tc)
            if unique_tool_calls != msg.tool_calls:
                msg = AIMessage(content=msg.content, tool_calls=unique_tool_calls)
            deduped.append(msg)
        else:
            deduped.append(sanitize_message(msg))
    return await handler(request.override(messages=deduped))


@wrap_tool_call
async def sanitize_tool_output(request, handler):
    """Middleware to ensure tool outputs are plain strings."""
    result = await handler(request)
    if hasattr(result, 'content'):
        content = result.content
        if isinstance(content, list):
            text_parts = [block.get('text', str(block)) if isinstance(block, dict) else str(block) for block in content]
            content = '\n'.join(text_parts)
        elif not isinstance(content, str):
            content = str(content)
        return ToolMessage(
            content=content,
            tool_call_id=result.tool_call_id if hasattr(result, 'tool_call_id') else request.tool_call.get('id', ''),
            name=result.name if hasattr(result, 'name') else request.tool_call.get('name', '')
        )
    return result


# Configure logging - only INFO to file, minimal console output
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


@tool(parse_docstring=True)
def getDateTime() -> str:
  """Get the date and time in the local timezone.

  Returns:
    str: The current date and time in ISO format
  """
  return str(datetime.now().astimezone().isoformat())


@tool(parse_docstring=True)
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



# Implement the agentic AI application experience
class Application:
    def __init__(self, args):
      try:
        logger.info("Initializing Application...")
        
        self.args = args
        if self.args.server_script is None or self.args.server_args is None or not (self.args.server_script.endswith('.py') or self.args.server_script == 'uvx' or self.args.server_script == 'uv'):
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



    async def setup(self):
      try:
        logger.info("Setting up agent...")
        
        # Basic config
        username = self.args.user if 'user' in self.args and self.args.user else "unknown"
        thread_id = self.args.thread if 'thread' in self.args and self.args.thread else username
        model_id = self.args.model if 'model' in self.args and self.args.model else "global.anthropic.claude-sonnet-4-6"
        
        # Inject username into system prompt
        self.system_prompt = f"Current user: {username}\n\n{self.system_prompt_template}"

        # Process server arguments
        if len(self.args.server_args) == 1:
            import shlex
            params = shlex.split(self.args.server_args[0])
        else:
            params = self.args.server_args

        # Initialize LLM
        self.llm = init_chat_model(model=model_id, model_provider='bedrock_converse', region_name=os.getenv('MODEL_REGION', 'us-west-2'))
        logger.info("LLM initialized")

        # Agent state configuration
        self.config = {
          "configurable": { "thread_id": thread_id },
          "identity": { "username": username }
        }

        # Initialize MCP client
        mcp_config = {
            "electrify": {
                "command": self.args.server_script,
                "args": params,
                "transport": "stdio",
            }
        }
        
        self.mcp = MultiServerMCPClient(mcp_config)
        mcp_tools = await self.mcp.get_tools()
        logger.info(f"Retrieved {len(mcp_tools)} MCP tools")
        
        # Wrap MCP tools for Amazon Bedrock compatibility and add custom tools
        all_tools = [wrap_tool_for_bedrock(t) for t in mcp_tools]
        all_tools.append(getDateTime)
        all_tools.append(calculateSavings)

        # Set up checkpointer with PostgreSQL for persistence across restarts
        try:
            conn_string = f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@{os.getenv('PGHOST')}/postgres"
            self._pg_conn = await psycopg.AsyncConnection.connect(conn_string, autocommit=True)
            checkpointer = SanitizedAsyncPostgresSaver(self._pg_conn)
            await checkpointer.setup()
            logger.info(f"PostgreSQL checkpointer ready - thread_id: {thread_id}")
        except Exception as checkpoint_error:
            logger.warning(f"Falling back to InMemorySaver (no persistence): {checkpoint_error}")
            print(f"[WARN] Using in-memory checkpointer - conversation won't persist across restarts")
            checkpointer = SanitizedInMemorySaver()

        # Create agent with middleware for Amazon Bedrock compatibility
        self.agent = create_agent(
          model=self.llm,
          tools=all_tools,
          system_prompt=self.system_prompt,
          checkpointer=checkpointer,
          middleware=[sanitize_messages_middleware, sanitize_tool_output]
        )
        logger.info("Agent setup complete")

      except Exception as e:
        logger.error(f"Error setting up agent: {str(e)}")
        print(f"[ERROR] setting up the agent: {str(e)}")
        sys.exit(2)


    async def invoke_agent(self, query):
      if not query or not query.strip():
        return "Please provide a valid query."
      
      logger.info(f"Query: {query}")
      
      try:
        result = await self.agent.ainvoke({
            "messages": [ HumanMessage(content=query) ]
          },
          config=self.config
        )
        
        # Get the final state and return the last assistant message
        after_state = await self.agent.aget_state(self.config)
        after_messages = after_state.values.get("messages", []) if after_state else []
        
        if after_messages:
          last_message = after_messages[-1]
          if hasattr(last_message, 'content'):
            return last_message.content
          else:
            return str(last_message)
        
        return result
        
      except (KeyboardInterrupt, asyncio.CancelledError):
        raise
      except Exception as e:
        logger.error(f"Error invoking agent: {str(e)}")
        return f"Error: {str(e)}"

    async def chat_loop(self):
        print("\n====================== Mini LangGraph Agent ========================")
        print("This agent is designed to explore connecting a custom MCP server to ")
        print("a LangGraph ReAct loop agent. Only 'stdio' transport is currently   ")
        print("supported.\n")
        print("How to use:")
        print("type your queries, or instructions for the agent or  ")
        print("type 'quit' or press CTRL+C to exit.")
        print("====================================================================")
    
        while True:
            try:
                query = input("\n>>> Your query: ").strip()
    
                if query.lower() == 'quit':
                    break
                if not query:
                    continue
    
                response = await self.invoke_agent(query)
                print(f"\n{response}")
    
            except (KeyboardInterrupt, asyncio.CancelledError):
                print("\n\nExiting...")
                break
            except Exception as e:
                error_msg = f"Error in chat loop: {str(e)}"
                logger.error(error_msg)
                print(f"\n{error_msg}")

async def main():
    try:
        logger.info("Starting application")
        
        agent = Application(args=args)
        await agent.setup()
        await agent.chat_loop()

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\nExiting...")
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Main error: {str(e)}")
        print(str(e))
        sys.exit(1)
     

if __name__ == "__main__":
    asyncio.run(main())