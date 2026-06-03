#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
import logging
import os
import sys
import boto3
import asyncio
import argparse
import traceback
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

# load environment variables from .env
load_dotenv()

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# define parser
parser = argparse.ArgumentParser()
required = parser.add_argument_group('Required Parameters')
required.add_argument('-s', '--server-script', help="The path to the MCP server script")
required.add_argument('-a', '--server-args', nargs='*', help="Arguments for the server script")
required.add_argument('-m', '--model', help="The model ID for the LLM to use for the agent")
args = parser.parse_args()

# Agent implementation
class Agent:
    def __init__(self, logger, provider, args):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.logger = logger
        self.bedrock = provider
        self.args = args


    async def __aenter__(self):
        await self.exit_stack.__aenter__()


    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.__aexit__(exc_type, exc_val, exc_tb)


    async def connect_to_mcp_server(self):
        if self.args.server_script is None or self.args.server_args is None or not (self.args.server_script.endswith('.py') or self.args.server_script == 'uvx'):
            raise ValueError("Server script must be a .py file or uvx, and a list of command parameters must be present")

        # If we have a single string argument, split it into separate arguments
        if len(self.args.server_args) == 1:
            import shlex
            params = shlex.split(self.args.server_args[0])
        else:
            params = self.args.server_args
            
        self.logger.info(f"Connecting to MCP server with command: {'uvx' if self.args.server_script == 'uvx' else 'python'} {' '.join(params)}")
        
        # Extract region from server args if present
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        for param in params:
            if param.startswith('--region='):
                aws_region = param.split('=')[1]
                break
        
        server_params = StdioServerParameters(
            command="uvx" if self.args.server_script == "uvx" else "python",
            args=params,
            env={
                "AWS_REGION": aws_region,
                "FASTMCP_LOG_LEVEL": "ERROR"
            }
        )
    
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
    
        await self.session.initialize()
    
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        # List available resources
        response = await self.session.list_resources()
        resources = response.resources
        print("\nConnected to server with resources:", [resource.name for resource in resources])


    async def process_task(self, message):
        messages = [
            {
                "role": "user",
                "content": [{"text": message}]
            }
        ]
        
        system = [{
            "text": "Provide responses in plain text format suitable for terminal display. Do not use markdown formatting, code blocks, or special characters. Use simple text with clear spacing and indentation."
        }]
    
        response = await self.session.list_tools()

        available_tools = []
        for tool in response.tools:
            available_tool = {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {
                        "json": tool.inputSchema
                    }
                }
            }
            available_tools.append(available_tool)

        response = self.bedrock.converse(
            modelId=self.args.model,
            messages=messages,
            system=system,
            inferenceConfig={
                "maxTokens": 4096,
                "temperature": 0.5
            },
            toolConfig={"tools": available_tools}
        )

        # Process response and handle tool calls in a loop (applies chain of thought)
        final_text = []

        while True:
            output_message = response['output']['message']
            stop_reason = response['stopReason']

            assistant_message_content = []
            for content in output_message['content']:
                if 'text' in content:
                    final_text.append(content['text'])
                assistant_message_content.append(content)

            if stop_reason != 'tool_use':
                break

            # Handle tool calls
            tool_results = []
            for tool_request in output_message['content']:
                if 'toolUse' in tool_request:
                    tool = tool_request['toolUse']
                    self.logger.info("Requesting tool %s. Request: %s", tool['name'], tool['toolUseId'])

                    tool_name = tool['name']
                    tool_args = tool['input']

                    # Execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                    self.logger.info('Debug Tool Response: %s', result)
                    
                    result_text = str(result) if result else "No result returned"
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool['toolUseId'],
                            "content": [{"text": result_text}]
                        }
                    })

            messages.append({
                "role": "assistant",
                "content": assistant_message_content
            })
            messages.append({
                "role": "user",
                "content": tool_results
            })

            response = self.bedrock.converse(
                modelId=self.args.model,
                messages=messages,
                system=system,
                inferenceConfig={
                    "maxTokens": 4096,
                    "temperature": 0.5
                },
                toolConfig={"tools": available_tools}
            )

        return "\n".join(final_text)


    async def chat_loop(self):
        print("\n================== Mini agent for MCP tool calling ====================")
        print("This agent is designed to explore the implementation of an MCP server.")
        print("It implements a minimal MCP client interface, and event loop to review")
        print("MPC server responses. Only 'stdio' transport is currently supported.\n")
        print("How to use:")
        print("type your queries, or instructions for the agent or type 'quit' or press CTRL+C to exit.")
        print("========================================================================")
    
        while True:
            try:
                query = input("\n>>> Your query: ").strip()
    
                if query.lower() == 'quit':
                    break
    
                response = await self.process_task(query)
                print(f"\n{response}")
    
            except (KeyboardInterrupt, asyncio.CancelledError):
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    try:
        # Initialize Bedrock client
        bedrock_runtime = None
        try:
            bedrock_runtime = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.getenv('MODEL_REGION', 'us-west-2'),
            )
        except Exception as e:
            logger.error(str(e))
        

        agent = Agent(logger=logger, provider=bedrock_runtime, args=args)
        await agent.connect_to_mcp_server()
        await agent.chat_loop()
        await agent.cleanup()

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\nExiting...")
        logger.info("Application terminated by user")
        try:
            await agent.cleanup()
        except Exception:
            pass
    except Exception as e:
        logger.error(str(e))
        traceback.print_exc()
        try:
            await agent.cleanup()
        except Exception:
            pass
        sys.exit(1)
     

if __name__ == "__main__":
    asyncio.run(main())
