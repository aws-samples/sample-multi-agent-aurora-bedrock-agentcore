# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import traceback
import asyncio
import logging
import uuid
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from common.types import AgentConfig, StdioServerConfig
from common.cli import agent_cli_runner


# Implement a base agent class
class BaseAgent:
    def __init__(self, config: AgentConfig):
        # save config
        self.agent = None
        self.config = config

        # Configure logging - only INFO to file, minimal console output
        self.logger = logging.getLogger(config.name)
        self.logger.setLevel(logging.INFO)
        if config.log_file: 
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)

        # Inject username into system prompt
        self.system_prompt = f"{self.config.system_prompt}\n\n# Current customer username is: {self.config.identity}\n\n"


    async def setup(self):
        self.logger.info(f"Setting up the ReAct agent...")


    async def invoke(self, query, use_fresh_thread=False):
        self.logger.info(f"{self.config.name if self.config.name else 'unknown_agent'} - invoked with query: {query}")
      
        # Agent state configuration
        # Use fresh thread_id for each invocation if requested (for child agents called as tools)
        thread_id = str(uuid.uuid4()) if use_fresh_thread else self.config.thread
        actor_id_hash = hashlib.sha256((self.config.identity or "unknown").encode()).hexdigest()
        
        config = {
            "configurable": { "thread_id": thread_id, "actor_id": actor_id_hash },
            "identity": { "username": self.config.identity }
        }

        # Invoke the ReAct loop
        if self.agent:
            # Invoke LangGraph agent
            result = await self.agent.ainvoke({
                "messages": [ HumanMessage(content=query) ]
            },
            config=config
            )

            # Get the final state and return the last assistant message
            after_state = await self.agent.aget_state(config)
            after_messages = after_state.values.get("messages", []) if after_state else []
            self.logger.info(f"{self.config.name if self.config.name else 'unknown_agent'} - conversation state:")
            self.logger.info([str(m) for m in after_messages])
            
            # Extract images from tool results
            images = []
            last_human_idx = -1
            
            # Find the last human message index
            for i in range(len(after_messages) - 1, -1, -1):
                if hasattr(after_messages[i], 'type') and after_messages[i].type == 'human':
                    last_human_idx = i
                    break
            
            # Iterate over messages after the last human message
            for i in range(last_human_idx + 1, len(after_messages)):
                msg = after_messages[i]
                # Check if this is a tool message with content
                if hasattr(msg, 'type') and msg.type == 'tool' and hasattr(msg, 'content'):
                    content = msg.content
                    # Look for <chart>...</chart> tags with SVG data URLs
                    import re
                    chart_pattern = r'<chart>(data:image/svg\+xml;base64,[^<]+)</chart>'
                    matches = re.findall(chart_pattern, content)
                    images.extend(matches)
            
            # Get the text response
            text_response = ""
            if after_messages:
                self.logger.info(f"{self.config.name if self.config.name else 'unknown_agent'} - found conversation state")
                last_message = after_messages[-1]
                if hasattr(last_message, 'content'):
                    self.logger.info(f"{self.config.name if self.config.name else 'unknown_agent'} - text found in 'content' key of last message")
                    text_response = last_message.content
                else:
                    self.logger.info(f"{self.config.name if self.config.name else 'unknown_agent'} - used str representation of last message as text")
                    text_response = str(last_message)

            self.logger.info(f"{self.config.name if self.config.name else 'unknown_agent'} - detected images: {len(images)}")
            
            return {
                "text": text_response,
                "images": images
            }

        else:
            raise ValueError("The LangGraph ReAct agent has not been initialized")


    async def stream(self, query):
        """Stream agent responses as they are generated."""
        import re
        self.logger.info(f"{self.config.name or 'unknown_agent'} - streaming query: {query}")

        if not self.agent:
            raise ValueError("The LangGraph ReAct agent has not been initialized")

        # Agent state configuration
        actor_id_hash = hashlib.sha256((self.config.identity or "unknown").encode()).hexdigest()
        config = {
            "configurable": {"thread_id": self.config.thread, "actor_id": actor_id_hash},
            "identity": {"username": self.config.identity}
        }

        collected_images = []
        streamed_text = []
        chart_pattern = r'<chart>(data:image/svg\+xml;base64,[^<]+)</chart>'

        # Stream events from the agent
        async for event in self.agent.astream_events(
            {"messages": [HumanMessage(content=query)]},
            config=config,
            version="v2"
        ):
            kind = event.get("event")

            # Stream AI message content chunks
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    content = chunk.content
                    # Ensure content is a string
                    if isinstance(content, str):
                        streamed_text.append(content)
                        yield {"type": "text", "content": content}
                    elif isinstance(content, list):
                        # Handle list of content blocks (e.g., from Claude)
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                streamed_text.append(text)
                                yield {"type": "text", "content": text}
                            elif isinstance(block, str):
                                streamed_text.append(block)
                                yield {"type": "text", "content": block}

            # Notify when tool is called
            elif kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                yield {"type": "tool_start", "tool": tool_name}

            # Capture chart images from tool outputs
            elif kind == "on_tool_end":
                tool_name = event.get("name", "unknown")
                output = event.get("data", {}).get("output", "")
                self.logger.info(f"Tool {tool_name} output type: {type(output).__name__}")
                # Convert output to string if needed
                output_str = ""
                if hasattr(output, 'content'):
                    output_str = output.content if isinstance(output.content, str) else str(output.content)
                elif isinstance(output, str):
                    output_str = output
                else:
                    output_str = str(output)
                self.logger.info(f"Tool {tool_name} output preview: {output_str[:300] if output_str else 'None'}")

                # Collect any chart images from tool output
                matches = re.findall(chart_pattern, output_str)
                collected_images.extend(matches)

                yield {"type": "tool_end", "tool": tool_name}

        # Emit deduplicated chart images that weren't already in the streamed text
        full_streamed = ''.join(streamed_text)
        seen = set()
        for img in collected_images:
            if img not in full_streamed and img not in seen:
                seen.add(img)
                yield {"type": "text", "content": f"<chart>{img}</chart>"}

        # Signal completion
        yield {"type": "done"}


    async def chat_loop(self):
        print("\n========================================================================================")
        print(self.config.name)
        print("----------------------------------------------------------------------------------------")
        print(self.config.description)
        print("\nHow to use:")
        print("Type your queries, or instructions for the agent or type 'quit' or press CTRL+C to exit.")
        print("========================================================================================")
    
        while True:
            try:
                query = input("\n>>> Your query: ").strip()
    
                if query.lower() == 'quit':
                    print("\n\nExiting...")
                    break
    
                response = await self.invoke(query)
                
                # Handle dict response with text and images
                if isinstance(response, dict):
                    print(f"\n{response.get('text', '')}")
                    if response.get('images'):
                        print(f"\n[Generated {len(response['images'])} chart(s)]")
                else:
                    print(f"\n{response}")
    
            except (KeyboardInterrupt, asyncio.CancelledError):
                print("\n\nExiting...")
                break
            except Exception as e:
                error_msg = f"Error in chat loop: {str(e)}"
                self.logger.error(error_msg)
                print(f"\n{error_msg}")
                break