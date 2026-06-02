# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any

from common.types import AgentConfig


class BaseAgent:
    """Base agent class for Strands SDK agents.

    Provides shared lifecycle methods (setup, invoke, chat_loop) that mirror
    the LangGraph BaseAgent interface. Subclasses override setup() to configure
    their specific MCP server connections, tools, and Strands Agent instance.
    """

    def __init__(self, config: AgentConfig):
        self.agent = None
        self.config = config

        # Configure logging
        self.logger = logging.getLogger(config.name)
        self.logger.setLevel(logging.INFO)
        if config.log_file:
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)

        # Inject username into system prompt
        self.system_prompt = f"{self.config.system_prompt}\n\n# Current customer username is: {self.config.identity}\n\n"

    def setup(self):
        """Set up the agent. Subclasses override this to initialize model, tools, and Strands Agent."""
        self.logger.info("Setting up the Strands agent...")

    def invoke(self, query: str, use_fresh_thread: bool = False) -> Dict[str, Any]:
        """Invoke the agent with a query and return the response.

        Args:
            query: The user's input query.
            use_fresh_thread: Ignored in Strands (no built-in thread checkpointing).

        Returns:
            Dict with 'text' and 'images' keys.
        """
        self.logger.info(f"{self.config.name or 'unknown_agent'} - invoked with query: {query}")

        if not self.agent:
            raise ValueError("The Strands agent has not been initialized. Call setup() first.")

        try:
            response = self.agent(query)
            result_text = str(response)

            # Extract chart images from the response
            images = []
            chart_pattern = r'<chart>(data:image/svg\+xml;base64,[^<]+)</chart>'
            matches = re.findall(chart_pattern, result_text)
            images.extend(matches)

            self.logger.info(f"{self.config.name or 'unknown_agent'} - detected images: {len(images)}")

            return {
                "text": result_text,
                "images": images
            }

        except Exception as e:
            self.logger.error(f"Error invoking agent: {str(e)}")
            return {
                "text": f"Error: {str(e)}",
                "images": []
            }

    def chat_loop(self):
        """Interactive chat loop for CLI usage."""
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

                response = self.invoke(query)

                if isinstance(response, dict):
                    print(f"\n{response.get('text', '')}")
                    if response.get('images'):
                        print(f"\n[Generated {len(response['images'])} chart(s)]")
                else:
                    print(f"\n{response}")

            except (KeyboardInterrupt, EOFError):
                print("\n\nExiting...")
                break
            except Exception as e:
                error_msg = f"Error in chat loop: {str(e)}"
                self.logger.error(error_msg)
                print(f"\n{error_msg}")
                break