# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
Data Visualization Agent Module

This module provides a data visualization agent that can be used both as a standalone
CLI application and as an importable Python module.

The agent accepts datasets and explanations to generate appropriate data visualizations
using matplotlib and returns base64-encoded PNG chart images.

Usage as a module:
    from modules.dataviz import DataVizAgent, create_dataviz_agent
    
    # Simple usage
    agent = await create_dataviz_agent()
    result = await agent.visualize_data(data_csv, "Create a bar chart of sales by month")
    
    # Advanced usage with custom config
    config = DataVizConfig(model="global.anthropic.claude-sonnet-4-6", user="analyst")
    agent = DataVizAgent(config)
    await agent.setup()
    result = await agent.visualize_data(data_csv, description)

Usage as CLI:
    python modules/03/langgraph/dataviz.py -m global.anthropic.claude-sonnet-4-6
"""

import os
import sys
import argparse
import traceback
import asyncio
import logging
import base64
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

# Configure logging
logger = logging.getLogger("dataviz-agent")

# Global configuration for chart tools
_chart_config = {
    "save_to_file": False,
    "output_dir": ".."
}


def set_chart_config(save_to_file: bool = False, output_dir: str = "."):
    """Set global configuration for chart creation tools."""
    global _chart_config
    _chart_config["save_to_file"] = save_to_file
    _chart_config["output_dir"] = output_dir


def create_matplotlib_chart_image(fig, chart_type: str = "chart", save_to_file: bool = False, output_dir: str = ".") -> str:
    """Convert matplotlib figure to base64 encoded image or save to file."""
    try:
        if save_to_file:
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{chart_type}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save to file
            fig.savefig(filepath, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)  # Clean up
            
            # Return file path information
            abs_path = os.path.abspath(filepath)
            return f"<chart>Chart saved to file: {abs_path}</chart>"
        else:
            # Return base64 encoded image
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)  # Clean up
            return f"<chart>data:image/png;base64,{img_b64}</chart>"
    except Exception as e:
        plt.close(fig)  # Clean up even on error
        raise e


class MCPConfigLoader:
    """Loads and validates MCP server configuration from YAML files."""
    
    def __init__(self, config_file_path: Optional[str] = None):
        """Initialize the config loader with optional file path.
        
        Args:
            config_file_path: Path to YAML configuration file. Defaults to 'dataviz.yml'
        """
        self.config_file_path = config_file_path or "dataviz.yml"
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Dict containing MCP server configurations, empty dict if file not found or invalid
        """
        try:
            if not os.path.exists(self.config_file_path):
                logger.info(f"MCP config file not found: {self.config_file_path}")
                return {}
            
            with open(self.config_file_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file) or {}
                logger.info(f"Loaded MCP configuration from {self.config_file_path}")
                return config
                
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file {self.config_file_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config file {self.config_file_path}: {e}")
            return {}
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure and required fields.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, dict):
            logger.error("Configuration must be a dictionary")
            return False
        
        if 'mcp_servers' not in config:
            logger.warning("No 'mcp_servers' section found in configuration")
            return True  # Empty config is valid
        
        mcp_servers = config['mcp_servers']
        if not isinstance(mcp_servers, dict):
            logger.error("'mcp_servers' must be a dictionary")
            return False
        
        for server_name, server_config in mcp_servers.items():
            if not isinstance(server_config, dict):
                logger.error(f"Server '{server_name}' configuration must be a dictionary")
                return False
            
            # Validate required fields
            if 'command' not in server_config:
                logger.error(f"Server '{server_name}' missing required 'command' field")
                return False
            
            # Validate optional fields
            if 'args' in server_config and not isinstance(server_config['args'], list):
                logger.error(f"Server '{server_name}' 'args' must be a list")
                return False
            
            if 'env' in server_config and not isinstance(server_config['env'], dict):
                logger.error(f"Server '{server_name}' 'env' must be a dictionary")
                return False
            
            if 'enabled' in server_config and not isinstance(server_config['enabled'], bool):
                logger.error(f"Server '{server_name}' 'enabled' must be a boolean")
                return False
            
            if 'transport' in server_config and server_config['transport'] != 'stdio':
                logger.error(f"Server '{server_name}' only 'stdio' transport is supported")
                return False
        
        return True
    
    def parse_servers(self, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Parse server configurations for MultiServerMCPClient.
        
        Args:
            config: Validated configuration dictionary
            
        Returns:
            Dictionary formatted for MultiServerMCPClient
        """
        if 'mcp_servers' not in config:
            return {}
        
        parsed_servers = {}
        mcp_servers = config['mcp_servers']
        
        for server_name, server_config in mcp_servers.items():
            # Skip disabled servers
            if not server_config.get('enabled', True):
                logger.info(f"Skipping disabled MCP server: {server_name}")
                continue
            
            parsed_config = {
                'command': server_config['command'],
                'args': server_config.get('args', []),
                'transport': server_config.get('transport', 'stdio'),
            }
            
            # Add environment variables if specified
            if 'env' in server_config:
                parsed_config['env'] = server_config['env']
            
            parsed_servers[server_name] = parsed_config
            logger.info(f"Configured MCP server: {server_name}")
        
        return parsed_servers


@dataclass
class DataVizConfig:
    """Configuration for the DataViz Agent."""
    model: str = "global.anthropic.claude-sonnet-4-6"
    user: str = "unknown"
    thread_id: Optional[str] = None
    mcp_config_file: Optional[str] = None
    log_level: str = "INFO"
    log_file: Optional[str] = None
    save_charts_to_file: bool = False
    chart_output_dir: str = "."


# Chart creation tools
@tool(parse_docstring=True)
def create_bar_chart(data: str, x_column: str, y_column: str, title: str = "Bar Chart", 
                     x_label: str = None, y_label: str = None) -> str:
    """Create a bar/column chart from data.
    
    Args:
        data: CSV formatted data as string
        x_column: Name of the column for x-axis (categories)
        y_column: Name of the column for y-axis (values)
        title: Chart title
        x_label: X-axis label (optional)
        y_label: Y-axis label (optional)
    """
    try:
        df = pd.read_csv(StringIO(data))
        
        # Create matplotlib chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df[x_column], df[y_column])
        ax.set_title(title)
        ax.set_xlabel(x_label or x_column)
        ax.set_ylabel(y_label or y_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return create_matplotlib_chart_image(
            fig, 
            chart_type="bar_chart", 
            save_to_file=_chart_config["save_to_file"], 
            output_dir=_chart_config["output_dir"]
        )
        
    except Exception as e:
        return f"Error creating bar chart: {str(e)}"


@tool(parse_docstring=True)
def create_line_chart(data: str, x_column: str, y_column: str, title: str = "Line Chart",
                      x_label: str = None, y_label: str = None, group_column: str = None) -> str:
    """Create a line chart from data.
    
    Args:
        data: CSV formatted data as string
        x_column: Name of the column for x-axis
        y_column: Name of the column for y-axis
        title: Chart title
        x_label: X-axis label (optional)
        y_label: Y-axis label (optional)
        group_column: Column to group by for multiple lines (optional)
    """
    try:
        df = pd.read_csv(StringIO(data))
        
        # Create matplotlib chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if group_column and group_column in df.columns:
            # Group by the specified column and plot multiple lines
            for group_name, group_data in df.groupby(group_column):
                ax.plot(group_data[x_column], group_data[y_column], label=group_name, marker='o')
            ax.legend()
        else:
            ax.plot(df[x_column], df[y_column], marker='o')
            
        ax.set_title(title)
        ax.set_xlabel(x_label or x_column)
        ax.set_ylabel(y_label or y_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return create_matplotlib_chart_image(
            fig, 
            chart_type="line_chart", 
            save_to_file=_chart_config["save_to_file"], 
            output_dir=_chart_config["output_dir"]
        )
        
    except Exception as e:
        return f"Error creating line chart: {str(e)}"


@tool(parse_docstring=True)
def create_scatter_plot(data: str, x_column: str, y_column: str, title: str = "Scatter Plot",
                        x_label: str = None, y_label: str = None, size_column: str = None,
                        color_column: str = None) -> str:
    """Create a scatter plot from data.
    
    Args:
        data: CSV formatted data as string
        x_column: Name of the column for x-axis
        y_column: Name of the column for y-axis
        title: Chart title
        x_label: X-axis label (optional)
        y_label: Y-axis label (optional)
        size_column: Column to determine point sizes (optional)
        color_column: Column to determine point colors (optional)
    """
    try:
        df = pd.read_csv(StringIO(data))
        
        # Create matplotlib chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Handle color and size columns
        if color_column and color_column in df.columns:
            # Use different colors for different categories
            for category in df[color_column].unique():
                mask = df[color_column] == category
                size_values = df[size_column][mask] if size_column and size_column in df.columns else 50
                ax.scatter(df[x_column][mask], df[y_column][mask], 
                          label=category, s=size_values, alpha=0.7)
            ax.legend()
        else:
            size_values = df[size_column] if size_column and size_column in df.columns else 50
            ax.scatter(df[x_column], df[y_column], s=size_values, alpha=0.7)
            
        ax.set_title(title)
        ax.set_xlabel(x_label or x_column)
        ax.set_ylabel(y_label or y_column)
        plt.tight_layout()
        
        return create_matplotlib_chart_image(
            fig, 
            chart_type="scatter_plot", 
            save_to_file=_chart_config["save_to_file"], 
            output_dir=_chart_config["output_dir"]
        )
        
    except Exception as e:
        return f"Error creating scatter plot: {str(e)}"


@tool(parse_docstring=True)
def create_pie_chart(data: str, values_column: str, names_column: str, title: str = "Pie Chart") -> str:
    """Create a pie chart from data.
    
    Args:
        data: CSV formatted data as string
        values_column: Name of the column containing values
        names_column: Name of the column containing category names
        title: Chart title
    """
    try:
        df = pd.read_csv(StringIO(data))
        
        # Create matplotlib chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(df[values_column], labels=df[names_column], 
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title(title)
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        return create_matplotlib_chart_image(
            fig, 
            chart_type="pie_chart", 
            save_to_file=_chart_config["save_to_file"], 
            output_dir=_chart_config["output_dir"]
        )
        
    except Exception as e:
        return f"Error creating pie chart: {str(e)}"


@tool(parse_docstring=True)
def analyze_data_structure(data: str) -> str:
    """Analyze the structure and characteristics of the provided data.
    
    Args:
        data: CSV formatted data as string
    """
    try:
        df = pd.read_csv(StringIO(data))
        
        analysis = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "column_types": {},
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "recommendations": []
        }
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            analysis["column_types"][col] = dtype
            
            if df[col].dtype in ['int64', 'float64']:
                analysis["numeric_columns"].append(col)
            elif df[col].dtype == 'object':
                # Check if it might be datetime
                try:
                    pd.to_datetime(df[col].head(), errors='raise', format='mixed')
                    analysis["datetime_columns"].append(col)
                except (ValueError, TypeError):
                    analysis["categorical_columns"].append(col)
        
        # Generate recommendations
        num_numeric = len(analysis["numeric_columns"])
        num_categorical = len(analysis["categorical_columns"])
        
        if num_numeric >= 2:
            analysis["recommendations"].append("Scatter plot for exploring relationships between numeric variables")
            analysis["recommendations"].append("Line chart if one variable represents time/sequence")
        
        if num_categorical >= 1 and num_numeric >= 1:
            analysis["recommendations"].append("Bar chart for comparing numeric values across categories")
            
        if num_categorical == 1 and num_numeric == 1:
            analysis["recommendations"].append("Pie chart if showing parts of a whole")
            
        if len(analysis["datetime_columns"]) >= 1 and num_numeric >= 1:
            analysis["recommendations"].append("Time series line chart for temporal data")
        
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        return f"Error analyzing data: {str(e)}"


@tool(parse_docstring=True)
def get_datetime() -> str:
    """Get the current date and time in ISO format."""
    return str(datetime.now().astimezone().isoformat())


class DataVizAgent:
    """Data Visualization Agent that can create charts from datasets."""
    
    def __init__(self, config: DataVizConfig):
        """Initialize the DataViz Agent with configuration.
        
        Args:
            config: DataVizConfig object with agent settings
        """
        self.config = config
        self.agent = None
        self.mcp = None
        self.mcp_config_loader = MCPConfigLoader(config.mcp_config_file)
        
        # Configure logging
        if config.log_file:
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setLevel(getattr(logging, config.log_level.upper()))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
        
        logger.setLevel(getattr(logging, config.log_level.upper()))
        logger.info("Initializing DataViz Agent...")
        
        # Set global chart configuration to always save to ../charts
        charts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'charts')
        set_chart_config(
            save_to_file=True,
            output_dir=charts_dir
        )
        
        # System prompt for data visualization
        self.system_prompt = """You are a data visualization expert agent. Your role is to:

1. Analyze provided datasets to understand their structure and characteristics
2. Recommend the most appropriate visualization type based on the data and user requirements
3. Create ONE clear, informative chart using matplotlib-based tools
4. Provide insights about the data and visualization choices

IMPORTANT: Always call tools ONE AT A TIME. Never issue multiple tool calls in parallel. Wait for each tool result before deciding the next step.

When given data and a description:
- First analyze the data structure using analyze_data_structure
- Consider the user's explanation and requirements
- Choose the most appropriate visualization type
- Create ONLY ONE chart with clear labels and titles
- Charts are saved as PNG files in the ../charts directory
- DO NOT create multiple charts unless explicitly requested

Available chart types:
- Bar charts: For comparing categories
- Line charts: For trends over time or continuous data
- Scatter plots: For exploring relationships between variables
- Pie charts: For showing parts of a whole

Charts are generated using matplotlib and saved as PNG files. Always explain your visualization choices and create professional-looking charts with appropriate titles and labels. Create only ONE chart per request."""

    async def setup(self):
        """Set up the agent with LLM, tools, and checkpointer."""
        try:
            logger.info("Setting up DataViz agent...")
            
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

            # Initialize tools
            all_tools = [
                create_bar_chart,
                create_line_chart,
                create_scatter_plot,
                create_pie_chart,
                analyze_data_structure,
                get_datetime
            ]

            # Load and configure MCP servers from YAML
            mcp_config = self.mcp_config_loader.load_config()
            if mcp_config and self.mcp_config_loader.validate_config(mcp_config):
                parsed_servers = self.mcp_config_loader.parse_servers(mcp_config)
                
                if parsed_servers:
                    try:
                        self.mcp = MultiServerMCPClient(parsed_servers)
                        mcp_tools = await self.mcp.get_tools()
                        all_tools.extend(mcp_tools)
                        logger.info(f"Added {len(mcp_tools)} MCP tools from {len(parsed_servers)} servers")
                    except Exception as e:
                        logger.error(f"Failed to initialize MCP servers: {e}")
                        logger.info("Continuing with built-in tools only")
                else:
                    logger.info("No enabled MCP servers found in configuration")
            else:
                logger.info("No valid MCP configuration found, using built-in tools only")

            # Set up checkpointer (using InMemorySaver for simplicity)
            checkpointer = InMemorySaver()
            logger.info("Using InMemorySaver for conversation state")

            # Create agent with increased recursion limit
            self.agent = create_agent(
                model=self.llm,
                tools=all_tools,
                system_prompt=self.system_prompt,
                checkpointer=checkpointer
            )
            
            # Configure recursion limit
            self.agent_config["configurable"]["recursion_limit"] = 50
            logger.info("DataViz agent setup complete")

        except Exception as e:
            logger.error(f"Error setting up agent: {str(e)}")
            raise

    async def visualize_data(self, data: str, description: str) -> str:
        """Create a visualization from data and description.
        
        Args:
            data: CSV formatted data as string
            description: Description of what to visualize
            
        Returns:
            str: Agent response with chart or error message
        """
        if not self.agent:
            raise RuntimeError("Agent not set up. Call setup() first.")
            
        query = f"Here's my data:\n{data}\n\n{description}"
        return await self.invoke_agent(query)

    async def invoke_agent(self, query: str) -> str:
        """Invoke the agent with a query.
        
        Args:
            query: The query/request for the agent
            
        Returns:
            str: Agent response
        """
        logger.info(f"Query: {query}")
        
        try:
            result = await self.agent.ainvoke({
                "messages": [HumanMessage(content=query)]
            }, config=self.agent_config)
            
            # Get the final state and return the last assistant message
            after_state = await self.agent.aget_state(self.agent_config)
            after_messages = after_state.values.get("messages", []) if after_state else []
            
            if after_messages:
                last_message = after_messages[-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
                else:
                    return str(last_message)
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error invoking agent: {str(e)}")
            return f"Error: {str(e)}"

    async def chat_loop(self):
        """Interactive chat loop for CLI usage."""
        charts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'charts')
        print("\n==================== Data Visualization Agent ====================")
        print("This agent creates data visualizations from your datasets.")
        print(f"Charts will be saved to: {os.path.abspath(charts_dir)}")
        print("\nPaste your input, then type END on a new line to submit.")
        print("Type 'quit' to exit.")
        print("===================================================================")

        while True:
            try:
                print("\n>>> ")
                lines = []
                while True:
                    line = input()
                    if line.strip().upper() == 'END':
                        break
                    if line.strip().lower() == 'quit' and not lines:
                        return
                    lines.append(line)
                
                user_input = '\n'.join(lines).strip()
                
                if not user_input:
                    continue

                response = await self.invoke_agent(user_input)
                print(f"\n{response}")

            except EOFError:
                return
            except KeyboardInterrupt:
                print("\n\nExiting...")
                return
            except Exception as e:
                error_msg = f"Error in chat loop: {str(e)}"
                logger.error(error_msg)
                print(f"\n{error_msg}")


# Convenience functions for module usage
async def create_dataviz_agent(config: Optional[DataVizConfig] = None) -> DataVizAgent:
    """Create and set up a DataViz agent with default or custom configuration.
    
    Args:
        config: Optional DataVizConfig. If None, uses defaults.
        
    Returns:
        DataVizAgent: Ready-to-use agent instance
    """
    if config is None:
        config = DataVizConfig()
    
    agent = DataVizAgent(config)
    await agent.setup()
    return agent


async def quick_visualize(data: str, description: str, model: str = "global.anthropic.claude-sonnet-4-6") -> str:
    """Quick visualization function for simple use cases.
    
    Args:
        data: CSV formatted data as string
        description: Description of what to visualize
        model: LLM model to use
        
    Returns:
        str: Visualization result
    """
    config = DataVizConfig(model=model)
    agent = await create_dataviz_agent(config)
    return await agent.visualize_data(data, description)


# CLI interface
def main():
    """Main function for CLI usage."""
    # Load environment variables
    load_dotenv('.env')
    
    # Define parser
    parser = argparse.ArgumentParser(description="Data Visualization Agent")
    parser.add_argument('-u', '--user', help="Username", default=os.getenv('USER', 'unknown'))
    parser.add_argument('-t', '--thread', help="Thread ID", default=None)
    parser.add_argument('-m', '--model', help="LLM model ID", default="global.anthropic.claude-sonnet-4-6")
    parser.add_argument('--mcp-config', help="Path to MCP configuration YAML file", default=None)
    parser.add_argument('--save-charts', action='store_true', help="Save charts as PNG files instead of base64")
    parser.add_argument('--chart-dir', help="Directory to save chart files", default=".")
    parser.add_argument('--log-level', help="Log level", default="INFO")
    parser.add_argument('--log-file', help="Log file path", default="dataviz_agent.log")
    
    args = parser.parse_args()
    
    # Create config from args
    config = DataVizConfig(
        model=args.model,
        user=args.user,
        thread_id=args.thread,
        mcp_config_file=args.mcp_config,
        save_charts_to_file=args.save_charts,
        chart_output_dir=args.chart_dir,
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    async def run_cli():
        try:
            logger.info("Starting DataViz application")
            
            agent = DataVizAgent(config)
            await agent.setup()
            await agent.chat_loop()

        except Exception as e:
            logger.error(f"Main error: {str(e)}")
            print(str(e))
            sys.exit(1)
    
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()