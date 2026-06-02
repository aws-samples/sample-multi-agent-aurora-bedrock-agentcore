# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
Data Visualization Agent Module (Strands SDK)

This module provides a data visualization agent that can be used both as a standalone
CLI application and as an importable Python module.

The agent accepts datasets and explanations to generate appropriate data visualizations
using matplotlib and returns base64-encoded PNG chart images.

Usage as a module:
    from modules.03.strands.dataviz import DataVizAgent, create_dataviz_agent
    
    # Simple usage
    agent = create_dataviz_agent()
    result = agent.visualize_data(data_csv, "Create a bar chart of sales by month")
    
    # Advanced usage with custom config
    config = DataVizConfig(model="global.anthropic.claude-sonnet-4-6", user="analyst")
    agent = DataVizAgent(config)
    agent.setup()
    result = agent.visualize_data(data_csv, description)

Usage as CLI:
    python modules/03/strands/dataviz.py -m global.anthropic.claude-sonnet-4-6
"""

import os
import sys
import argparse
import traceback
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
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands.tools.mcp import MCPClient
from mcp import StdioServerParameters

# Configure logging
logger = logging.getLogger("dataviz-agent")

# Global configuration for chart tools
_chart_config = {
    "save_to_file": False,
    "output_dir": "."
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{chart_type}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(filepath, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            abs_path = os.path.abspath(filepath)
            return f"<chart>Chart saved to file: {abs_path}</chart>"
        else:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            return f"<chart>data:image/png;base64,{img_b64}</chart>"
    except Exception as e:
        plt.close(fig)
        raise e


class MCPConfigLoader:
    """Loads and validates MCP server configuration from YAML files."""

    def __init__(self, config_file_path: Optional[str] = None):
        self.config_file_path = config_file_path or "dataviz.yml"

    def load_config(self) -> Dict[str, Any]:
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
        if not isinstance(config, dict):
            logger.error("Configuration must be a dictionary")
            return False
        if 'mcp_servers' not in config:
            logger.warning("No 'mcp_servers' section found in configuration")
            return True
        mcp_servers = config['mcp_servers']
        if not isinstance(mcp_servers, dict):
            logger.error("'mcp_servers' must be a dictionary")
            return False
        for server_name, server_config in mcp_servers.items():
            if not isinstance(server_config, dict):
                logger.error(f"Server '{server_name}' configuration must be a dictionary")
                return False
            if 'command' not in server_config:
                logger.error(f"Server '{server_name}' missing required 'command' field")
                return False
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

    def parse_servers(self, config: Dict[str, Any]) -> List[MCPClient]:
        """Parse server configurations into MCPClient instances for Strands.

        Returns:
            List of MCPClient instances for enabled servers
        """
        if 'mcp_servers' not in config:
            return []
        mcp_clients = []
        mcp_servers = config['mcp_servers']
        for server_name, server_config in mcp_servers.items():
            if not server_config.get('enabled', True):
                logger.info(f"Skipping disabled MCP server: {server_name}")
                continue
            env = server_config.get('env', None)
            client = MCPClient(
                lambda sc=server_config, e=env: StdioServerParameters(
                    command=sc['command'],
                    args=sc.get('args', []),
                    env=e
                )
            )
            mcp_clients.append(client)
            logger.info(f"Configured MCP server: {server_name}")
        return mcp_clients


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
    chart_output_dir: str = ".."


# Chart creation tools — using Strands @tool decorator
@tool
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

    Returns:
        str: Base64 encoded chart image or file path
    """
    try:
        df = pd.read_csv(StringIO(data))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df[x_column], df[y_column])
        ax.set_title(title)
        ax.set_xlabel(x_label or x_column)
        ax.set_ylabel(y_label or y_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return create_matplotlib_chart_image(
            fig, chart_type="bar_chart",
            save_to_file=_chart_config["save_to_file"],
            output_dir=_chart_config["output_dir"]
        )
    except Exception as e:
        return f"Error creating bar chart: {str(e)}"


@tool
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

    Returns:
        str: Base64 encoded chart image or file path
    """
    try:
        df = pd.read_csv(StringIO(data))
        fig, ax = plt.subplots(figsize=(10, 6))
        if group_column and group_column in df.columns:
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
            fig, chart_type="line_chart",
            save_to_file=_chart_config["save_to_file"],
            output_dir=_chart_config["output_dir"]
        )
    except Exception as e:
        return f"Error creating line chart: {str(e)}"


@tool
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

    Returns:
        str: Base64 encoded chart image or file path
    """
    try:
        df = pd.read_csv(StringIO(data))
        fig, ax = plt.subplots(figsize=(10, 6))
        if color_column and color_column in df.columns:
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
            fig, chart_type="scatter_plot",
            save_to_file=_chart_config["save_to_file"],
            output_dir=_chart_config["output_dir"]
        )
    except Exception as e:
        return f"Error creating scatter plot: {str(e)}"


@tool
def create_pie_chart(data: str, values_column: str, names_column: str, title: str = "Pie Chart") -> str:
    """Create a pie chart from data.

    Args:
        data: CSV formatted data as string
        values_column: Name of the column containing values
        names_column: Name of the column containing category names
        title: Chart title

    Returns:
        str: Base64 encoded chart image or file path
    """
    try:
        df = pd.read_csv(StringIO(data))
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(df[values_column], labels=df[names_column],
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title(title)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        plt.tight_layout()
        return create_matplotlib_chart_image(
            fig, chart_type="pie_chart",
            save_to_file=_chart_config["save_to_file"],
            output_dir=_chart_config["output_dir"]
        )
    except Exception as e:
        return f"Error creating pie chart: {str(e)}"


@tool
def analyze_data_structure(data: str) -> str:
    """Analyze the structure and characteristics of the provided data.

    Args:
        data: CSV formatted data as string

    Returns:
        str: JSON analysis of the data structure with recommendations
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
                try:
                    pd.to_datetime(df[col].head(), errors='raise', format='mixed')
                    analysis["datetime_columns"].append(col)
                except (ValueError, TypeError):
                    analysis["categorical_columns"].append(col)
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


@tool
def create_histogram(data: str, column: str, bins: int = 10, title: str = "Histogram",
                     x_label: str = None, y_label: str = "Frequency") -> str:
    """Create a histogram to show the distribution of numeric data.

    Args:
        data: CSV formatted data as string
        column: Name of the numeric column to create histogram for
        bins: Number of bins to group the data (default: 10)
        title: Chart title
        x_label: X-axis label (optional)
        y_label: Y-axis label (default: "Frequency")

    Returns:
        str: Base64 encoded chart image or file path
    """
    try:
        df = pd.read_csv(StringIO(data))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[column], bins=bins, edgecolor='black', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(x_label or column)
        ax.set_ylabel(y_label)
        plt.tight_layout()
        return create_matplotlib_chart_image(
            fig, chart_type="histogram",
            save_to_file=_chart_config["save_to_file"],
            output_dir=_chart_config["output_dir"]
        )
    except Exception as e:
        return f"Error creating histogram: {str(e)}"


@tool
def get_datetime() -> str:
    """Get the current date and time in ISO format.

    Returns:
        str: The current date and time in ISO format
    """
    return str(datetime.now().astimezone().isoformat())


class DataVizAgent:
    """Data Visualization Agent that can create charts from datasets (Strands SDK)."""

    # All built-in tools
    BUILTIN_TOOLS = [
        create_bar_chart,
        create_line_chart,
        create_scatter_plot,
        create_pie_chart,
        analyze_data_structure,
        get_datetime,
        create_histogram
    ]

    def __init__(self, config: DataVizConfig):
        self.config = config
        self.agent = None
        self.model = None
        self.mcp_clients = []
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
        set_chart_config(save_to_file=True, output_dir=charts_dir)

        # System prompt for data visualization
        self.system_prompt = """You are a data visualization expert agent. Your role is to:

1. Analyze provided datasets to understand their structure and characteristics
2. Recommend the most appropriate visualization type based on the data and user requirements
3. Create ONE clear, informative chart using matplotlib-based tools
4. Provide insights about the data and visualization choices

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

    def setup(self):
        """Set up the agent with Amazon Bedrock model and tools."""
        try:
            logger.info("Setting up DataViz agent...")

            model_id = self.config.model
            username = self.config.user

            # Initialize Amazon Bedrock model
            self.model = BedrockModel(
                model_id=model_id,
                region_name=os.getenv('MODEL_REGION', 'us-west-2')
            )
            logger.info("Amazon Bedrock model initialized")

            # Inject username into system prompt
            self.system_prompt = f"Current user: {username}\n\n{self.system_prompt}"

            # Load and configure MCP servers from YAML
            mcp_config = self.mcp_config_loader.load_config()
            if mcp_config and self.mcp_config_loader.validate_config(mcp_config):
                self.mcp_clients = self.mcp_config_loader.parse_servers(mcp_config)
                if self.mcp_clients:
                    logger.info(f"Configured {len(self.mcp_clients)} MCP clients")
                else:
                    logger.info("No enabled MCP servers found in configuration")
            else:
                logger.info("No valid MCP configuration found, using built-in tools only")

            logger.info("DataViz agent setup complete")

        except Exception as e:
            logger.error(f"Error setting up agent: {str(e)}")
            raise

    def _create_agent_with_tools(self, extra_tools=None):
        """Create a Strands Agent with built-in tools and optional extras."""
        all_tools = list(self.BUILTIN_TOOLS)
        if extra_tools:
            all_tools.extend(extra_tools)
        return Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=all_tools
        )

    def visualize_data(self, data: str, description: str) -> str:
        """Create a visualization from data and description.

        Args:
            data: CSV formatted data as string
            description: Description of what to visualize

        Returns:
            str: Agent response with chart or error message
        """
        if not self.model:
            raise RuntimeError("Agent not set up. Call setup() first.")

        query = f"Here's my data:\n{data}\n\n{description}"
        return self.invoke_agent(query)

    def invoke_agent(self, query: str) -> str:
        """Invoke the agent with a query.

        Args:
            query: The query/request for the agent

        Returns:
            str: Agent response
        """
        logger.info(f"Query: {query}")

        try:
            response = self.agent(query)
            return str(response)
        except Exception as e:
            logger.error(f"Error invoking agent: {str(e)}")
            return f"Error: {str(e)}"

    def chat_loop(self):
        """Interactive chat loop for CLI usage."""
        charts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'charts')
        print("\n==================== Data Visualization Agent (Strands) ====================")
        print("This agent creates data visualizations from your datasets.")
        print(f"Charts will be saved to: {os.path.abspath(charts_dir)}")
        print("\nPaste your input, then type END on a new line to submit.")
        print("Type 'quit' to exit.")
        print("============================================================================")

        # Gather MCP tools from all configured clients, keeping contexts open
        if self.mcp_clients:
            # Enter all MCP client contexts and collect tools
            mcp_contexts = [client.__enter__() for client in self.mcp_clients]
            try:
                mcp_tools = []
                for client in self.mcp_clients:
                    mcp_tools.extend(client.list_tools_sync())
                logger.info(f"Retrieved {len(mcp_tools)} MCP tools from {len(self.mcp_clients)} servers")
                self.agent = self._create_agent_with_tools(extra_tools=mcp_tools)
            except Exception as e:
                logger.error(f"Failed to initialize MCP servers: {e}")
                logger.info("Continuing with built-in tools only")
                self.agent = self._create_agent_with_tools()
        else:
            self.agent = self._create_agent_with_tools()

        logger.info("Agent created")

        try:
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

                    response = self.invoke_agent(user_input)
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
        finally:
            # Clean up MCP client contexts
            for client in self.mcp_clients:
                try:
                    client.__exit__(None, None, None)
                except Exception:
                    pass


# Convenience functions for module usage
def create_dataviz_agent(config: Optional[DataVizConfig] = None) -> DataVizAgent:
    """Create and set up a DataViz agent with default or custom configuration.

    Args:
        config: Optional DataVizConfig. If None, uses defaults.

    Returns:
        DataVizAgent: Ready-to-use agent instance
    """
    if config is None:
        config = DataVizConfig()
    agent = DataVizAgent(config)
    agent.setup()
    return agent


def quick_visualize(data: str, description: str, model: str = "global.anthropic.claude-sonnet-4-6") -> str:
    """Quick visualization function for simple use cases.

    Args:
        data: CSV formatted data as string
        description: Description of what to visualize
        model: LLM model to use

    Returns:
        str: Visualization result
    """
    config = DataVizConfig(model=model)
    agent = create_dataviz_agent(config)
    return agent.visualize_data(data, description)


# CLI interface
def main():
    """Main function for CLI usage."""
    load_dotenv('.env')

    parser = argparse.ArgumentParser(description="Data Visualization Agent (Strands)")
    parser.add_argument('-u', '--user', help="Username", default=os.getenv('USER', 'unknown'))
    parser.add_argument('-t', '--thread', help="Thread ID", default=None)
    parser.add_argument('-m', '--model', help="LLM model ID", default="global.anthropic.claude-sonnet-4-6")
    parser.add_argument('--mcp-config', help="Path to MCP configuration YAML file", default=None)
    parser.add_argument('--save-charts', action='store_true', help="Save charts as PNG files instead of base64")
    parser.add_argument('--chart-dir', help="Directory to save chart files", default=".")
    parser.add_argument('--log-level', help="Log level", default="INFO")
    parser.add_argument('--log-file', help="Log file path", default="dataviz_agent.log")

    args = parser.parse_args()

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

    try:
        logger.info("Starting DataViz application")
        agent = DataVizAgent(config)
        agent.setup()
        agent.chat_loop()
    except Exception as e:
        logger.error(f"Main error: {str(e)}")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()