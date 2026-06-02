#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
DataViz Agent - Strands SDK equivalent of the LangGraph dataviz_agent.py.
Calls DataViz MCP server via HTTPS gateway.
"""

import os
import sys
import argparse
import json
import logging
import base64
import threading
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from contextvars import ContextVar
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands.tools.mcp import MCPClient

# Configure logging
logger = logging.getLogger("dataviz-agent")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('/tmp/dataviz_agent.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

_ = load_dotenv('.env')


# --- Chart helper ---
# Side-channel to capture chart outputs (bypasses conversation manager truncation)
# Uses ContextVar for thread-safety across concurrent AgentCore invocations
_captured_charts: ContextVar[list] = ContextVar('_captured_charts')

def _fig_to_svg_base64(fig) -> str:
    """Convert matplotlib figure to base64 SVG."""
    buffer = BytesIO()
    fig.savefig(buffer, format='svg', bbox_inches='tight')
    buffer.seek(0)
    svg_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    chart_tag = f"<chart>data:image/svg+xml;base64,{svg_b64}</chart>"
    try:
        _captured_charts.get().append(chart_tag)
    except LookupError:
        _captured_charts.set([chart_tag])
    return chart_tag


# --- Chart tools ---
_chart_lock = threading.Lock()

@tool
def create_bar_chart(data: str, x_column: str, y_column: str, title: str = "Bar Chart",
                     x_label: str = None, y_label: str = None) -> str:
    """Create a bar/column chart from CSV data. Returns base64-encoded SVG image.

    Args:
        data: CSV formatted data as string
        x_column: Column name for x-axis (categories)
        y_column: Column name for y-axis (values)
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
    """
    with _chart_lock:
        df = pd.read_csv(StringIO(data))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df[x_column], df[y_column])
        ax.set_title(title)
        ax.set_xlabel(x_label or x_column)
        ax.set_ylabel(y_label or y_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return _fig_to_svg_base64(fig)


@tool
def create_line_chart(data: str, x_column: str, y_column: str, title: str = "Line Chart",
                      x_label: str = None, y_label: str = None, group_column: str = None) -> str:
    """Create a line chart from CSV data. Returns base64-encoded SVG image.

    Args:
        data: CSV formatted data as string
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        group_column: Column to group by for multiple lines
    """
    with _chart_lock:
        df = pd.read_csv(StringIO(data))
        fig, ax = plt.subplots(figsize=(10, 6))
        if group_column and group_column in df.columns:
            for name, group in df.groupby(group_column):
                ax.plot(group[x_column], group[y_column], label=name, marker='o')
            ax.legend()
        else:
            ax.plot(df[x_column], df[y_column], marker='o')
        ax.set_title(title)
        ax.set_xlabel(x_label or x_column)
        ax.set_ylabel(y_label or y_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return _fig_to_svg_base64(fig)


@tool
def create_pie_chart(data: str, values_column: str, names_column: str, title: str = "Pie Chart") -> str:
    """Create a pie chart from CSV data. Returns base64-encoded SVG image.

    Args:
        data: CSV formatted data as string
        values_column: Column containing values
        names_column: Column containing category names
        title: Chart title
    """
    with _chart_lock:
        df = pd.read_csv(StringIO(data))
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(
            df[values_column], labels=df[names_column],
            autopct='%1.1f%%', startangle=90
        )
        ax.set_title(title)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        plt.tight_layout()
        return _fig_to_svg_base64(fig)


@tool
def create_scatter_plot(data: str, x_column: str, y_column: str, title: str = "Scatter Plot",
                        x_label: str = None, y_label: str = None, color_column: str = None) -> str:
    """Create a scatter plot from CSV data. Returns base64-encoded SVG image.

    Args:
        data: CSV formatted data as string
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        color_column: Column for point colors
    """
    with _chart_lock:
        df = pd.read_csv(StringIO(data))
        fig, ax = plt.subplots(figsize=(10, 6))
        if color_column and color_column in df.columns:
            for cat in df[color_column].unique():
                mask = df[color_column] == cat
                ax.scatter(df[x_column][mask], df[y_column][mask], label=cat, alpha=0.7)
            ax.legend()
        else:
            ax.scatter(df[x_column], df[y_column], alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(x_label or x_column)
        ax.set_ylabel(y_label or y_column)
        plt.tight_layout()
        return _fig_to_svg_base64(fig)


@tool
def analyze_data_structure(data: str) -> str:
    """Analyze CSV data structure and recommend chart types.

    Args:
        data: CSV formatted data as string
    """
    df = pd.read_csv(StringIO(data))
    analysis = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "recommendations": []
    }
    num_numeric = len(analysis["numeric_columns"])
    num_categorical = len(analysis["categorical_columns"])
    if num_numeric >= 2:
        analysis["recommendations"].append("Scatter plot or line chart for numeric relationships")
    if num_categorical >= 1 and num_numeric >= 1:
        analysis["recommendations"].append("Bar chart for comparing values across categories")
    if num_categorical == 1 and num_numeric == 1:
        analysis["recommendations"].append("Pie chart if showing parts of a whole")
    return json.dumps(analysis, indent=2)


@tool
def getDateTime() -> str:
    """Get the date and time in the local timezone.

    Returns:
        str: The current date and time in ISO format
    """
    return str(datetime.now().astimezone().isoformat())


BUILTIN_TOOLS = [
    create_bar_chart, create_line_chart, create_pie_chart,
    create_scatter_plot, analyze_data_structure, getDateTime
]


@dataclass
class DataVizConfig:
    """Configuration for the DataViz Agent."""
    model: str = "global.anthropic.claude-sonnet-4-6"
    user: str = "unknown"
    thread_id: Optional[str] = None
    https_url: Optional[str] = None
    https_headers: dict = None

    def __post_init__(self):
        if self.https_headers is None:
            self.https_headers = {}


class DataVizAgent:
    """DataViz Agent using Strands SDK with HTTPS gateway support."""

    SYSTEM_PROMPT = """You are a data visualization expert agent. Your role is to:

1. Create clear, informative charts using the available chart tools
2. Provide brief insights about the visualization

IMPORTANT: Always call tools ONE AT A TIME. Never issue multiple tool calls in parallel. Wait for each tool result before deciding the next step.

When given data and a description:
- If the user specifies a chart type, create that chart directly without calling analyze_data_structure first
- If no chart type is specified, analyze the data structure to choose the best visualization
- Create the chart with clear labels and titles
- Return the chart as a base64 encoded SVG image

Available chart types:
- Bar charts: For comparing categories
- Line charts: For trends over time or continuous data
- Scatter plots: For exploring relationships between variables
- Pie charts: For showing parts of a whole

Create professional-looking charts with appropriate titles and labels."""

    def __init__(self, config: DataVizConfig):
        self.config = config
        self.agent = None
        self._builtin_agent = None
        self.model = None
        self.mcp_client = None

    def setup(self):
        """Set up the agent with Amazon Bedrock model."""
        logger.info("Setting up DataViz agent...")

        self.model = BedrockModel(
            model_id=self.config.model,
            region_name=os.getenv('MODEL_REGION', os.getenv('AWS_REGION', 'us-east-1'))
        )
        logger.info("Amazon Bedrock model initialized")

        system_prompt = f"Current user: {self.config.user}\n\n{self.SYSTEM_PROMPT}"

        # If HTTPS URL provided, set up MCP client for gateway
        # When a gateway is configured, we use ONLY gateway tools so that
        # AgentCore policy enforcement is applied to every tool call.
        # Local built-in tools are a fallback for when no gateway is available.
        if self.config.https_url:
            from strands.tools.mcp import MCPClient
            from mcp.client.streamable_http import streamablehttp_client

            headers = dict(self.config.https_headers) if self.config.https_headers else {}
            self.mcp_client = MCPClient(
                lambda: streamablehttp_client(url=self.config.https_url, headers=headers)
            )
            logger.info(f"Configured HTTPS MCP client at {self.config.https_url}")
        else:
            # Use built-in tools only (no gateway = no policy enforcement)
            self.agent = Agent(
                model=self.model,
                system_prompt=system_prompt,
                tools=list(BUILTIN_TOOLS)
            )
            logger.info("DataViz agent created with built-in tools")

        self._system_prompt = system_prompt
        logger.info("DataViz agent setup complete")

    def visualize_data(self, data: str, description: str) -> str:
        """Create a visualization from data and description.

        Routes through the gateway MCP client when configured so AgentCore
        Cedar policies (e.g. block_pie_charts) are enforced on every chart
        tool call. Falls back to local matplotlib tools only when no
        gateway URL is configured (e.g. local CLI runs without a gateway).
        Charts are surfaced via <chart>...</chart> blocks emitted by the
        chart tools and extracted by the runtime adapter from the agent's
        message history.
        """
        query = f"Here's my data:\n{data}\n\n{description}"
        _captured_charts.set([])
        try:
            from strands.tools.executors.sequential import SequentialToolExecutor
            executor = SequentialToolExecutor()
        except ImportError:
            executor = None

        if self.mcp_client:
            # Gateway path — every chart tool call is policy-enforced
            logger.info(f"visualize_data via gateway: {query[:100]}")
            try:
                with self.mcp_client:
                    mcp_tools = self.mcp_client.list_tools_sync()
                    gw_agent = Agent(
                        model=self.model,
                        system_prompt=self._system_prompt,
                        tools=[getDateTime] + mcp_tools,
                        **({"tool_executor": executor} if executor else {})
                    )
                    response = gw_agent(query)
                    return str(response)
            except Exception as e:
                logger.error(f"Error in visualize_data (gateway): {str(e)}")
                return f"Error: {str(e)}"

        # Local fallback path — used when no gateway URL is configured
        logger.info(f"visualize_data via builtin tools: {query[:100]}")
        if not self._builtin_agent:
            self._builtin_agent = Agent(
                model=self.model,
                system_prompt=self._system_prompt,
                tools=list(BUILTIN_TOOLS),
                **({"tool_executor": executor} if executor else {})
            )
        try:
            response = self._builtin_agent(query)
            result = str(response)
        except Exception as e:
            logger.error(f"Error in visualize_data (builtin): {str(e)}")
            result = f"Error: {str(e)}"
        charts = _captured_charts.get()
        if charts:
            chart_tags = "\n".join(charts)
            return f"{result}\n{chart_tags}"
        return result

    def invoke_agent(self, query: str) -> str:
        """Invoke the agent with a query (uses gateway agent if available)."""
        if not query or not query.strip():
            return "Please provide a valid query."
        logger.info(f"Query: {query}")
        try:
            if not self.agent:
                self.agent = Agent(
                    model=self.model,
                    system_prompt=self._system_prompt,
                    tools=list(BUILTIN_TOOLS)
                )
            response = self.agent(query)
            return str(response)
        except Exception as e:
            logger.error(f"Error invoking agent: {str(e)}")
            return f"Error: {str(e)}"

    def chat_loop(self):
        """Interactive chat loop for CLI usage."""
        print("\n==================== Data Visualization Agent (Strands) ====================")
        print("This agent creates data visualizations from your datasets.")
        print("\nPaste your input, then type END on a new line to submit.")
        print("Type 'quit' to exit.")
        print("============================================================================")

        if self.mcp_client:
            with self.mcp_client:
                mcp_tools = self.mcp_client.list_tools_sync()
                logger.info(f"Retrieved {len(mcp_tools)} MCP tools from gateway")
                # Use ONLY gateway tools (+ getDateTime) so policy enforcement applies.
                # Do NOT include BUILTIN_TOOLS — they duplicate gateway tools and bypass policies.
                self.agent = Agent(
                    model=self.model,
                    system_prompt=self._system_prompt,
                    tools=[getDateTime] + mcp_tools
                )
                self._run_chat()
        else:
            self._run_chat()

    def _run_chat(self):
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

            except (EOFError, KeyboardInterrupt):
                print("\n\nExiting...")
                return
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                print(f"\nError in chat loop: {str(e)}")


def main():
    _ = load_dotenv('.env')

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user', default=os.getenv('USER', 'unknown'))
    parser.add_argument('-t', '--thread', default=None)
    parser.add_argument('--https-url')
    parser.add_argument('-m', '--model', default="global.anthropic.claude-sonnet-4-6")
    args = parser.parse_args()

    config = DataVizConfig(
        model=args.model,
        user=args.user,
        thread_id=args.thread or args.user,
        https_url=args.https_url
    )

    try:
        agent = DataVizAgent(config)
        agent.setup()
        agent.chat_loop()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        logger.error(f"Main error: {str(e)}")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
