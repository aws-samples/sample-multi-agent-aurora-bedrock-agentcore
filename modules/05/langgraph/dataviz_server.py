#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
DataViz MCP Server - Exposes chart creation tools via MCP protocol.
Can be deployed to AgentCore Gateway as a Lambda function.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
import logging
import argparse
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from typing import Any, Dict, List, Sequence
from mcp.server import Server
from mcp.types import Tool, TextContent


class DataVizMCPServer:
    def __init__(self, name: str = "dataviz-server", log_file: str = None):
        self.server = Server(name)
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if log_file and __name__ == "__main__":
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
        self.setup_handlers()

    def get_tool_schema(self):
        return [
            {
                "name": "create_bar_chart",
                "description": "Create a bar/column chart from CSV data. Returns base64-encoded SVG image.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "CSV formatted data as string"},
                        "x_column": {"type": "string", "description": "Column name for x-axis (categories)"},
                        "y_column": {"type": "string", "description": "Column name for y-axis (values)"},
                        "title": {"type": "string", "description": "Chart title", "default": "Bar Chart"},
                        "x_label": {"type": "string", "description": "X-axis label"},
                        "y_label": {"type": "string", "description": "Y-axis label"}
                    },
                    "required": ["data", "x_column", "y_column"]
                }
            },
            {
                "name": "create_line_chart",
                "description": "Create a line chart from CSV data. Returns base64-encoded SVG image.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "CSV formatted data as string"},
                        "x_column": {"type": "string", "description": "Column name for x-axis"},
                        "y_column": {"type": "string", "description": "Column name for y-axis"},
                        "title": {"type": "string", "description": "Chart title", "default": "Line Chart"},
                        "x_label": {"type": "string", "description": "X-axis label"},
                        "y_label": {"type": "string", "description": "Y-axis label"},
                        "group_column": {"type": "string", "description": "Column to group by for multiple lines"}
                    },
                    "required": ["data", "x_column", "y_column"]
                }
            },
            {
                "name": "create_pie_chart",
                "description": "Create a pie chart from CSV data. Returns base64-encoded SVG image.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "CSV formatted data as string"},
                        "values_column": {"type": "string", "description": "Column containing values"},
                        "names_column": {"type": "string", "description": "Column containing category names"},
                        "title": {"type": "string", "description": "Chart title", "default": "Pie Chart"}
                    },
                    "required": ["data", "values_column", "names_column"]
                }
            },
            {
                "name": "create_scatter_plot",
                "description": "Create a scatter plot from CSV data. Returns base64-encoded SVG image.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "CSV formatted data as string"},
                        "x_column": {"type": "string", "description": "Column name for x-axis"},
                        "y_column": {"type": "string", "description": "Column name for y-axis"},
                        "title": {"type": "string", "description": "Chart title", "default": "Scatter Plot"},
                        "x_label": {"type": "string", "description": "X-axis label"},
                        "y_label": {"type": "string", "description": "Y-axis label"},
                        "color_column": {"type": "string", "description": "Column for point colors"}
                    },
                    "required": ["data", "x_column", "y_column"]
                }
            },
            {
                "name": "analyze_data_structure",
                "description": "Analyze CSV data structure and recommend chart types.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "CSV formatted data as string"}
                    },
                    "required": ["data"]
                }
            }
        ]

    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        self.logger.info(f"Tool called: {name}")
        try:
            if name == "create_bar_chart":
                return self._create_bar_chart(arguments)
            elif name == "create_line_chart":
                return self._create_line_chart(arguments)
            elif name == "create_pie_chart":
                return self._create_pie_chart(arguments)
            elif name == "create_scatter_plot":
                return self._create_scatter_plot(arguments)
            elif name == "analyze_data_structure":
                return self._analyze_data_structure(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            self.logger.error(f"Error executing tool {name}: {e}", exc_info=True)
            raise

    def _fig_to_svg_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 SVG."""
        buffer = BytesIO()
        fig.savefig(buffer, format='svg', bbox_inches='tight')
        buffer.seek(0)
        svg_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        return f"<chart>data:image/svg+xml;base64,{svg_b64}</chart>"

    def _create_bar_chart(self, args: Dict[str, Any]) -> str:
        df = pd.read_csv(StringIO(args["data"]))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df[args["x_column"]], df[args["y_column"]])
        ax.set_title(args.get("title", "Bar Chart"))
        ax.set_xlabel(args.get("x_label") or args["x_column"])
        ax.set_ylabel(args.get("y_label") or args["y_column"])
        plt.xticks(rotation=45)
        plt.tight_layout()
        return self._fig_to_svg_base64(fig)

    def _create_line_chart(self, args: Dict[str, Any]) -> str:
        df = pd.read_csv(StringIO(args["data"]))
        fig, ax = plt.subplots(figsize=(10, 6))
        group_col = args.get("group_column")
        if group_col and group_col in df.columns:
            for name, group in df.groupby(group_col):
                ax.plot(group[args["x_column"]], group[args["y_column"]], label=name, marker='o')
            ax.legend()
        else:
            ax.plot(df[args["x_column"]], df[args["y_column"]], marker='o')
        ax.set_title(args.get("title", "Line Chart"))
        ax.set_xlabel(args.get("x_label") or args["x_column"])
        ax.set_ylabel(args.get("y_label") or args["y_column"])
        plt.xticks(rotation=45)
        plt.tight_layout()
        return self._fig_to_svg_base64(fig)

    def _create_pie_chart(self, args: Dict[str, Any]) -> str:
        df = pd.read_csv(StringIO(args["data"]))
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(
            df[args["values_column"]], 
            labels=df[args["names_column"]], 
            autopct='%1.1f%%', 
            startangle=90
        )
        ax.set_title(args.get("title", "Pie Chart"))
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        plt.tight_layout()
        return self._fig_to_svg_base64(fig)

    def _create_scatter_plot(self, args: Dict[str, Any]) -> str:
        df = pd.read_csv(StringIO(args["data"]))
        fig, ax = plt.subplots(figsize=(10, 6))
        color_col = args.get("color_column")
        if color_col and color_col in df.columns:
            for cat in df[color_col].unique():
                mask = df[color_col] == cat
                ax.scatter(df[args["x_column"]][mask], df[args["y_column"]][mask], label=cat, alpha=0.7)
            ax.legend()
        else:
            ax.scatter(df[args["x_column"]], df[args["y_column"]], alpha=0.7)
        ax.set_title(args.get("title", "Scatter Plot"))
        ax.set_xlabel(args.get("x_label") or args["x_column"])
        ax.set_ylabel(args.get("y_label") or args["y_column"])
        plt.tight_layout()
        return self._fig_to_svg_base64(fig)

    def _analyze_data_structure(self, args: Dict[str, Any]) -> str:
        df = pd.read_csv(StringIO(args["data"]))
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

    def setup_handlers(self):
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"]) 
                    for t in self.get_tool_schema()]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[TextContent]:
            try:
                result = await self.execute_tool(name, arguments)
                return [TextContent(type="text", text=result)]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]


def main():
    parser = argparse.ArgumentParser(description="DataViz MCP Server")
    parser.parse_args()
    
    from common.cli import mcp_stdio_cli_runner
    server = DataVizMCPServer(name="dataviz-server", log_file="dataviz-server.log")
    asyncio.run(mcp_stdio_cli_runner(server))


async def _async_lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Lambda handler for AgentCore Gateway invocations."""
    logger = logging.getLogger("dataviz-server")
    logger.setLevel(logging.INFO)
    
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        server = DataVizMCPServer(name="dataviz-server")

        # Extract tool name from AgentCore context
        tool_name_with_prefix = None
        if hasattr(context, 'client_context') and context.client_context:
            custom = getattr(context.client_context, 'custom', {})
            tool_name_with_prefix = custom.get('bedrockAgentCoreToolName')

        if tool_name_with_prefix:
            function_name = context.function_name
            prefix = f"{function_name}___"
            requested_tool = tool_name_with_prefix[len(prefix):] if tool_name_with_prefix.startswith(prefix) else tool_name_with_prefix
            result = await server.execute_tool(requested_tool, event)
            return {'statusCode': 200, 'body': result}
        else:
            # Fallback for direct invocations
            mcp_request = event.get('mcp_request', event)
            method = mcp_request.get('method', '')
            params = mcp_request.get('params', {})

            if method == 'tools/list':
                return {'statusCode': 200, 'body': json.dumps({'tools': server.get_tool_schema()})}
            
            requested_tool = params.get("name") or event.get("tool_name")
            tool_arguments = params.get("arguments", {}) or event.get("arguments", {})
            
            if requested_tool:
                result = await server.execute_tool(requested_tool, tool_arguments)
                return {'statusCode': 200, 'body': result}
            
            return {'statusCode': 400, 'body': json.dumps({'error': 'No tool specified'})}

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Synchronous Lambda entry point."""
    print(f"Invoked event: {json.dumps(event, default=str)}")
    return asyncio.run(_async_lambda_handler(event, context))


if __name__ == "__main__":
    main()
