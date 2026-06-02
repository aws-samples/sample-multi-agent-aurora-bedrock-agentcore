#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
ElectrifyMCPServer
A Model Context Protocol server that connects to Electrify's PostgreSQL database and exposes canned database queries as tools.
"""

import asyncio
import json
import logging
import os
import argparse
from typing import Any, Dict, List, Optional, Sequence

import psycopg
from psycopg.rows import dict_row
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("electrify-mcp-server")

# Define argument parser
parser = argparse.ArgumentParser(description="Electrify MCP Server - PostgreSQL database query server")
db_group = parser.add_argument_group('Database Connection Parameters')
db_group.add_argument('-e', '--endpoint', help="PostgreSQL host", default=os.getenv('PGHOST', 'localhost'))
db_group.add_argument('-p', '--port', help="PostgreSQL port", default=os.getenv('PGPORT', '5432'))
db_group.add_argument('-d', '--database', help="PostgreSQL database name", default=os.getenv('PGDATABASE', 'postgres'))
db_group.add_argument('-u', '--user', help="PostgreSQL username", default=os.getenv('PGUSER', 'postgres'))
db_group.add_argument('--password', help="PostgreSQL password", default=os.getenv('PGPASSWORD', ''))

# Electrify database MCP server
class ElectrifyMCPServer:
    def __init__(self, args):
        self.server = Server("electrify-mcp-server")
        self.args = args
        self.connection_string = self._get_connection_string()
        self.setup_handlers()
    
    def _get_connection_string(self) -> str:
        # Security notes:
        #   - In production, source credentials from AWS Secrets Manager
        #     instead of the --password CLI arg (used here for local dev only).
        #   - sslmode=require enforces TLS on the wire to Aurora; pair with
        #     rds.force_ssl=1 on the cluster parameter group to require it
        #     server-side too.
        #   - Aurora encryption-at-rest must be set at cluster creation
        #     (cannot be enabled later); confirm StorageEncrypted=true.
        #   - This DSN includes the password as a query string; never log it.
        host = self.args.endpoint
        port = self.args.port
        database = self.args.database
        user = self.args.user
        password = self.args.password
        
        return f"host={host} port={port} dbname={database} user={user} password={password} sslmode=require"
    
    def setup_handlers(self):
        """Set up MCP server handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available database query tools."""
            return [
                Tool(
                    name="get_rates",
                    description="Get all available rate plans and pricing tiers for time of day.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Limit number of results",
                                "default": 100
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="get_customer",
                    description="Retrieve customer profile information for the indicated customer, including the currently active electricity rate plan information and devices.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "customer_username": {
                                "type": "string",
                                "description": "The customer username to retrieve information for"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_bills",
                    description="Retrieve the latest bills for the indicated customer",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "customer_username": {
                                "type": "string",
                                "description": "The customer username to retrieve bills for"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Limit number of results",
                                "default": 100
                            }
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[TextContent]:
            """Handle tool calls."""
            logger.info(f"Tool called: {name} with arguments: {arguments}")
            try:
                if name == "get_rates":
                    return await self._get_rates(arguments)
                elif name == "get_customer":
                    return await self._get_customer(arguments)
                elif name == "get_bills":
                    return await self._get_bills(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}", exc_info=True)
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _execute_query(self, query: str, parameters: List[Any] = None) -> List[Dict[str, Any]]:
        """Execute a query against the PostgreSQL database."""
        if parameters is None:
            parameters = []
        
        logger.info(f"Executing query: {query}")
        logger.info(f"With parameters: {parameters}")
        
        try:
            async with await psycopg.AsyncConnection.connect(
                self.connection_string,
                row_factory=dict_row
            ) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, parameters)
                    if cur.description:  # Query returns results
                        results = await cur.fetchall()
                        logger.info(f"Query returned {len(results)} rows")
                        return [dict(row) for row in results]
                    else:  # Query doesn't return results (INSERT, UPDATE, DELETE)
                        return [{"affected_rows": cur.rowcount}]
        except Exception as e:
            logger.error(f"Database error: {e}", exc_info=True)
            raise
    
    async def _get_rates(self, arguments: Dict[str, Any]) -> Sequence[TextContent]:
        base_query = "SELECT rate_id, rate_program, start_hour, end_hour, price_kwh FROM rates LIMIT %s::INTEGER;"
        parameters = []
        param_count = 0
        
        parameters.append(arguments.get("limit", 100))
        param_count += 1
        
        results = await self._execute_query(base_query, parameters)
        
        return [TextContent(
            type="text",
            text=json.dumps(results, indent=2, default=str)
        )]
    
    async def _get_customer(self, arguments: Dict[str, Any]) -> Sequence[TextContent]:
        base_query = "SELECT c.*, cre.device_id::text AS device_id, cre.rate_program, cre.start_date, cre.end_date FROM customers c INNER JOIN customer_rate_enrollment cre ON c.customer_id = cre.customer_id AND cre.status = 'active' WHERE customer_username = %s::text;"
        parameters = []
        param_count = 0
        
        parameters.append(arguments.get("customer_username", 0))
        param_count += 1
        
        results = await self._execute_query(base_query, parameters)
        
        return [TextContent(
            type="text",
            text=json.dumps(results, indent=2, default=str)
        )]
    
    async def _get_bills(self, arguments: Dict[str, Any]) -> Sequence[TextContent]:
        base_query = "SELECT DISTINCT TO_CHAR(i.bill_date, 'YYYY-MM-DD') AS sorter, i.invoice_no, i.customer_id, TO_CHAR(i.bill_date, 'MM/DD/YYYY') AS bill_date, TO_CHAR(i.due_date, 'MM/DD/YYYY') AS due_date, i.invoice_amount FROM invoices i INNER JOIN customers c ON c.customer_id = i.customer_id WHERE c.customer_username = %s::text ORDER BY sorter DESC LIMIT %s::INTEGER;"
        parameters = []
        param_count = 0
        
        parameters.append(arguments.get("customer_username", 0))
        param_count += 1

        parameters.append(arguments.get("limit", 100))
        param_count += 1
        
        results = await self._execute_query(base_query, parameters)
        
        return [TextContent(
            type="text",
            text=json.dumps(results, indent=2, default=str)
        )]

async def main():
    """Main entry point for the MCP server."""
    args = parser.parse_args()
    
    logger.info("Starting Electrify MCP Server...")
    logger.info(f"Arguments: host={args.endpoint}, port={args.port}, database={args.database}, user={args.user}")
    
    server_instance = ElectrifyMCPServer(args)
    
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            server_instance.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
