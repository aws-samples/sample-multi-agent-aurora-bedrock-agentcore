# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import boto3
import asyncio
import json
import logging
import argparse
from typing import Any, Dict, List, Optional, Sequence
from mcp.server import Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

from common.types import MPCServerConfig, DataApiDatabaseConfig, NativeDatabaseConfig


# Electrify database MCP server
class ElectrifyMCPServer:
    def __init__(self, config: MPCServerConfig):
        # Init MCP server, save config
        self.server = Server(config.name)
        self.config = config
        self.logger = logging.getLogger(config.name)
        self.logger.setLevel(logging.INFO)
        if __name__ == "__main__":
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)

        self.setup_handlers()
    

    def get_tool_schema(self):
        return [
            {
                "name": "get_rates",
                "description": "Get all available rate plans and pricing tiers for time of day.",
                "inputSchema": {
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
            },
            {
                "name": "get_customer",
                "description": "Retrieve customer profile information for the indicated customer, including the currently active electricity rate plan information and devices.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "customer_username": {
                            "type": "string",
                            "description": "The customer username to retrieve information for"
                        }
                    },
                    "required": ["customer_username"]
                }
            },
            {
                "name": "get_bills",
                "description": "Retrieve the latest bills for the indicated customer",
                "inputSchema": {
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
                    },
                    "required": ["customer_username"]
                }
            }
        ]


    async def execute_tool(self, name: str, arguments: Dict[str, Any]):
        self.logger.info(f"Tool called: {name} with arguments: {arguments}")
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
            self.logger.error(f"Error executing tool {name}: {e}", exc_info=True)
            raise ValueError(f"Error executing tool {name}: {e}")

    
    def setup_handlers(self):
        """Set up MCP server handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available database query tools."""
            return [Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"]
            ) for t in self.get_tool_schema()]

        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[TextContent]:
            """Handle tool calls."""
            self.logger.info(f"Tool called: {name} with arguments: {arguments}")
            try:
                results = await self.execute_tool(name, arguments)
                
                return [TextContent(
                    type="text",
                    text=json.dumps(results, indent=2, default=str)
                )]

            except Exception as e:
                self.logger.error(f"Error executing tool {name}: {e}", exc_info=True)
                return [TextContent(
                    type="text", 
                    text=f"Error: {str(e)}"
                )]
    

    def _convert_placeholders(self, query: str) -> str:
        """
        Convert %s placeholders to :paramX format (simpler version).
        
        Args:
            query: SQL query string with %s placeholders
            
        Returns:
            SQL query string with :param0, :param1, etc. placeholders
        """
        param_count = 0
        
        def replace_placeholder(match):
            nonlocal param_count
            replacement = f':param{param_count}'
            param_count += 1
            return replacement
        
        import re
        return re.sub(r'%s', replace_placeholder, query)
    
    
    async def _execute_query(self, query: str, parameters: List[Any] = None) -> List[Dict[str, Any]]:
        """Execute a query against the PostgreSQL database."""
        if parameters is None:
            parameters = []
        
        self.logger.info(f"Executing query: {query}")
        self.logger.info(f"With parameters: {parameters}")
        
        try:
            if isinstance(self.config.db, NativeDatabaseConfig):
                async with await psycopg.AsyncConnection.connect(
                    self.config.db.to_connstring(),
                    row_factory=dict_row
                ) as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(query, parameters)
                        if cur.description:  # Query returns results
                            results = await cur.fetchall()
                            self.logger.info(f"Query returned {len(results)} rows")
                            return [dict(row) for row in results]
                        else:  # Query doesn't return results (INSERT, UPDATE, DELETE)
                            return [{"affected_rows": cur.rowcount}]
            elif isinstance(self.config.db, DataApiDatabaseConfig):
                params = []
                i = 0
                for p in parameters:
                    v = {}
                    if isinstance(p, int):
                        v = { "longValue": p }
                    elif isinstance(p, float):
                        v = { "doubleValue": p }
                    elif isinstance(p, bool):
                        v = { "booleanValue": p }
                    else:
                        v = { "stringValue": str(p) }

                    params.append({ "name": f"param{i}", "value": v })
                    i += 1

                api = boto3.client('rds-data', region_name=self.config.db.region)
                response = api.execute_statement(
                    resourceArn=self.config.db.cluster_arn,
                    secretArn=self.config.db.secret_arn,
                    database=self.config.db.database,
                    sql=self._convert_placeholders(query),
                    parameters=params,
                    formatRecordsAs='JSON')

                if "formattedRecords" in response and isinstance(response["formattedRecords"], str) and len(response["formattedRecords"]):
                    results = json.loads(response["formattedRecords"])
                    self.logger.info(f"Query returned {len(results)} rows")
                    return [dict(row) for row in results]

                else:  # Query doesn't return results (INSERT, UPDATE, DELETE)
                    return [{"affected_rows": response["numberOfRecordsUpdated"] if "numberOfRecordsUpdated" in response else 0}]

            else:
                raise ValueError(f"Unknown database connection menthod '{self.config.db.method}'.")
        except Exception as e:
            self.logger.error(f"Database error: {e}", exc_info=True)
            raise
    
    async def _get_rates(self, arguments: Dict[str, Any]) -> Sequence[TextContent]:
        base_query = "SELECT rate_id, rate_program, start_hour, end_hour, price_kwh FROM rates LIMIT %s::INTEGER;"
        parameters = []
        param_count = 0
        
        parameters.append(arguments.get("limit", 100))
        param_count += 1
        
        return await self._execute_query(base_query, parameters)
        
   
    async def _get_customer(self, arguments: Dict[str, Any]) -> Sequence[TextContent]:
        base_query = "SELECT c.*, cre.device_id::text AS device_id, cre.rate_program, cre.start_date, cre.end_date FROM customers c INNER JOIN customer_rate_enrollment cre ON c.customer_id = cre.customer_id AND cre.status = 'active' WHERE customer_username = %s::text;"
        
        customer_username = arguments.get("customer_username")
        if not customer_username:
            raise ValueError("customer_username is required")
        
        self.logger.info(f"customer_username received: {customer_username}")
        return await self._execute_query(base_query, [customer_username])

    
    async def _get_bills(self, arguments: Dict[str, Any]) -> Sequence[TextContent]:
        base_query = "SELECT DISTINCT TO_CHAR(i.bill_date, 'YYYY-MM-DD') AS sorter, i.invoice_no, i.customer_id, TO_CHAR(i.bill_date, 'MM/DD/YYYY') AS bill_date, TO_CHAR(i.due_date, 'MM/DD/YYYY') AS due_date, i.invoice_amount FROM invoices i INNER JOIN customers c ON c.customer_id = i.customer_id WHERE c.customer_username = %s::text ORDER BY sorter DESC LIMIT %s::INTEGER;"
        
        customer_username = arguments.get("customer_username")
        if not customer_username:
            raise ValueError("customer_username is required")
        
        self.logger.info(f"customer_username received: {customer_username}")
        return await self._execute_query(base_query, [customer_username, arguments.get("limit", 100)])


def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="Electrify MCP Server - PostgreSQL database query server")
    parser.add_argument('-c', '--cluster-arn', help="Aurora PostgreSQL DB cluster ARN")
    parser.add_argument('-s', '--secret-arn', help="AWS Secrets Manager secret ARN")
    parser.add_argument('-r', '--region', help="AWS Region to use", default=os.getenv("AWS_REGION", "us-east-1"))
    parser.add_argument('-d', '--database', help="The name of the database to connect to", default="postgres")
    args = parser.parse_args()
    
    # Set config
    config = MPCServerConfig(
        name="electrify-server",
        log_file="electrify-server.log",
        db=DataApiDatabaseConfig(
            cluster_arn=args.cluster_arn,
            secret_arn=args.secret_arn,
            region=args.region,
            database=args.database
        )
    )
    
    from common.cli import mcp_stdio_cli_runner

    # Run CLI async
    asyncio.run(mcp_stdio_cli_runner(ElectrifyMCPServer(config)))


async def _async_lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Async implementation of Lambda handler for MCP server requests.
    
    When invoked via AgentCore Gateway:
    - event contains the tool arguments directly
    - context.client_context.custom['bedrockAgentCoreToolName'] contains the tool name
      prefixed with function_name + '___'
    - Identity headers are passed via context.client_context.custom or event headers
    
    Args:
        event: Lambda event containing tool arguments
        context: Lambda context object with AgentCore metadata
        
    Returns:
        Dict with statusCode and body containing MCP response
    """
    logger = logging.getLogger("electrify-server")
    logger.setLevel(logging.INFO)
    
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Extract identity from various sources
        user_id = None
        username = None
        user_email = None
        
        # Try to get identity from context.client_context.custom (AgentCore Gateway)
        if hasattr(context, 'client_context') and context.client_context:
            custom = getattr(context.client_context, 'custom', {}) or {}
            user_id = custom.get('X-User-Id') or custom.get('x-user-id')
            username = custom.get('X-Username') or custom.get('x-username')
            user_email = custom.get('X-User-Email') or custom.get('x-user-email')
            logger.info(f"Identity from context.client_context.custom: user_id={user_id}, username={username}, email={user_email}")
        
        # Try to get identity from event headers (API Gateway style)
        if not user_id and 'headers' in event:
            headers = event.get('headers', {})
            user_id = headers.get('X-User-Id') or headers.get('x-user-id')
            username = headers.get('X-Username') or headers.get('x-username')
            user_email = headers.get('X-User-Email') or headers.get('x-user-email')
            # Log Authorization header presence (for debugging JWT flow)
            auth_header = headers.get('Authorization') or headers.get('authorization')
            if auth_header:
                auth_preview = auth_header[:30] + "..." if len(auth_header) > 30 else auth_header
                logger.info(f"Authorization header present: {auth_preview}")
            else:
                logger.warning("No Authorization header in event.headers")
            logger.info(f"All headers received: {list(headers.keys())}")
            logger.info(f"Identity from event.headers: user_id={user_id}, username={username}, email={user_email}")
        
        # Try to get identity from requestContext (AgentCore may pass it here)
        if not user_id and 'requestContext' in event:
            req_ctx = event.get('requestContext', {})
            authorizer = req_ctx.get('authorizer', {})
            user_id = authorizer.get('userId') or authorizer.get('sub')
            username = authorizer.get('username')
            user_email = authorizer.get('email')
            logger.info(f"Identity from event.requestContext: user_id={user_id}, username={username}, email={user_email}")
        
        logger.info(f"Final identity: user_id={user_id}, username={username}, email={user_email}")

        # Set config
        config = MPCServerConfig(
            name="electrify-server",
            log_file="electrify-server.log",
            db=DataApiDatabaseConfig(
                cluster_arn=os.getenv("DB_CLUSTER_ARN"),
                secret_arn=os.getenv("SECRET_ARN"),
                region=os.getenv("REGION", "us-east-1"),
                database=os.getenv("DATABASE", "postgres")
            )
        )

        # electrify server
        electrify = ElectrifyMCPServer(config)
        electrify.setup_handlers()

        # Extract tool name from AgentCore context
        # Format: context.client_context.custom['bedrockAgentCoreToolName'] = 'function-name___tool_name'
        tool_name_with_prefix = None
        if hasattr(context, 'client_context') and context.client_context:
            custom = getattr(context.client_context, 'custom', {})
            tool_name_with_prefix = custom.get('bedrockAgentCoreToolName')
        
        if tool_name_with_prefix:
            # Strip function name prefix (function_name + '___')
            function_name = context.function_name
            prefix = f"{function_name}___"
            if tool_name_with_prefix.startswith(prefix):
                requested_tool = tool_name_with_prefix[len(prefix):]
            else:
                requested_tool = tool_name_with_prefix
            
            # Tool arguments are in the event directly
            tool_arguments = event

            logger.info(f"AgentCore invocation - Tool: {requested_tool}, Arguments: {tool_arguments}")

            # Authorization: if a customer_username is requested and we know
            # who the authenticated caller is, refuse cross-user access.
            requested_user = tool_arguments.get("customer_username") if isinstance(tool_arguments, dict) else None
            authenticated = (user_email or username or user_id)
            if requested_user and authenticated and requested_user != authenticated:
                logger.warning(
                    f"Refusing cross-user access: caller={authenticated} requested={requested_user}"
                )
                return {
                    'statusCode': 403,
                    'body': json.dumps({
                        'error': 'forbidden',
                        'reason': 'authenticated principal does not match requested customer_username',
                    })
                }

            # Execute the tool
            result = await electrify.execute_tool(requested_tool, tool_arguments)
            
            return {
                'statusCode': 200,
                'body': json.dumps(result, default=str)
            }
        else:
            # Fallback for direct invocations (non-AgentCore)
            mcp_request = event.get('mcp_request', event)
            method = mcp_request.get('method', '')
            params = mcp_request.get('params', {})

            # Get available tools
            tools = electrify.get_tool_schema()

            # Handle different MCP methods
            if method == 'tools/list':
                result = { 'tools': tools }
            else:
                requested_tool = params.get("name", False) or event.get("tool_name", False)
                tool_arguments = params.get("arguments", {}) or event.get("arguments", {})

                if requested_tool:
                    result = await electrify.execute_tool(requested_tool, tool_arguments)
                else:
                    return {
                        'statusCode': 400,
                        'body': json.dumps({'error': 'No tool specified'})
                    }

            return {
                'statusCode': 200,
                'body': json.dumps(result, default=str)
            }

    except Exception as e:
        logger.error(f"Error in MCP Lambda handler: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Synchronous Lambda handler entry point that wraps the async implementation.
    
    Args:
        event: Lambda event containing MCP request
        context: Lambda context object
        
    Returns:
        Dict with statusCode and body containing MCP response
    """
    print("Invoked event:")
    print(json.dumps(event, default=str))
    print("Invoked context:")
    print(f"function_name: {context.function_name}")
    print(f"function_version: {context.function_version}")
    print(f"invoked_function_arn: {context.invoked_function_arn}")
    print(f"memory_limit_in_mb: {context.memory_limit_in_mb}")
    print(f"aws_request_id: {context.aws_request_id}")
    
    # Log client_context details for AgentCore Gateway debugging
    if hasattr(context, 'client_context') and context.client_context:
        print(f"client_context.custom: {context.client_context.custom}")
        print(f"client_context.env: {context.client_context.env}")
        print(f"client_context.client: {context.client_context.client}")
    else:
        print("client_context: None")
    
    return asyncio.run(_async_lambda_handler(event, context))


if __name__ == "__main__":
    main()