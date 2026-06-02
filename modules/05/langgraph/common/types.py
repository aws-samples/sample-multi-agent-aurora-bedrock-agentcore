# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
import os
from typing import Optional
from pydantic import BaseModel


class IdentityContext(BaseModel):
    """User identity information from authentication."""
    username: str
    sub: str  # Cognito user ID (subject)
    email: Optional[str] = None
    groups: list[str] = []
    jwt_token: Optional[str] = None  # For propagation to downstream services


class StdioServerConfig(BaseModel):
    name: str = "stdio_mcp_server"
    script: str = "uvx"
    args: list[str] = []


class HttpsServerConfig(BaseModel):
    """Configuration for HTTPS-based MCP servers (e.g., AgentCore Gateway)."""
    name: str = "https_mcp_server"
    url: str  # Full URL to the MCP server endpoint
    headers: dict[str, str] = {}  # Optional headers (e.g., authorization)
    timeout: int = 30  # Request timeout in seconds
    propagate_identity: bool = True  # Whether to propagate identity to this server


class AgentConfig(BaseModel):
    name: str = "react_agent"
    description: str = "This is a minimal ReAct agent implemented using LangGraph."
    identity: str = os.getenv('USER', 'unknown')
    thread: str = os.getenv('USER', 'unknown') + '-01'
    system_prompt: str = "# System prompt"
    log_file: str | None = None
    model: str = "global.anthropic.claude-sonnet-4-6"
    memory: str = None
    region: str = None
    stdio_servers: list[StdioServerConfig] = []
    https_servers: list[HttpsServerConfig] = []
    identity_context: Optional[IdentityContext] = None  # User identity for propagation


class NativeDatabaseConfig(BaseModel):
    endpoint: str = "localhost"
    port: str = "5432"
    database: str = "postgres"
    user: str
    password: str

    def to_connstring(self):
        return f"host={self.endpoint} port={self.port} dbname={self.database} user={self.user} password={self.password}"


class DataApiDatabaseConfig(BaseModel):
    cluster_arn: str
    secret_arn: str
    region: str
    database: str = "postgres"


class MPCServerConfig(BaseModel):
    name: str = "mpc-server"
    log_file: str = "server.log"
    db: NativeDatabaseConfig | DataApiDatabaseConfig