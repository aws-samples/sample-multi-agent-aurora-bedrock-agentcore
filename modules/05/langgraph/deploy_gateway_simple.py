#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
Deploy MCP Servers to Amazon Bedrock AgentCore Gateway

Supports deploying multiple MCP servers (Lambda functions) as targets on a single gateway.
"""

import boto3
import json
import os
import sys
import time
from typing import Dict, List, Any
from dataclasses import dataclass, field
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tool schemas for known MCP servers
TOOL_SCHEMAS = {
    "electrify": {
        "target_name": "electrify-server-function",
        "description": "Electrify Lambda MCP server - electricity billing and rates",
        "tools": [
            {"name": "get_rates", "description": "Get all available rate plans and pricing tiers for time of day",
             "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "description": "Limit number of results"}}, "required": []}},
            {"name": "get_customer", "description": "Retrieve customer profile information including rate plan and devices. If customer_username is not provided, uses the authenticated user's identity.",
             "inputSchema": {"type": "object", "properties": {"customer_username": {"type": "string", "description": "The customer username to retrieve information for. Optional if user is authenticated."}}, "required": []}},
            {"name": "get_bills", "description": "Retrieve the latest bills for the indicated customer. If customer_username is not provided, uses the authenticated user's identity.",
             "inputSchema": {"type": "object", "properties": {"customer_username": {"type": "string", "description": "The customer username to retrieve bills for. Optional if user is authenticated."}, "limit": {"type": "integer", "description": "Limit number of results"}}, "required": []}}
        ]
    },
    "dataviz": {
        "target_name": "dataviz-server-function",
        "description": "DataViz Lambda MCP server - chart and visualization creation",
        "tools": [
            {"name": "create_bar_chart", "description": "Create a bar chart from CSV data",
             "inputSchema": {"type": "object", "properties": {"data": {"type": "string", "description": "CSV formatted data"}, "x_column": {"type": "string", "description": "Column name for x-axis"}, "y_column": {"type": "string", "description": "Column name for y-axis"}, "title": {"type": "string", "description": "Chart title"}, "x_label": {"type": "string", "description": "X-axis label"}, "y_label": {"type": "string", "description": "Y-axis label"}}, "required": ["data", "x_column", "y_column"]}},
            {"name": "create_line_chart", "description": "Create a line chart from CSV data",
             "inputSchema": {"type": "object", "properties": {"data": {"type": "string", "description": "CSV formatted data"}, "x_column": {"type": "string", "description": "Column name for x-axis"}, "y_column": {"type": "string", "description": "Column name for y-axis"}, "title": {"type": "string", "description": "Chart title"}, "x_label": {"type": "string", "description": "X-axis label"}, "y_label": {"type": "string", "description": "Y-axis label"}}, "required": ["data", "x_column", "y_column"]}},
            {"name": "create_pie_chart", "description": "Create a pie chart from CSV data",
             "inputSchema": {"type": "object", "properties": {"data": {"type": "string", "description": "CSV formatted data"}, "values_column": {"type": "string", "description": "Column name for values"}, "names_column": {"type": "string", "description": "Column name for slice names"}, "title": {"type": "string", "description": "Chart title"}}, "required": ["data", "values_column", "names_column"]}},
            {"name": "create_scatter_plot", "description": "Create a scatter plot from CSV data",
             "inputSchema": {"type": "object", "properties": {"data": {"type": "string", "description": "CSV formatted data"}, "x_column": {"type": "string", "description": "Column name for x-axis"}, "y_column": {"type": "string", "description": "Column name for y-axis"}, "title": {"type": "string", "description": "Chart title"}, "x_label": {"type": "string", "description": "X-axis label"}, "y_label": {"type": "string", "description": "Y-axis label"}}, "required": ["data", "x_column", "y_column"]}},
            {"name": "analyze_data_structure", "description": "Analyze CSV data structure and suggest visualization options",
             "inputSchema": {"type": "object", "properties": {"data": {"type": "string", "description": "CSV formatted data"}}, "required": ["data"]}}
        ]
    }
}


@dataclass
class MCPTarget:
    """Configuration for a single MCP target."""
    lambda_arn: str
    server_type: str  # "electrify" or "dataviz"


@dataclass
class MCPDeploymentConfig:
    """Configuration for MCP gateway deployment."""
    region: str = "us-east-1"
    gateway_name: str = "mcp-gateway"
    gateway_description: str = "Gateway for MCP Servers"
    gateway_role_name: str = None
    targets: List[MCPTarget] = field(default_factory=list)
    cognito_user_pool_id: str = None
    cognito_client_id: str = None
    
    def __post_init__(self):
        if not self.gateway_role_name:
            self.gateway_role_name = f"{self.gateway_name}-role"


class MCPServerDeployer:
    """Deploys MCP servers to Amazon Bedrock AgentCore Gateway."""
    
    def __init__(self, config: MCPDeploymentConfig):
        self.config = config
        self.session = boto3.Session(region_name=config.region)
        self.agentcore_client = self.session.client('bedrock-agentcore-control')
        self.iam_client = self.session.client('iam')
        self.created_resources = {'iam_roles': [], 'gateway_id': None}
    
    def deploy(self) -> Dict[str, Any]:
        """Deploy gateway with all configured targets."""
        try:
            logger.info("Starting MCP gateway deployment...")
            
            # Build list of all Lambda ARNs for IAM policy
            lambda_arns = [t.lambda_arn for t in self.config.targets]
            gateway_role_arn = self._create_gateway_role(lambda_arns)
            
            # Check if gateway exists
            gateway_id = self._get_existing_gateway()
            if gateway_id:
                logger.info(f"Gateway '{self.config.gateway_name}' exists: {gateway_id}")
                # Update role policy to include all Lambda ARNs
                self._update_role_policy(lambda_arns)
            else:
                gateway_id = self._create_gateway(gateway_role_arn)
            
            # Add each target
            for target in self.config.targets:
                self._add_target(gateway_id, target)
            
            # Get gateway URL
            gateway_info = self.agentcore_client.get_gateway(gatewayIdentifier=gateway_id)
            
            logger.info("MCP gateway deployment completed!")
            return {
                'gateway_id': gateway_id,
                'gateway_url': gateway_info.get('gatewayUrl'),
                'targets': [t.server_type for t in self.config.targets],
                'region': self.config.region
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            raise
    
    def _get_existing_gateway(self) -> str:
        """Check if gateway already exists, return gateway_id or None."""
        try:
            response = self.agentcore_client.list_gateways()
            for gw in response.get('items', []):
                if gw.get('name') == self.config.gateway_name:
                    return gw.get('gatewayId')
        except Exception:
            pass
        return None
    
    def _create_gateway_role(self, lambda_arns: List[str]) -> str:
        """Create or update IAM role for gateway."""
        role_name = self.config.gateway_role_name
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [{"Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}, "Action": "sts:AssumeRole"}]
        }
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Action": ["lambda:InvokeFunction"], "Resource": lambda_arns},
                {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], "Resource": self._logs_resource_arns()}
            ]
        }

        try:
            response = self.iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(trust_policy), Description=f"Role for {role_name}")
            role_arn = response['Role']['Arn']
            self.created_resources['iam_roles'].append(role_name)
            logger.info(f"Created IAM role: {role_arn}")
            time.sleep(10)
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            logger.info(f"IAM role {role_name} exists, updating policy")
            response = self.iam_client.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
        
        self.iam_client.put_role_policy(RoleName=role_name, PolicyName=f"{role_name}Policy", PolicyDocument=json.dumps(policy_document))
        return role_arn
    
    def _logs_resource_arns(self) -> List[str]:
        """Scope CloudWatch Logs IAM permissions for the AgentCore Gateway role.
        Gateway-emitted logs land under /aws/bedrock-agentcore/*; scope to that
        prefix instead of '*' to follow least-privilege."""
        account_id = boto3.client('sts').get_caller_identity()['Account']
        log_group_arn = f"arn:aws:logs:{self.config.region}:{account_id}:log-group:/aws/bedrock-agentcore/*"
        return [log_group_arn, log_group_arn + ":*"]

    def _update_role_policy(self, lambda_arns: List[str]):
        """Update existing role policy with new Lambda ARNs."""
        role_name = self.config.gateway_role_name
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Action": ["lambda:InvokeFunction"], "Resource": lambda_arns},
                {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], "Resource": self._logs_resource_arns()}
            ]
        }
        self.iam_client.put_role_policy(RoleName=role_name, PolicyName=f"{role_name}Policy", PolicyDocument=json.dumps(policy_document))
        logger.info(f"Updated IAM role policy for {len(lambda_arns)} Lambda functions")
    
    def _create_gateway(self, gateway_role_arn: str) -> str:
        """Create new gateway."""
        gateway_params = {
            "name": self.config.gateway_name,
            "roleArn": gateway_role_arn,
            "description": self.config.gateway_description,
            "protocolType": "MCP",
            "protocolConfiguration": {"mcp": {"searchType": "SEMANTIC", "supportedVersions": ["2025-03-26"]}}
        }
        
        if self.config.cognito_user_pool_id and self.config.cognito_client_id:
            discovery_url = f"https://cognito-idp.{self.config.region}.amazonaws.com/{self.config.cognito_user_pool_id}/.well-known/openid-configuration"
            gateway_params["authorizerType"] = "CUSTOM_JWT"
            gateway_params["authorizerConfiguration"] = {"customJWTAuthorizer": {"discoveryUrl": discovery_url, "allowedClients": [self.config.cognito_client_id]}}
            logger.info(f"Configuring JWT auth with Cognito: {self.config.cognito_user_pool_id}")
        else:
            gateway_params["authorizerType"] = "NONE"
        
        response = self.agentcore_client.create_gateway(**gateway_params)
        gateway_id = response['gatewayId']
        self.created_resources['gateway_id'] = gateway_id
        logger.info(f"Created gateway: {gateway_id}")
        
        self._wait_for_gateway_ready(gateway_id)
        return gateway_id
    
    def _wait_for_gateway_ready(self, gateway_id: str, max_wait: int = 300, interval: int = 10):
        """Wait for gateway to be ready."""
        logger.info(f"Waiting for gateway {gateway_id} to be ready...")
        start = time.time()
        while time.time() - start < max_wait:
            response = self.agentcore_client.get_gateway(gatewayIdentifier=gateway_id)
            status = response.get('status', 'UNKNOWN')
            logger.info(f"Gateway status: {status}")
            if status == 'READY':
                return
            if status in ['FAILED', 'DELETING', 'DELETED']:
                raise Exception(f"Gateway failed: {status}")
            time.sleep(interval)
        raise TimeoutError(f"Gateway not ready within {max_wait}s")
    
    def _add_target(self, gateway_id: str, target: MCPTarget):
        """Add a target to the gateway."""
        schema = TOOL_SCHEMAS.get(target.server_type)
        if not schema:
            raise ValueError(f"Unknown server type: {target.server_type}. Use: {list(TOOL_SCHEMAS.keys())}")
        
        target_name = schema["target_name"]
        
        # Check if target already exists
        try:
            existing = self.agentcore_client.list_gateway_targets(gatewayIdentifier=gateway_id)
            for t in existing.get('items', []):
                if t.get('name') == target_name:
                    logger.info(f"Target '{target_name}' already exists, skipping")
                    return
        except Exception:
            pass
        
        mcp_target_config = {
            "mcp": {"lambda": {"lambdaArn": target.lambda_arn, "toolSchema": {"inlinePayload": schema["tools"]}}}
        }
        
        self.agentcore_client.create_gateway_target(
            gatewayIdentifier=gateway_id,
            name=target_name,
            description=schema["description"],
            targetConfiguration=mcp_target_config,
            credentialProviderConfigurations=[{"credentialProviderType": "GATEWAY_IAM_ROLE"}]
        )
        logger.info(f"Added target '{target_name}' to gateway")


def main():
    parser = argparse.ArgumentParser(description="Deploy MCP Servers to AgentCore Gateway")
    parser.add_argument('--region', default=os.getenv("AWS_REGION", "us-east-1"))
    parser.add_argument('--gateway-name', default='mcp-gateway')
    parser.add_argument('--cognito-user-pool-id')
    parser.add_argument('--cognito-client-id')
    
    # Support multiple targets
    parser.add_argument('--electrify-lambda-arn', help='Lambda ARN for Electrify MCP server')
    parser.add_argument('--dataviz-lambda-arn', help='Lambda ARN for DataViz MCP server')
    
    # Legacy single-target mode (for backward compatibility)
    parser.add_argument('--lambda-arn', help='(Legacy) Single Lambda ARN - auto-detects type from gateway name')
    
    args = parser.parse_args()
    
    # Build target list
    targets = []
    if args.electrify_lambda_arn:
        targets.append(MCPTarget(lambda_arn=args.electrify_lambda_arn, server_type="electrify"))
    if args.dataviz_lambda_arn:
        targets.append(MCPTarget(lambda_arn=args.dataviz_lambda_arn, server_type="dataviz"))
    
    # Legacy mode: single --lambda-arn with type detection from gateway name
    if args.lambda_arn and not targets:
        server_type = "dataviz" if "dataviz" in args.gateway_name else "electrify"
        targets.append(MCPTarget(lambda_arn=args.lambda_arn, server_type=server_type))
    
    if not targets:
        parser.error("Provide at least one Lambda ARN: --electrify-lambda-arn, --dataviz-lambda-arn, or --lambda-arn")
    
    config = MCPDeploymentConfig(
        region=args.region,
        gateway_name=args.gateway_name,
        targets=targets,
        cognito_user_pool_id=args.cognito_user_pool_id,
        cognito_client_id=args.cognito_client_id
    )
    
    deployer = MCPServerDeployer(config)
    result = deployer.deploy()
    
    print("\n" + "="*60)
    print("MCP GATEWAY DEPLOYMENT COMPLETED!")
    print("="*60)
    print(f"Gateway ID: {result['gateway_id']}")
    print(f"Gateway URL: {result['gateway_url']}")
    print(f"Targets: {', '.join(result['targets'])}")
    print(f"Region: {result['region']}")
    print("="*60)
    
    with open('mcp_gateway_deployment.json', 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
