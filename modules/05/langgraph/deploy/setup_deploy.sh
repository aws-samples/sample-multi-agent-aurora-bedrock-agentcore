#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
# =============================================================
# Setup the AgentCore deploy project for LangGraph track
# Run from: ~/workshop/modules/05/langgraph/deploy/
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up AgentCore deploy project (LangGraph) ==="

# Verify env vars
for var in AWS_REGION AGENTCORE_ROLE_ARN MCP_GATEWAY_URL COGNITO_POOL COGNITO_CLIENT OAUTH_ISSUER_URL; do
    if [ -z "${!var}" ]; then
        echo "ERROR: $var is not set. Run the gateway deployment steps first."
        exit 1
    fi
done

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# 1. Copy agent code into app/ directory
echo "Copying agent code..."
rm -rf app/electrify_assistant
mkdir -p app/electrify_assistant
cp ../agentcore_runtime_adapter.py app/electrify_assistant/
cp ../orchestrator_agent.py app/electrify_assistant/
cp ../electrify_agent.py app/electrify_assistant/
cp ../dataviz_agent.py app/electrify_assistant/
cp ../orchestrator_prompt.md app/electrify_assistant/
cp ../electrify_prompt.md app/electrify_assistant/
cp ../requirements.txt app/electrify_assistant/
cp -r ../common app/electrify_assistant/

# Create pyproject.toml with proper [project] dependencies for CLI bundling
cat > app/electrify_assistant/pyproject.toml <<'EOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "electrify_assistant"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "bedrock-agentcore>=1.0.5",
    "aws-opentelemetry-distro>=0.10.0",
    "opentelemetry-instrumentation-langchain>=0.40.0",
    "langchain>=0.3.0",
    "langchain-aws>=0.2.0",
    "langgraph>=0.3.0",
    "langgraph-checkpoint-aws>=1.0.2",
    "langchain-mcp-adapters>=0.1.0",
    "mcp>=1.16.0",
    "pandas>=2.1.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
    "matplotlib>=3.7.0",
    "boto3>=1.40.67",
    "python-dotenv>=1.0.0",
    "requests>=2.32.0",
    "psycopg[binary]>=3.2.0",
]

[tool.hatch.build.targets.wheel]
packages = ["."]
EOF

# 2. Write aws-targets.json with participant's account
cat > agentcore/aws-targets.json <<EOF
[
  {
    "name": "default",
    "account": "$ACCOUNT_ID",
    "region": "$AWS_REGION"
  }
]
EOF

# 3. Update agentcore.json with environment variables
cat > agentcore/agentcore.json <<EOF
{
  "\$schema": "https://schema.agentcore.aws.dev/v1/agentcore.json",
  "name": "electrifylanggraph",
  "version": 1,
  "managedBy": "CDK",
  "runtimes": [
    {
      "name": "electrify_assistant",
      "build": "CodeZip",
      "entrypoint": "agentcore_runtime_adapter.py",
      "codeLocation": "app/electrify_assistant/",
      "runtimeVersion": "PYTHON_3_12",
      "networkMode": "PUBLIC",
      "protocol": "HTTP",
      "envVars": [
        {"name": "MCP_GATEWAY_URL", "value": "$MCP_GATEWAY_URL"},
        {"name": "AGENT_MODEL_ID", "value": "global.anthropic.claude-sonnet-4-6"}
      ],
      "roleArn": "$AGENTCORE_ROLE_ARN",
      "authorizerType": "CUSTOM_JWT",
      "authorizerConfiguration": {
        "customJwtAuthorizer": {
          "discoveryUrl": "$OAUTH_ISSUER_URL",
          "allowedClients": ["$COGNITO_CLIENT"]
        }
      },
      "requestHeaderAllowlist": ["Authorization"]
    }
  ],
  "memories": [
    {
      "name": "electrify_stm",
      "eventExpiryDuration": 30,
      "strategies": [
        {
          "type": "SEMANTIC",
          "name": "Facts",
          "description": "Extract and store customer facts from conversations",
          "namespaces": ["/facts/{actorId}"]
        },
        {
          "type": "USER_PREFERENCE",
          "name": "Preferences",
          "description": "Extract user preferences, choices, and communication styles",
          "namespaces": ["/preferences/{actorId}"]
        },
        {
          "type": "SUMMARIZATION",
          "name": "Summaries",
          "namespaces": ["/summaries/{actorId}/{sessionId}"]
        },
        {
          "type": "EPISODIC",
          "name": "Episodes",
          "namespaces": ["/strategy/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}/"],
          "reflectionNamespaceTemplates": ["/strategy/{memoryStrategyId}/actors/{actorId}/"]
        }
      ]
    }
  ],
  "credentials": [],
  "evaluators": [],
  "onlineEvalConfigs": [],
  "agentCoreGateways": [],
  "policyEngines": [],
  "configBundles": [],
  "abTests": [],
  "httpGateways": []
}
EOF

# 4. Install CDK dependencies if not already installed
if [ ! -d "agentcore/cdk/node_modules" ]; then
    echo "Installing CDK dependencies..."
    cd agentcore/cdk && npm install --silent && cd ../..
fi

# 5. Validate
echo "Validating configuration..."
agentcore validate

echo ""
echo "=== Setup complete ==="
echo "Account: $ACCOUNT_ID"
echo "Region:  $AWS_REGION"
echo "Gateway: $MCP_GATEWAY_URL"
echo ""
echo "Next steps:"
echo "  agentcore deploy -y    # Deploy agent to AgentCore Runtime"
echo ""
echo "  agentcore invoke --bearer-token \$TOKEN --stream 'What rate plans are available?'"
