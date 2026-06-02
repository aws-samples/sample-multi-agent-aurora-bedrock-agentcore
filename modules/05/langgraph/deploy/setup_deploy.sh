#!/bin/bash
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
    # Pinned to the repo's committed uv.lock — the exact tree the test harness
    # validates. The AgentCore CodeZip build resolves these server-side at deploy
    # time; floor pins (>=) let it drift to latest-on-PyPI (the strands track
    # broke 2026-06-01 when a floor-pinned deploy pulled a same-day release).
    # Exact pins keep the deployed runtime identical to what the harness ran.
    # See feedback_version_upgrades: deploy what you tested.
    "bedrock-agentcore==1.8.0",
    "aws-opentelemetry-distro==0.17.0",
    "opentelemetry-instrumentation-langchain==0.60.0",
    "langchain==1.2.17",
    "langchain-core==1.3.2",
    "langchain-aws==1.4.5",
    "langgraph==1.1.10",
    "langgraph-checkpoint-aws==1.0.7",
    "langchain-mcp-adapters==0.2.2",
    "mcp==1.27.0",
    "pandas==3.0.2",
    "numpy==2.4.4",
    "pyyaml==6.0.3",
    "matplotlib==3.10.9",
    "boto3==1.43.1",
    "python-dotenv==1.2.2",
    "requests==2.33.1",
    "psycopg[binary]==3.3.3",
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
