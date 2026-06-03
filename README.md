# Multi-Agent AI with Amazon Bedrock AgentCore, Aurora, and MCP

Sample multi-agent application demonstrating how to build a collaborative
agentic AI system using **Amazon Bedrock AgentCore** (Memory, Gateway, Runtime),
**Amazon Aurora PostgreSQL**, and the **Model Context Protocol (MCP)**.

The repo includes parallel implementations in two agent frameworks —
**LangGraph** and **Strands Agents SDK** — so you can compare patterns
side by side.

> **⚠️ Sample code disclaimer:** This is sample code for non-production usage.
> You should work with your security and legal teams to meet your
> organizational security, regulatory, and compliance requirements before
> deployment.

> **Companion code for the AWS Workshop:**
> *[Build a Collaborative Agentic AI Solution with Amazon Aurora and Bedrock AgentCore](https://catalog.workshops.aws/agentic-aurora)*
>
> The full hands-on tutorial — with infrastructure provisioning, web frontend,
> end-to-end deployment, and a step-by-step walkthrough — lives at the
> workshop catalog. **Visit the workshop to deploy and run this code.**
> This repository is for studying or extending the agent code on its own.

## What's in this repo

```
modules/
├── 01/    — MCP primer (single file)
│            A minimal raw Model Context Protocol client (mcp.ClientSession
│            over stdio) — the protocol fundamentals before any agent
│            framework. Start here to see MCP with no abstractions.
├── 02/    — Custom MCP server + first agent (Electrify)
│            Aurora PostgreSQL queries exposed as MCP tools, agent
│            consumes them via stdio transport.
├── 03/    — Data visualization agent (DataViz)
│            Charting tools (bar, line, pie, scatter, histogram)
│            built with matplotlib + pandas.
├── 04/    — Orchestrator agent (multi-agent)
│            Coordinates Electrify and DataViz sub-agents to handle
│            requests like "show me my billing trend".
└── 05/    — AgentCore deployment
             Lambda MCP servers + AgentCore Gateway + Runtime adapter,
             with deploy scripts (deploy_lambda.py, deploy_gateway_simple.py).
```

Module numbering follows the companion workshop's chapters; this repo
includes the agent-code modules (01–05). Modules 02–05 each have both
`langgraph/` and `strands/` subdirectories implementing the same logical
agent in each framework; module 01 is a single framework-agnostic file.

## Architecture

```
                          ┌──────────────────┐
                          │ Orchestrator     │
                          │ (LangGraph or    │
                          │  Strands)        │
                          └─────────┬────────┘
                                    │
                ┌───────────────────┴────────────────────┐
                │                                        │
        ┌───────▼────────┐                       ┌───────▼────────┐
        │ Electrify      │                       │ DataViz        │
        │ Sub-Agent      │                       │ Sub-Agent      │
        └───────┬────────┘                       └────────────────┘
                │ MCP                                     ▲
                ▼                                         │
       ┌─────────────────────┐                            │
       │ AgentCore Gateway   │ ◄──── unified MCP endpoint ┘
       │ (single endpoint    │       with semantic search
       │  for all tools)     │
       └────────┬────────────┘
                │
       ┌────────▼─────────┐    ┌─────────────────┐
       │ Electrify Lambda │    │ DataViz Lambda  │
       │  - get_rates     │    │  - create_bar   │
       │  - get_customer  │    │  - create_line  │
       │  - get_bills     │    │  - create_pie   │
       └────────┬─────────┘    └─────────────────┘
                │
                ▼
       ┌─────────────────┐
       │ Aurora          │
       │ PostgreSQL      │
       └─────────────────┘
```

## Prerequisites

- AWS account with access to Amazon Bedrock and Aurora PostgreSQL
- Python 3.13+ and [uv](https://docs.astral.sh/uv/) for dependency management
- Node.js 22+ and the [`@aws/agentcore`](https://www.npmjs.com/package/@aws/agentcore)
  CLI for AgentCore deployment (Module 05)
- Required Python packages declared in the top-level `pyproject.toml`

## Running the code

The recommended path is to **launch the workshop event** at
[catalog.workshops.aws/agentic-aurora](https://catalog.workshops.aws/agentic-aurora) —
it provisions a complete environment (VS Code, Aurora, Cognito, the works) and
walks you through every module step by step.

If you want to study the code locally, install the deps and run a single
module's agent against your own Aurora cluster:

```bash
# Install dependencies
uv sync

# Run the MCP server + agent (requires Aurora connection details in env)
uv run modules/02/langgraph/agent.py \
  -p modules/02/langgraph/system.md \
  -m global.anthropic.claude-sonnet-4-6 \
  -s uv \
  -a "run modules/02/langgraph/server.py -e $PGHOST -u $PGUSER --password $PGPASSWORD" \
  -u alice@example.com
```

For deploying to AWS via AgentCore Runtime (Module 05), the workshop's Module 7
walkthrough is the canonical guide — it covers all the IAM, Cognito, gateway,
and CDK setup needed.

## Framework comparison

This repo intentionally implements every agent in both frameworks so you can
compare patterns. A few highlights of where they differ:

| Aspect | LangGraph | Strands |
|--------|-----------|---------|
| Tool decorator | `@tool(parse_docstring=True)` | `@tool` |
| Async model | `async`/`await` throughout | sync by default |
| Tool output sanitization | middleware (`sanitize_tool_output`) | built into SDK |
| Session persistence (local) | in-memory or PostgreSQL checkpointer | `FileSessionManager` |
| AgentCore Memory adapter | `AgentCoreMemorySaver` (`langgraph_checkpoint_aws`) | `MemorySessionManager` |

## Security

This sample follows the **AWS Shared Responsibility Model**:

- **AWS is responsible for** the security *of* the cloud — the underlying
  infrastructure of Amazon Bedrock, Amazon Aurora, AWS Lambda, Amazon
  Cognito, and the AgentCore Gateway/Runtime/Memory services this sample
  invokes.
- **You are responsible for** the security *in* the cloud — your AWS
  account configuration, IAM roles you create from this code, network
  controls, secret management, encryption settings on resources you
  provision, and operational monitoring of any deployment derived from
  this sample.

This sample is intended for educational use. Production hardening (S3
encryption, KMS key management, IAM least-privilege scoping, Lambda
resource policies, network isolation, secret rotation, regulatory
compliance) is your responsibility before any deployment that handles
real customer data. See [THREAT_MODEL.md](THREAT_MODEL.md) for a
detailed scope statement, threat analysis, and a Production Hardening
Checklist.

If you discover a potential security issue in this project, please follow the
[AWS vulnerability reporting process](http://aws.amazon.com/security/vulnerability-reporting/).
**Please do not open a public GitHub issue.**

## License

This sample code is licensed under the MIT-0 License. See the
[LICENSE](LICENSE) file for details.
