# Threat Model

This document describes the threat model for the sample multi-agent
application in this repository. It is **not** a deployment guide.

## Scope

This sample demonstrates a multi-agent agentic-AI pattern using:

- Amazon Bedrock AgentCore (Memory, Gateway, Runtime)
- Amazon Aurora PostgreSQL
- The Model Context Protocol (MCP)
- LangGraph and Strands Agents SDK

The sample is published as **non-production educational content** under the
MIT-0 license. It is intended to be deployed only in development/sandbox AWS
accounts.

## In-scope assets

| Asset | Where it lives |
|---|---|
| MCP server code (Python) | `modules/02/{langgraph,strands}/server.py`, `modules/05/.../{electrify,dataviz}_server.py` |
| Agent code (Python) | `modules/02/.../agent.py`, `modules/03/.../dataviz.py`, `modules/04/.../orchestrator_agent.py` |
| Runtime adapter | `modules/05/.../agentcore_runtime_adapter.py` |
| Lambda + Gateway deploy scripts | `modules/05/.../deploy_lambda.py`, `deploy_gateway_simple.py` |
| Memory configuration | `modules/05/.../deploy/setup_deploy.sh` |

## Out of scope

- Workshop infrastructure (CloudFormation, VPC, IAM roles) — not shipped in
  this repo
- Web frontend, API gateway, Cognito user pool — not shipped
- Test harness, content authoring tools — not shipped
- Authorization policies (Cedar) — covered in the companion workshop,
  intentionally excluded from this sample; implementing fine-grained
  authorization is a production concern documented in the workshop

## Trust boundaries

```
┌─ External (untrusted) ──────────────────────────────────────────┐
│ Workshop participant / developer running the sample             │
└──────────────────┬──────────────────────────────────────────────┘
                   │ Bedrock InvokeModel (signed via SigV4)
                   ▼
┌─ AWS-managed ────────────────────────────────────────────────────┐
│ Bedrock AgentCore Runtime / Gateway / Memory                     │
│ Bedrock model endpoints (Claude Sonnet 4.6)                      │
│                                                                   │
│  ┌─ Customer-managed (this sample's deployment) ──────────────┐ │
│  │ Lambda (Electrify, DataViz) — invoked by Gateway via       │ │
│  │   AssumeRole; reads RDS via Data API + Secrets Manager     │ │
│  │ Aurora PostgreSQL — accessed only via Lambda execution role│ │
│  │ Cognito user pool (workshop only) — JWT verifier on Gateway│ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Shared responsibility

This sample follows the AWS Shared Responsibility Model. The trust
boundary diagram above maps directly onto the responsibility split:

**AWS is responsible for** (Security *of* the Cloud):
- Patching, isolation, and physical security of the underlying
  Amazon Bedrock, Amazon Aurora, AWS Lambda, Amazon Cognito, and
  AgentCore (Gateway, Runtime, Memory) service infrastructure
- Foundation model safety classifiers and runtime sandboxing for
  Amazon Bedrock model invocations
- Encryption-at-rest defaults for Aurora storage volumes and
  AgentCore Memory storage
- TLS termination at AWS service endpoints

**The deployer is responsible for** (Security *in* the Cloud):
- IAM role creation, least-privilege scoping, and inline-policy
  resource ARNs (this sample's `deploy_lambda.py` and
  `deploy_gateway_simple.py` scope CloudWatch Logs to specific
  log-group ARNs; production deployments should audit any other
  IAM policies they add)
- S3 bucket security (Block Public Access, server-side encryption,
  TLS-required bucket policies) — `deploy_lambda.py` already applies
  these via `apply_s3_bucket_security()` on the Lambda packaging bucket
- Lambda resource-based policies restricting `lambda:InvokeFunction`
  to the specific gateway role ARN (opt-in via the
  `--gateway-role-arn` flag on `deploy_lambda.py`)
- Secret management — Aurora credentials should be kept in AWS
  Secrets Manager and rotated; the sample's `--password` CLI argument
  is acceptable only for local development
- Authorization enforcement at the data layer (e.g., row-level
  security in PostgreSQL, IAM conditions, or Cedar policies — the
  workshop's Module 06, intentionally excluded from this sample)
- Network isolation (deploying Aurora into private subnets, NAT
  Gateway for Lambda egress, security group scoping)
- Cost monitoring, abuse detection, rate limiting on the agent
  invocation surface
- Compliance assessments (HIPAA, PCI, GDPR, etc.) — this sample
  makes no compliance claims

This sample is **for educational use only**. Production deployment
requires applying every "deployer is responsible for" item above, plus
a fresh threat model against the deployer's specific data classification
and operational environment.

## Threats and mitigations

### T1: Prompt injection in agent tool descriptions

**Threat:** A malicious actor could craft a prompt that abuses the
orchestrator's tool descriptions to call sub-agents with attacker-chosen
arguments — for example, asking "show me Bob's bills" while logged in as
Alice.

**Likelihood:** Medium. LLMs are known to follow instructions embedded in
input.

**Mitigation:**
- The `username` is injected into the system prompt at runtime from the
  authenticated identity, not from user input. See
  `modules/05/.../orchestrator_agent.py` (`Current user identity:` block).
- The MCP tools `get_customer` and `get_bills` accept a `customer_username`
  parameter; the sample's orchestrator passes the authenticated identity, but
  the tool itself does not enforce that the caller is authorized for that
  username. **A production deployment must enforce this at the data access
  layer (e.g., RLS in PostgreSQL or row-scoped IAM conditions).**
- Sample includes a top-level disclaimer stating non-production use only.

### T2: SQL injection via MCP tool arguments

**Threat:** Tool arguments (e.g., `customer_username`) flow into SQL queries.
A malicious LLM-generated argument could inject SQL.

**Likelihood:** Low when implemented as shown.

**Mitigation:**
- All queries use parameterized statements with `%s` placeholders and
  positional parameters (psycopg). See `modules/02/.../server.py`
  (`_get_bills`, `_get_customer`, `_get_rates` methods).
- No string interpolation of user-controlled values into SQL is present in
  the sample code.

### T3: Credential leakage through agent output

**Threat:** Agents log conversation history and tool outputs. Database
errors or stack traces could leak schema info, IPs, or credentials.

**Likelihood:** Low.

**Mitigation:**
- Database connections use AWS Secrets Manager via the RDS Data API in the
  Lambda path; secrets are never read by the agent process.
- Local stdio MCP server accepts a `--password` argument for development
  only; the workshop's CFN template provisions a Secrets Manager secret and
  a SecretArn output rather than a plain password (out of scope for this
  repo, but referenced in module READMEs).
- The `system.md` system prompt explicitly tells the agent never to share
  sensitive information like full SSN or passwords.

### T4: Cross-session data bleed via AgentCore Memory

**Threat:** AgentCore Memory stores conversation traces and extracted facts
indexed by namespace. A misconfigured namespace could allow user A's facts
to be retrieved while user B is in session.

**Likelihood:** Medium if namespace misconfigured.

**Mitigation:**
- Memory namespaces in `setup_deploy.sh` are scoped to `actorId` (the
  authenticated user), not session ID, for Facts and Preferences strategies.
  This is a deliberate fix; cross-session recall for the same user is the
  intended behavior, cross-user recall is prevented by the namespace.
- Summaries are still session-scoped (`/summaries/{actorId}/{sessionId}`).
- The Memory IAM policy generated by `agentcore deploy` further restricts
  access to namespaces matching the authenticated actor's prefix.

### T5: MCP gateway authentication bypass

**Threat:** An attacker bypasses the gateway's JWT authentication and
invokes Lambda MCP servers directly.

**Likelihood:** Low.

**Mitigation:**
- AgentCore Gateway is configured with a Cognito JWT verifier
  (`customJwtAuthorizer`) — see `modules/05/.../deploy_gateway_simple.py`.
- Lambda execution is gated on the gateway's IAM role, not on the JWT
  itself, but the Gateway only invokes the Lambdas after JWT verification
  succeeds.
- The Lambda functions don't expose public endpoints; they're invoked only
  via the gateway's AWS-internal control plane.
- **A production deployment should additionally restrict Lambda invocations
  to the specific gateway role ARN.**

### T6: Outbound exfiltration via tool calls

**Threat:** A malicious LLM response could attempt to call tools that
exfiltrate data to attacker-controlled endpoints.

**Likelihood:** Low. The sample's tools are strictly scoped to Aurora reads
and chart generation; none call external HTTP endpoints.

**Mitigation:**
- All MCP tools in this sample are read-only against Aurora or pure
  computation (chart rendering).
- No `requests.get()` or arbitrary HTTP calls are exposed as tools.
- A production deployment that adds external tool calls should review them
  individually for SSRF and exfiltration risks.

### T7: Cost explosion via runaway agent loop

**Threat:** A misbehaving orchestrator could enter a tool-calling loop,
running up Bedrock token charges and Lambda invocations.

**Likelihood:** Medium during development.

**Mitigation:**
- The orchestrator system prompt enforces sequential tool use and a
  single-chart rule.
- LangGraph's `recursion_limit` is set to 50 in the orchestrator config.
- Strands has built-in iteration caps via the `Agent()` runtime.
- **Sample disclaimer notes participants should monitor AWS usage in
  development accounts.**

## Dependencies

The sample depends on third-party Python packages declared in
`pyproject.toml`. None are pinned to vulnerable versions at the time of
publication. Dependency vulnerability scanning is the responsibility of the
deployer; we recommend `uv pip audit` or equivalent.

## Logging and observability

The sample uses OpenTelemetry instrumentation (LangChain) for the LangGraph
path, which emits spans to Bedrock AgentCore's observability surface.
No PII is logged in span attributes; tool inputs/outputs are stored in
AgentCore Memory which is access-scoped per actor as described in T4.

## Production hardening checklist

If you adapt this sample for any deployment beyond a sandbox, apply these
steps in priority order. Each item is concrete and verifiable.

### Phase 1 — Identity and access (do first)

1. **Enable Cognito JWT verifier** on the AgentCore Gateway (already done by
   `deploy_gateway_simple.py`'s `customJwtAuthorizer` config). Verify with
   `aws bedrock-agentcore-control get-gateway --gateway-identifier <id>`
   and confirm `authorizerConfiguration.customJWTAuthorizer.discoveryUrl`
   points at your Cognito user pool.
2. **Restrict Lambda invocations to the gateway role** by passing
   `--gateway-role-arn <role-arn>` to `deploy_lambda.py`. This calls
   `lambda:AddPermission` to scope `lambda:InvokeFunction` to the principal.
   Verify with `aws lambda get-policy --function-name <name>`.
3. **Identity-equality checks already enforced** in M5's
   `electrify_server.py` Lambda handler — refuses cross-user access when
   the requested `customer_username` doesn't match the authenticated
   principal extracted from JWT. Replicate this pattern for any tool that
   accepts a user-scoped argument.

### Phase 2 — Data confidentiality

4. **Use AWS Secrets Manager for database credentials** instead of the
   sample's `--password` CLI flag (used here for local dev only).
   `secret = boto3.client("secretsmanager").get_secret_value(SecretId="electrify-db")`.
5. **TLS enforced on the PostgreSQL connection.** The sample's
   `_get_connection_string()` already appends `sslmode=require`; verify
   your Aurora cluster has `rds.force_ssl=1` set on the parameter group
   to enforce server-side as well.
6. **Confirm Aurora encryption at rest** is enabled at cluster creation
   (cannot be enabled retroactively). Verify with
   `aws rds describe-db-clusters --query 'DBClusters[].StorageEncrypted'`.

### Phase 3 — Least privilege

7. **IAM Resource ARN scoping is already applied** in the sample's
   `deploy_lambda.py` (CloudWatch Logs scoped to `/aws/lambda/<function>`)
   and `deploy_gateway_simple.py` (CloudWatch Logs scoped to
   `/aws/bedrock-agentcore/*`). Audit any IAM policies you add for
   `Resource: "*"` and replace with specific ARNs.
8. **S3 hardening already applied** by `apply_s3_bucket_security()` in
   `deploy_lambda.py`: Block Public Access, SSE-S3 default encryption,
   and a TLS-only bucket policy on the Lambda packaging bucket.
9. **Use row-level security or per-user IAM conditions** at the data layer.
   The workshop's [Module 06](https://catalog.workshops.aws/agentic-aurora)
   covers Cedar policies for fine-grained authorization; intentionally
   excluded from this OSS sample but available for production deployments.

### Phase 4 — Operational

10. **Enable AgentCore Observability** for runtime/gateway/memory traces.
    The sample's runtime adapter is already instrumented with OpenTelemetry;
    wire `LogDeliveryConfiguration` for X-Ray + CloudWatch.
11. **Set up cost guardrails** on Amazon Bedrock model invocations. The
    multi-agent pattern can fan out into many model calls per user request;
    add CloudWatch alarms on `Invocations` and `OutputTokens` metrics.

## Out-of-scope risks acknowledged

The following are not addressed by this sample and are the responsibility
of any production deployment:

1. **Authorization policy enforcement** — the workshop's Module 06 covers
   Cedar policy engines for fine-grained authorization. That module is
   intentionally excluded from this OSS publication; consult the workshop
   for guidance.
2. **Network isolation** — the workshop deploys Aurora into a VPC with
   private subnets; this sample shows agent code only.
3. **Secret rotation** — Aurora secrets in the workshop are managed by
   Secrets Manager with manual rotation.
4. **Compliance frameworks** (HIPAA, PCI, etc.) — none of these are claimed
   for the sample; deployers must do their own assessment.

## References

- AWS Workshop: [Build a Collaborative Agentic AI Solution with Amazon Aurora and Bedrock AgentCore](https://catalog.workshops.aws/agentic-aurora)
- [Amazon Bedrock AgentCore documentation](https://docs.aws.amazon.com/bedrock-agentcore/)
- [Model Context Protocol specification](https://modelcontextprotocol.io)
