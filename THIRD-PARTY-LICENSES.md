# Third-Party Software Notices

This sample uses third-party software components. The components and their
licenses are listed below. The `pyproject.toml` + `uv.lock` pin exact
versions; this list is informational.

## Direct Python dependencies

The following are declared in `pyproject.toml` and pulled in via `uv sync`.
Customers running this sample install these directly from PyPI; the source
of these libraries is not redistributed in this repository.

### Apache License 2.0

- **boto3** — AWS SDK for Python
- **bedrock-agentcore** — Amazon Bedrock AgentCore SDK
- **bedrock-agentcore-starter-toolkit** — AgentCore CLI helpers
- **strands-agents** — Strands Agents SDK
- **strands-agents-tools** — Strands tools collection
- **opentelemetry-instrumentation-langchain** — OTel LangChain instrumentation
- **aws-opentelemetry-distro** — AWS OTel distribution
- **langgraph-checkpoint-aws** — LangGraph AgentCore Memory adapter

### MIT License

- **langchain** — LangChain framework
- **langchain-core**, **langchain-aws**, **langchain-mcp-adapters**
- **langgraph** — Stateful graph runtime
- **mcp** — Model Context Protocol Python SDK
- **python-dotenv** — Environment variable loader
- **pyyaml** — YAML parser
- **requests** — HTTP client

### BSD License

- **pandas** — Data analysis (BSD-3-Clause)
- **numpy** — Numerical computing (BSD-3-Clause)
- **matplotlib** — Plotting (Matplotlib License, BSD-style)

### LGPL (LGPL-3.0 / LGPLv2+, dynamically linked)

- **psycopg** + **psycopg-binary** + **psycopg-pool** — PostgreSQL adapter
  (LGPL-3.0). Used as a runtime dependency only; LGPL'd source is not
  bundled in this repository. Customers install via PyPI as part of normal
  Python dependency resolution.
- **chardet** — Character encoding detection (LGPLv2+). Transitive runtime
  dependency.

LGPL is acceptable here because:
- The LGPL'd source is dynamically loaded at runtime via `pip install`,
  not vendored or compiled into our distribution.
- This sample's source is licensed MIT-0; LGPL only governs its own components.
- Customers are free to swap psycopg for another PostgreSQL driver if their
  legal/compliance requirements forbid LGPL dependencies (psycopg2 is BSD;
  asyncpg is Apache 2.0).

## MCP server references

The Module 03 dataviz YAML configurations (`modules/03/strands/dataviz.yml`)
reference `fastmcp` as an MCP server runtime (`uvx fastmcp@latest`).

- **fastmcp** — Apache License 2.0
  https://github.com/jlowin/fastmcp

This is invoked at runtime via `uvx`; the FastMCP source is not bundled.

## Dependency vulnerability scanning

The full transitive dependency graph (197 packages) is captured in
`uv.lock`. Customers are responsible for vulnerability scanning the
deployed dependency graph in their environment using tools like
`pip-audit`, `safety`, or AWS Inspector. This sample's CI does not run
dependency scans; the dependency set was inventoried at the time of
publication.

## Generating an up-to-date license list

```bash
uv sync
uv run pip-licenses --format=plain
```
