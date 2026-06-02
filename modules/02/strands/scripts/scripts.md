<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
Sample code, non-production. See README.md for full disclaimer.
-->

# Module 02 - Strands Version Test Commands

## Test the MCP Server

```bash
uv run modules/02/strands/test_server.py user@example.com
```

## Launch the Strands Agent

```bash
uv run modules/02/strands/agent.py -p modules/02/langgraph/system.md -m "global.anthropic.claude-sonnet-4-6" -s uv -a "run modules/02/strands/server.py -e $PGHOST -u $PGUSER --password \"$PGPASSWORD\"" -u user@example.com -t conv-001
```