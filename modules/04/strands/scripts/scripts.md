<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT-0
Sample code, non-production. See README.md for full disclaimer.
-->

# Module 04 - Strands Version Test Commands

## Launch the Orchestrator Agent

```bash
uv run modules/04/strands/orchestrator_agent.py -m "global.anthropic.claude-sonnet-4-6" -u user@example.com -t conv-001
```

## Test Queries

Once the orchestrator is running, try these queries:

**Simple database query (single agent):**
```
What rate plans are available?
```

**User-specific data query (single agent):**
```
What is my last month's bill?
```

**Usage-based recommendation (single agent):**
```
Based on my usage, can you suggest a better rate plan?
```

**Multi-agent workflow (chained operation):**
```
Can you plot a pie chart of my last 6 months bills
```

## Exit

```
quit
```