#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
# Test the ElectrifyMCPServer get_bills tool (Strands version)
# Usage: ./test_server.sh <username>
# Example: ./test_server.sh user@example.com

set -euo pipefail

USERNAME="${1:?Usage: $0 <customer_username>}"

echo "Testing MCP server (strands) for user: ${USERNAME}"
uv run modules/02/strands/test_server.py "${USERNAME}"
