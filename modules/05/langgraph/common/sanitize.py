# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
Shared message sanitization middleware for Amazon Bedrock compatibility.

Handles:
- Flattening list-type content blocks to plain strings
- Stripping base64 chart data from tool messages (LLM doesn't need to see it)
- Deduplicating tool call IDs in conversation history
- Ensuring tool outputs are plain strings

Usage:
    from common.sanitize import sanitize_messages_middleware, sanitize_tool_output

    agent = create_agent(
        model=llm,
        tools=tools,
        middleware=[sanitize_messages_middleware, sanitize_tool_output],
    )
"""

import re
from typing import Callable
from langchain_core.messages import ToolMessage, AIMessage
from langchain.agents.middleware import wrap_model_call, wrap_tool_call, ModelRequest, ModelResponse


def sanitize_message(msg):
    """Sanitize a single message to be Bedrock-compatible."""
    if isinstance(msg, ToolMessage):
        content = msg.content
        if isinstance(content, list):
            text_parts = [
                block.get('text', str(block)) if isinstance(block, dict) else str(block)
                for block in content
            ]
            content = '\n'.join(text_parts)
        elif not isinstance(content, str):
            content = str(content)
        # Strip base64 chart data — LLM doesn't need to see it
        content = re.sub(
            r'<chart>data:image/[^;]+;base64,[^<]+</chart>',
            '<chart>Chart generated successfully</chart>',
            content
        )
        return ToolMessage(
            content=content,
            tool_call_id=msg.tool_call_id,
            name=getattr(msg, 'name', None)
        )
    return msg


@wrap_model_call
async def sanitize_messages_middleware(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Middleware to sanitize and deduplicate messages before sending to LLM."""
    messages = request.messages
    seen_tool_call_ids = set()
    deduped = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if msg.tool_call_id in seen_tool_call_ids:
                continue
            seen_tool_call_ids.add(msg.tool_call_id)
            deduped.append(sanitize_message(msg))
        elif isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            seen_ai_tool_ids = set()
            unique_tool_calls = []
            for tc in msg.tool_calls:
                tc_id = tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None)
                if tc_id and tc_id not in seen_ai_tool_ids:
                    seen_ai_tool_ids.add(tc_id)
                    unique_tool_calls.append(tc)
            if unique_tool_calls != msg.tool_calls:
                msg = AIMessage(content=msg.content, tool_calls=unique_tool_calls)
            deduped.append(msg)
        else:
            deduped.append(sanitize_message(msg))
    return await handler(request.override(messages=deduped))


@wrap_tool_call
async def sanitize_tool_output(request, handler):
    """Middleware to ensure tool outputs are plain strings for Amazon Bedrock compatibility."""
    result = await handler(request)
    if hasattr(result, 'content'):
        content = result.content
        if isinstance(content, list):
            text_parts = [
                block.get('text', str(block)) if isinstance(block, dict) else str(block)
                for block in content
            ]
            content = '\n'.join(text_parts)
        elif not isinstance(content, str):
            content = str(content)
        return ToolMessage(
            content=content,
            tool_call_id=result.tool_call_id if hasattr(result, 'tool_call_id') else request.tool_call.get('id', ''),
            name=result.name if hasattr(result, 'name') else request.tool_call.get('name', '')
        )
    return result