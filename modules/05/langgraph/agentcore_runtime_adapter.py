#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
AgentCore Runtime Adapter for Orchestrator Agent with AgentCore Memory

This adapter wraps the OrchestratorAgentWithMemory to work with Amazon Bedrock
AgentCore Runtime. It uses AgentCore Memory for persistent conversation state
that survives runtime restarts.

Key Features:
- Persistent memory using AgentCore Memory service
- Session ID maps to LangGraph thread_id
- Actor ID support for multi-user scenarios
- Automatic session management via RequestContext
"""

import os
import sys
import json as _json
import asyncio
import logging
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- OTel setup ---
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
LangchainInstrumentor().instrument()
# --- End OTel setup ---

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.runtime.context import RequestContext
from bedrock_agentcore.memory.session import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole

from orchestrator_agent import OrchestratorAgent
from common.types import AgentConfig, StdioServerConfig, IdentityContext
from common.prompts import orchestrator_prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("04 - Adapter code called")

# Initialize AgentCore app
app = BedrockAgentCoreApp(debug=True)


async def _setup_agent(payload: Dict[str, Any], context: RequestContext = None):
    """Common setup logic for agent initialization."""
    # Extract prompt from payload
    prompt = payload.get('prompt', '')
    session_id = context.session_id if context else os.getenv('SESSION_ID', 'default-session')
    
    if not prompt:
        return None, None, session_id, "Missing 'prompt' field in request payload"
    
    # Extract JWT from Authorization header (sole identity source)
    identity = None
    jwt_token = None
    if context and context.request_headers:
        auth_header = context.request_headers.get('authorization') or context.request_headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            jwt_token = auth_header[7:]
            logger.info(f"JWT token extracted from headers (len={len(jwt_token)})")

    # Derive identity from JWT claims
    if not identity and jwt_token:
        try:
            import base64
            claims_b64 = jwt_token.split('.')[1]
            claims_b64 += '=' * (4 - len(claims_b64) % 4)
            claims = _json.loads(base64.urlsafe_b64decode(claims_b64))
            identity = claims.get('email') or claims.get('username') or claims.get('cognito:username') or claims.get('sub')
            logger.info(f"Identity derived from JWT claims: {identity}")

            # If identity is still a UUID-shaped string (Cognito sub from access
            # token), look up the email via cognito-idp.GetUser using the access
            # token itself (no IAM permission needed). Access tokens lack 'email'
            # claim by default; ID tokens have it but AgentCore Runtime's JWT
            # authorizer requires access tokens (validates 'client_id' claim).
            looks_like_uuid = isinstance(identity, str) and len(identity) == 36 and identity.count('-') == 4
            if claims.get('token_use') == 'access' and looks_like_uuid:
                try:
                    import boto3
                    region = os.getenv('AWS_REGION', 'us-east-1')
                    cognito = boto3.client('cognito-idp', region_name=region)
                    user_resp = cognito.get_user(AccessToken=jwt_token)
                    attrs = {a['Name']: a['Value'] for a in user_resp.get('UserAttributes', [])}
                    email = attrs.get('email')
                    if email:
                        logger.info(f"Identity remapped via Cognito GetUser: {identity} -> {email}")
                        identity = email
                except Exception as e:
                    logger.warning(f"Could not look up email via Cognito GetUser: {e}")
        except Exception as e:
            logger.debug(f"Could not decode JWT for identity: {e}")

    logger.info(f"Identity resolved: {identity}")

    if not jwt_token:
        logger.warning("No JWT token found in headers")
    
    # Create IdentityContext if we have identity info
    identity_context = None
    if identity or jwt_token:
        identity_context = IdentityContext(
            username=identity or 'unknown',
            sub=identity or 'unknown',
            email=identity if identity and '@' in identity else None,
            jwt_token=jwt_token
        )
    
    logger.info(f"Processing request for session {session_id}, identity {identity}: {prompt[:100]}...")
    
    # Use AgentCore Memory for persistent conversation state.
    # The agentcore CDK construct auto-provisions a Memory from `memories: [...]`
    # in agentcore.json and sets MEMORY_<NAME>_ID; Module 10 may override this
    # by injecting BEDROCK_AGENTCORE_MEMORY_ID via update-agent-runtime.
    memory_id = os.getenv('BEDROCK_AGENTCORE_MEMORY_ID') or os.getenv('MEMORY_ELECTRIFY_STM_ID')
    if memory_id:
        logger.info(f"Using AgentCore Memory: {memory_id}")
    else:
        logger.info("Using InMemorySaver (no persistent memory)")
    
    config = AgentConfig(
        name="orchestrator_agent",
        description="This agent reasons and decides what downstream tools or agents to invoke to complete the user request.",
        identity=identity or 'unknown',
        thread=session_id,
        system_prompt=orchestrator_prompt(),
        model=os.getenv('AGENT_MODEL_ID', 'global.anthropic.claude-sonnet-4-6'),
        memory=memory_id or '',
        region=os.getenv('AWS_REGION', 'us-east-1'),
        identity_context=identity_context,
    )

    agent = OrchestratorAgent(config)
    await agent.setup()
    agent.config.thread = session_id

    # Query long-term memories and prepend to prompt
    memory_session = None
    original_prompt = prompt
    if memory_id:
        try:
            actor_id = (identity or 'unknown').replace('@', '-').replace('.', '-')
            mgr = MemorySessionManager(memory_id=memory_id, region_name=config.region)
            memory_session = mgr.create_memory_session(actor_id=actor_id, session_id=session_id)

            # Search each strategy's concrete namespace (the service doesn't support "/" prefix traversal).
            import boto3 as _boto3
            _ctl = _boto3.client("bedrock-agentcore-control", region_name=config.region)
            _strategies = _ctl.get_memory(memoryId=memory_id)["memory"].get("strategies", [])

            def _expand_ns(tpl: str, sid: str) -> str:
                return (tpl
                        .replace("{memoryStrategyId}", sid)
                        .replace("{actorId}", actor_id)
                        .replace("{sessionId}", session_id))

            records = []
            for _strat in _strategies:
                _sid = _strat.get("strategyId", "")
                for _ns_tpl in _strat.get("namespaces", []):
                    _ns = _expand_ns(_ns_tpl, _sid)
                    try:
                        _found = memory_session.search_long_term_memories(
                            query=prompt, namespace_prefix=_ns, top_k=5
                        )
                        if _found:
                            records.extend(_found)
                    except Exception as _e:
                        logger.debug(f"LTM search skipped ns={_ns}: {_e}")

            if records:
                facts = [r.get('content', {}).get('text', '') for r in records if r.get('content', {}).get('text')]
                # Dedupe while preserving order, cap at top 8 to keep prompt concise
                seen = set()
                facts = [f for f in facts if not (f in seen or seen.add(f))][:8]
                if facts:
                    ltm_context = "\n".join(f"- {f}" for f in facts)
                    prompt = f"[Recalled from previous sessions]\n{ltm_context}\n\n[Current request]\n{prompt}"
                    logger.info(f"Injected {len(facts)} long-term memories into prompt")
        except Exception as e:
            logger.warning(f"LTM retrieval failed (non-fatal): {e}")

    return agent, prompt, session_id, original_prompt, memory_session, None


def _write_memory_turns(memory_session, user_prompt: str, agent_response: str):
    """Write user prompt and agent response to memory for LTM strategy extraction."""
    try:
        memory_session.add_turns(messages=[
            ConversationalMessage(user_prompt, MessageRole.USER),
            ConversationalMessage(agent_response, MessageRole.ASSISTANT),
        ])
        logger.info("Wrote conversation turns to memory for LTM extraction")
    except Exception as e:
        logger.warning(f"Failed to write memory turns (non-fatal): {e}")


@app.entrypoint
async def orchestrator_handler(payload: Dict[str, Any], context: RequestContext = None):
    """Streaming handler for orchestrator agent."""
    session_id = None
    try:
        logger.info("Initializing OrchestratorAgent...")
        
        # Check if streaming is requested
        stream_enabled = payload.get('stream', False)
        
        agent, prompt, session_id, original_prompt, memory_session, error = await _setup_agent(payload, context)
        
        if error:
            yield {"error": error, "session_id": session_id}
            return
        
        logger.info("OrchestratorAgent initialized successfully")
        
        if stream_enabled:
            # Streaming mode - yield chunks as they come
            logger.info("Streaming mode enabled")
            collected_text = []
            try:
                async for chunk in agent.stream(prompt):
                    if isinstance(chunk, dict) and chunk.get("type") == "text":
                        collected_text.append(chunk.get("content", ""))
                    yield chunk
            except (ExceptionGroup, BaseExceptionGroup) as eg:
                # Extract meaningful error messages from the exception group
                error_msgs = []
                for exc in eg.exceptions:
                    error_msgs.append(str(exc))
                combined = "; ".join(error_msgs)
                logger.error(f"Tool call error during streaming: {combined}")
                yield {"type": "text", "content": f"\n\n⚠️ A tool call was denied by policy: {combined}"}
                yield {"type": "done"}
            
            # Write conversation turns to memory for LTM extraction
            if memory_session and collected_text:
                agent_response = "".join(collected_text)
                _write_memory_turns(memory_session, original_prompt, agent_response)
        else:
            # Non-streaming mode - return full result
            try:
                result = await agent.invoke(prompt)
            except (ExceptionGroup, BaseExceptionGroup) as eg:
                error_msgs = []
                for exc in eg.exceptions:
                    error_msgs.append(str(exc))
                combined = "; ".join(error_msgs)
                logger.error(f"Tool call error: {combined}")
                yield {
                    "result": {"text": f"⚠️ A tool call was denied by policy: {combined}", "images": []},
                    "session_id": session_id
                }
                return
            logger.info(f"Request processed successfully for session {session_id}")
            
            # Write conversation turns to memory for LTM extraction
            if memory_session:
                agent_text = result.get("text", "") if isinstance(result, dict) else str(result)
                _write_memory_turns(memory_session, original_prompt, agent_text)
            
            response = {
                "result": result,
                "session_id": session_id
            }
            
            identity = payload.get('identity')
            if identity:
                response["identity"] = identity
            
            yield response
        
    except (ExceptionGroup, BaseExceptionGroup) as eg:
        error_msgs = [str(exc) for exc in eg.exceptions]
        combined = "; ".join(error_msgs)
        logger.error(f"Tool call error (outer): {combined}", exc_info=True)
        if payload.get('stream', False):
            yield {"type": "text", "content": f"\n\n⚠️ A tool call was denied by policy: {combined}"}
            yield {"type": "done"}
        else:
            yield {
                "result": {"text": f"⚠️ A tool call was denied by policy: {combined}", "images": []},
                "session_id": session_id
            }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        yield {
            "error": f"Agent error: {str(e)}",
            "session_id": session_id
        }


@app.ping
def health_check():
    """Custom health check for the orchestrator agent."""
    from bedrock_agentcore.runtime import PingStatus
    
    # Agent not yet initialized (will be on first request)
    return PingStatus.HEALTHY


if __name__ == "__main__":
    # Run the AgentCore app
    app.run()
