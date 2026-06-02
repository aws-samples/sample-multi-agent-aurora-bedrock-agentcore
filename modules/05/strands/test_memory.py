#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""
Test Long-Term Memory: Write events and query extracted memories from AgentCore Memory.

Supports multiple conversation scenarios for different strategy types:
  --scenario preferences   Customer preferences (good for semantic + user preference strategies)
  --scenario troubleshoot  Multi-step troubleshooting (good for episodic + summary strategies)
  --scenario billing       Billing inquiry session (good for summary strategy)

Usage:
    uv run test_memory.py --memory-id <MEMORY_ID> --write-event
    uv run test_memory.py --memory-id <MEMORY_ID> --write-event --scenario troubleshoot
    uv run test_memory.py --memory-id <MEMORY_ID> --query "customer preferences"
    uv run test_memory.py --memory-id <MEMORY_ID> --list-memories
    uv run test_memory.py --memory-id <MEMORY_ID> --list-events
"""

import os
import argparse
import boto3
from bedrock_agentcore.memory.session import MemorySessionManager
from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole


def expand_namespaces(memory_id, region, actor_id, session_id):
    """Fetch the memory's strategies and expand each namespace template.

    AgentCore Memory does not traverse '/' as a namespace prefix — every
    strategy stores records under its own concrete path. This helper reads
    the strategies via the control plane and substitutes the actor/session
    placeholders so callers can list or search per-namespace.
    """
    ctl = boto3.client("bedrock-agentcore-control", region_name=region)
    strategies = ctl.get_memory(memoryId=memory_id)["memory"].get("strategies", [])
    expanded = []
    for strat in strategies:
        sid = strat.get("strategyId", "")
        sname = strat.get("name", "")
        for ns_tpl in strat.get("namespaces", []):
            ns = (ns_tpl
                  .replace("{memoryStrategyId}", sid)
                  .replace("{actorId}", actor_id)
                  .replace("{sessionId}", session_id))
            expanded.append((sname, ns))
    return expanded


SCENARIOS = {
    "preferences": {
        "description": "Customer states rate plan preferences and communication style",
        "session_id": "memory-lab-preferences-001",
        "turns": [
            ("What rate plans do you offer? I'm looking for something eco-friendly.", "USER"),
            ("We offer three plans: Standard ($0.12/kWh), Green Energy ($0.15/kWh, 100% renewable), "
             "and Time-of-Use (variable rates). The Green Energy plan sounds like a great fit for you!", "ASSISTANT"),
            ("I definitely prefer the Green Energy plan. Also, can you always show me costs in monthly format?", "USER"),
            ("Noted! I'll show costs in monthly format going forward. The Green Energy plan would be "
             "approximately $45/month based on average usage in your area.", "ASSISTANT"),
            ("Great. I prefer email communication for any billing updates. My billing cycle starts on the 15th.", "USER"),
            ("Got it! I've noted your preference for email communication and your billing cycle starting on the 15th.", "ASSISTANT"),
        ]
    },
    "troubleshoot": {
        "description": "Multi-step troubleshooting: high bill investigation with tool use",
        "session_id": "memory-lab-troubleshoot-001",
        "turns": [
            ("My electricity bill this month is way higher than usual. Can you help me figure out why?", "USER"),
            ("I'll look into that for you. Let me pull up your recent billing history and usage data.", "ASSISTANT"),
            ("I can see your usage spiked significantly in the last two weeks. Your average daily usage went from "
             "18 kWh to 35 kWh starting around the 10th. Do you recall any changes — new appliances, "
             "guests staying over, or a change in heating/cooling?", "ASSISTANT"),
            ("We did get a new space heater around that time. Could that be it?", "USER"),
            ("That's very likely the cause. Space heaters are one of the most energy-intensive household appliances. "
             "A typical 1500W space heater running 8 hours a day adds about 12 kWh/day, which matches your spike. "
             "I'd recommend using it on a timer or switching to a lower wattage setting.", "ASSISTANT"),
            ("That makes sense. I'll put it on a timer. Can you also check if I'd save money on the Time-of-Use plan "
             "given my usage pattern?", "USER"),
            ("Based on your usage pattern — heavy evening usage from 6-10 PM — the Time-of-Use plan would actually "
             "cost you more since those are peak hours at $0.22/kWh. Your current Standard plan at $0.12/kWh flat "
             "rate is the better choice for your pattern. The issue was resolved by identifying the space heater "
             "as the cause of the usage spike.", "ASSISTANT"),
        ]
    },
    "billing": {
        "description": "Detailed billing inquiry covering multiple topics",
        "session_id": "memory-lab-billing-001",
        "turns": [
            ("Can you show me my bills for the last 6 months?", "USER"),
            ("Here are your last 6 bills: Jan $42.50, Feb $38.20, Mar $35.10, Apr $33.80, "
             "May $45.60, Jun $52.30. I notice your costs have been trending up since April.", "ASSISTANT"),
            ("Why did June jump so much?", "USER"),
            ("June's increase is primarily due to air conditioning usage during the heat wave. "
             "Your usage went from 280 kWh in May to 410 kWh in June. The region experienced "
             "record temperatures averaging 95°F that month.", "ASSISTANT"),
            ("Is there a way to reduce my summer bills?", "USER"),
            ("A few options: 1) Set your thermostat to 78°F instead of 72°F — saves about 15% on cooling. "
             "2) Use ceiling fans to supplement AC. 3) Consider our Time-of-Use plan if you can shift "
             "heavy usage to off-peak hours (10 PM - 6 AM). 4) Check if you qualify for our energy "
             "efficiency rebate program — up to $200 back for smart thermostat installation.", "ASSISTANT"),
            ("I'll try the thermostat adjustment. Also, can you set up autopay for my account?", "USER"),
            ("I've noted your interest in autopay. You can enable it through your account settings "
             "or I can walk you through it. You'll need your bank routing number and account number. "
             "Autopay processes on your billing date, which is the 15th of each month.", "ASSISTANT"),
        ]
    },
}


def format_record(record, show_score=False):
    """Format a memory record for readable output."""
    text = record.get('content', {}).get('text', str(record))
    strategy = record.get('memoryStrategyId', '').split('-')[0] or 'unknown'
    parts = [f'"{text}"']
    parts.append(f"strategy={strategy}")
    if show_score and 'score' in record:
        parts.append(f"relevance={record['score']:.2f}")
    return "  ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Query AgentCore Memory long-term memories")
    parser.add_argument('--memory-id', required=True, help='AgentCore Memory resource ID')
    parser.add_argument('--region', default=os.getenv('AWS_REGION', 'us-east-1'), help='AWS region')
    parser.add_argument('--actor-id', default='rroe-example-com',
                        help='Actor ID (user identity). The deployed agent derives this '
                             'from the JWT email by replacing "@" and "." with "-", '
                             'so user@example.com becomes rroe-example-com.')
    parser.add_argument('--session-id', default=None, help='Session ID (auto-set per scenario if omitted)')
    parser.add_argument('--scenario', default='preferences', choices=SCENARIOS.keys(),
                        help='Conversation scenario to write (default: preferences)')
    parser.add_argument('--query', default=None, help='Semantic search query for long-term memories')
    parser.add_argument('--list-events', action='store_true', help='List recent conversation events (STM)')
    parser.add_argument('--list-memories', action='store_true', help='List all extracted long-term memories')
    parser.add_argument('--write-event', action='store_true', help='Write a sample conversation event')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    session_id = args.session_id or scenario["session_id"]

    session_manager = MemorySessionManager(
        memory_id=args.memory_id,
        region_name=args.region
    )

    session = session_manager.create_memory_session(
        actor_id=args.actor_id,
        session_id=session_id
    )

    if args.write_event:
        print(f"\n📝 Writing conversation events: {scenario['description']}")
        print(f"   Memory:   {args.memory_id}")
        print(f"   Actor:    {args.actor_id}")
        print(f"   Session:  {session_id}")
        print(f"   Scenario: {args.scenario}\n")

        turns = scenario["turns"]
        turn_count = 0
        for text, role in turns:
            msg_role = MessageRole.USER if role == "USER" else MessageRole.ASSISTANT
            session.add_turns(messages=[ConversationalMessage(text, msg_role)])
            turn_count += 1

        print(f"✅ Wrote {turn_count} conversation turns.")
        print("⏳ Strategy extraction runs asynchronously — wait 30-60 seconds, then query with --list-memories or --query.\n")

    if args.list_events:
        print(f"\n📋 Recent conversation events (STM):")
        print(f"   Memory:  {args.memory_id}")
        print(f"   Actor:   {args.actor_id}")
        print(f"   Session: {session_id}\n")

        turns = session.get_last_k_turns(k=20)
        if not turns:
            print("   (no events found)")
        for i, turn in enumerate(turns, 1):
            print(f"   {i}. {turn}")
        print()

    if args.list_memories:
        print(f"\n🧠 Extracted long-term memories (LTM):")
        print(f"   Memory: {args.memory_id}")
        print(f"   Actor:  {args.actor_id}\n")

        namespaces = expand_namespaces(args.memory_id, args.region, args.actor_id, session_id)
        if not namespaces:
            print("   (no strategies declared on this memory)")
        total = 0
        for sname, ns in namespaces:
            records = session.list_long_term_memory_records(namespace_prefix=ns)
            print(f"   [{sname}] {ns}")
            if not records:
                print(f"      (no records yet)")
            for i, record in enumerate(records, 1):
                print(f"      {i}. {format_record(record)}")
                total += 1
            print()
        if total == 0:
            print("   (no memories extracted yet — strategies may still be processing; "
                  "Preferences and Facts can take 3-5 minutes after the conversation)")

    if args.query:
        print(f"\n🔍 Semantic search for: \"{args.query}\"")
        print(f"   Memory: {args.memory_id}")
        print(f"   Actor:  {args.actor_id}")
        print(f"   Top-K:  {args.top_k}\n")

        namespaces = expand_namespaces(args.memory_id, args.region, args.actor_id, session_id)
        total = 0
        for sname, ns in namespaces:
            records = session.search_long_term_memories(
                query=args.query,
                namespace_prefix=ns,
                top_k=args.top_k
            )
            if records:
                print(f"   [{sname}] {ns}")
                for i, record in enumerate(records, 1):
                    print(f"      {i}. {format_record(record, show_score=True)}")
                    total += 1
                print()
        if total == 0:
            print("   (no matching memories found — extraction may still be in progress)")
        print()

    if not any([args.write_event, args.list_events, args.list_memories, args.query]):
        print("\nNo action specified. Use one of:")
        print("  --write-event                Write sample conversation events")
        print("  --list-events                List recent conversation events (STM)")
        print("  --list-memories              List extracted long-term memories (LTM)")
        print("  --query TEXT                 Semantic search for memories")
        print()
        print("Scenarios (use with --write-event):")
        for name, info in SCENARIOS.items():
            print(f"  --scenario {name:15s}  {info['description']}")
        print()


if __name__ == "__main__":
    main()
