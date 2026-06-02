#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Sample code, non-production. See README.md for full disclaimer.
"""Test script for ElectrifyMCPServer"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from server import ElectrifyMCPServer
import argparse

async def test_get_bills(username):
    """Test the get_bills tool implementation"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--endpoint', default=os.getenv('PGHOST', 'localhost'))
    parser.add_argument('-p', '--port', default=os.getenv('PGPORT', '5432'))
    parser.add_argument('-d', '--database', default=os.getenv('PGDBNAME', 'postgres'))
    parser.add_argument('-u', '--user', default=os.getenv('PGUSER', 'postgres'))
    parser.add_argument('--password', default=os.getenv('PGPASSWORD', ''))
    
    args = parser.parse_args([])
    
    print(f"Testing get_bills tool for user: {username}...")
    
    try:
        server = ElectrifyMCPServer(args)
        result = await server._get_bills({"customer_username": username, "limit": 10})
        
        import json
        bills = json.loads(result[0].text)
        
        if len(bills) > 0:
            print(f"✓ Successfully retrieved {len(bills)} bills for {username}")
            
            required_fields = ['invoice_no', 'bill_date', 'due_date', 'invoice_amount']
            if all(field in bills[0] for field in required_fields):
                print(f"✓ Bills contain required fields: {', '.join(required_fields)}")
            else:
                print("✗ Missing required fields in bills")
                return False
                
            print("\nSample bill:")
            print(json.dumps(bills[0], indent=2))
        else:
            print("✗ No bills returned")
            return False
            
        print("\nAll tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the get_bills MCP tool')
    parser.add_argument('username', help='Customer username to test (the email you created during login)')
    args = parser.parse_args()
    
    success = asyncio.run(test_get_bills(args.username))
    sys.exit(0 if success else 1)
