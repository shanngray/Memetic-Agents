import asyncio
import httpx
from datetime import datetime
from base_agent.models import AgentStatus
from icecream import ic

async def test_direct_server_call():
    """Test calling the server endpoint directly with different payload formats"""
    async with httpx.AsyncClient() as client:
        test_cases = [
            "idle",        # String format
            "IDLE",        # Uppercase string
            "processing",  # Another valid status
            "invalid"      # Invalid status
        ]
        
        print("\n=== Testing Direct Server Calls ===")
        for status in test_cases:
            try:
                print(f"\nTesting status: {status}")
                # Test with query parameters
                print("\nTesting with query parameters:")
                response = await client.post(
                    "http://localhost:8000/agent/Curio/status",
                    params={"status": status}
                )
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")
    
            except Exception as e:
                print(f"Error: {str(e)}")

async def test_direct_agent_call():
    """Test calling the agent's status endpoint directly"""
    async with httpx.AsyncClient() as client:
        test_cases = [
            ("String value", "idle"),
            ("Uppercase", "IDLE"),
            ("Invalid status", "invalid")
        ]
        
        print("\n=== Testing Direct Agent Calls ===")
        for desc, status in test_cases:
            try:
                print(f"\nTesting {desc}: {status}")
                response = await client.post(
                    "http://localhost:8016/status",
                    params={"status": status}
                )
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")
            except Exception as e:
                print(f"Error: {str(e)}")

async def test_status_enum_validation():
    """Test the AgentStatus enum validation"""
    print("\n=== Testing Status Enum Validation ===")
    test_statuses = ["idle", "IDLE", "processing", "PROCESSING", "invalid_status"]
    
    for status in test_statuses:
        print(f"\nTesting status: {status}")
        try:
            status_enum = AgentStatus(status)  # Adjust based on enum's case handling
            print(f"Valid status enum: {status_enum}")
        except ValueError as e:
            print(f"Invalid status: {str(e)}")

async def test_status_transitions():
    """Test valid status transitions"""
    print("\n=== Testing Status Transitions ===")
    current_status = AgentStatus.IDLE
    print(f"Current status: {current_status}")
    
    valid_transitions = AgentStatus.get_valid_transitions(current_status)
    print(f"Valid transitions: {[t.value for t in valid_transitions]}")

async def test_full_pipeline():
    """Test the complete status update pipeline with detailed logging"""
    print("\n=== Testing Full Pipeline ===")
    
    test_cases = [
        ("idle", "standard lowercase"),
        ("IDLE", "standard uppercase"),
        ("Processing", "mixed case"),
        ("invalid", "invalid status"),
    ]
    
    for status, desc in test_cases:
        print(f"\nTesting {desc} - Status: {status}")
        
        # Step 1: Status validation
        print("1. Validating status...")
        try:
            status_enum = AgentStatus(status)  # Adjust based on enum's case handling
            print(f"✓ Status validated: {status_enum}")
        except ValueError as e:
            print(f"✗ Status validation failed: {str(e)}")
            continue
            
        # Step 2: API call
        print("2. Making API call...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8000/agent/Curio/status",
                    params={"status": status}
                )
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"✗ API call failed: {str(e)}")

async def main():
    """Run all tests sequentially including new tests"""
    await test_direct_server_call()
    await test_status_enum_validation()
    await test_status_transitions()
    await test_direct_agent_call()
    await test_full_pipeline()

if __name__ == "__main__":
    asyncio.run(main())
