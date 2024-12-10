import asyncio
import httpx
from datetime import datetime
from api_server.models.api_models import APIMessage

async def run_scenario():
    # Create initial message for FrankenAgent
    message = APIMessage(
        sender="User",
        receiver="FrankenAgent",
        content="""I'd like you to:
1. Find an agent that can help you search for information on MemGPT and/or MemoryGPT.
2. Look for agents that can help you write and edit a blog post about MemGPT and/or MemoryGPT. (Remember to send them your findings!)
3. Reflect on what you've learned from both the agents and the news

Please organize your findings in a clear and structured way.""",
        conversation_id="franken_exploration_001",
        timestamp=datetime.now().isoformat()
    )

    # Send message to FrankenAgent
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8015/receive",  # Using FrankenAgent's port
                json=message.dict(),
                timeout=6000.0  # Increased timeout for longer interaction
            )
            response.raise_for_status()
            print(f"Response from FrankenAgent: {response.json()}")
        except Exception as e:
            print(f"Error sending message: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_scenario())
