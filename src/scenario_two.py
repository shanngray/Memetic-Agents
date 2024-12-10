import asyncio
import httpx
from datetime import datetime
from api_server.models.api_models import APIMessage

async def run_scenario():
    # Create message for MathAgent
    message = APIMessage(
        sender="ScenarioRunner",
        receiver="BloggerAgent",
        content="I need to write a blog post about the latest developments in LLM Agents. Can you find a team of agents to help?",
        conversation_id="blog_request_001",
        timestamp=datetime.now().isoformat()
    )

    # Send message to MathAgent
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8012/receive",
                json=message.dict(),
                timeout=30.0
            )
            response.raise_for_status()
            print(f"Response from BloggerAgent: {response.json()}")
        except Exception as e:
            print(f"Error sending message: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_scenario())
