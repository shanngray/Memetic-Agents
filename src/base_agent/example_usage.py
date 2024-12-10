import asyncio
import os
from base_agent import BaseAgent
from config import AgentConfig

async def main():
    # Create custom config
    config = AgentConfig(
        debug=True,
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant skilled at math and time-related queries.",
        agent_name="MathTime Assistant",
        enabled_tools=["get_current_time", "calc"]
    )
    
    agent = BaseAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        config=config
    )
    
    # Run interactive mode
    await agent.run_interactive()

if __name__ == "__main__":
    asyncio.run(main())