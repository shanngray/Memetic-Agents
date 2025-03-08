import os
from pathlib import Path
from src.memetic_agent.memetic_agent import MemeticAgent
from src.base_agent.config import AgentConfig
from src.base_agent.type import Agent
from chromadb import PersistentClient

def create_jerry_agent(chroma_client: PersistentClient) -> MemeticAgent:
    """Create an MemeticAgent instance."""
    
    config = AgentConfig(
        debug=os.getenv("AGENT_DEBUG", "false").lower() == "true",
        console_logging=True,
        model="gpt-4o",
        submodel="gpt-4o-mini",
        temperature=1.0,
        agent_name="Jerry",
        description="",
        enabled_tools=['agent_search', 'list_agents', 'web_search'],
        api_port=8028,
        log_path=Path("logs"),
        log_level=os.getenv("AGENT_LOG_LEVEL", "DEBUG"),
        reasoning_effort="low"
    )
    
    agent = MemeticAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        chroma_client=chroma_client,
        config=config
    )
    
    return agent
