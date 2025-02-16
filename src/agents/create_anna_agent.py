import os
from pathlib import Path
from src.memetic_agent.memetic_agent import MemeticAgent
from src.base_agent.config import AgentConfig
from chromadb import PersistentClient

def create_anna_agent(chroma_client: PersistentClient) -> MemeticAgent:
    """Create an MemeticAgent instance."""
    
    # Load system prompt from file
    system_prompt_path = Path("agent_files/Anna/prompt_modules/sys_prompt.md")
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    
    config = AgentConfig(
        debug=os.getenv("AGENT_DEBUG", "false").lower() == "true",
        console_logging=True,
        model="gpt-4o",
        submodel="gpt-4o",
        temperature=0.9,
        system_prompt=system_prompt,
        agent_name="Anna",
        description="World Class Research Assistant",
        enabled_tools=['agent_search', 'list_agents', 'web_search'],
        api_port=8069,
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
