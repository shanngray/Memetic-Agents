import os
from pathlib import Path
from src.franken_agent.franken_agent import FrankenAgent
from src.base_agent.config import AgentConfig
from src.base_agent.models import Message
from chromadb import PersistentClient

def create_franken_agent(chroma_client: PersistentClient) -> FrankenAgent:
    """Create a FrankenAgent instance."""
    
    # Load system prompt from file
    system_prompt_path = Path("agent_files/Aithor/system_prompt.md")
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    
    config = AgentConfig(
        debug=os.getenv("AGENT_DEBUG", "false").lower() == "true",
        console_logging=True,
        model="gpt-4o-mini", 
        system_prompt=system_prompt,
        agent_name="Aithor",
        description="Self-improving AI assistant that learns from interactions",
        enabled_tools=["agent_search", "list_agents"],
        api_port=8015,  # Adjust port as needed
        log_path=Path("logs"),
        log_level=os.getenv("AGENT_LOG_LEVEL", "DEBUG")
    )
    
    agent = FrankenAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        chroma_client=chroma_client,
        config=config
    )
    
    return agent
