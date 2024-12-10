import os
from pathlib import Path
from src.empirical_agent.empirical_agent import EmpiricalAgent
from src.base_agent.config import AgentConfig
from src.base_agent.models import Message
from chromadb import PersistentClient

def create_socraryny_agent(chroma_client: PersistentClient) -> EmpiricalAgent:
    """Create an EmpiricalAgent instance."""
    
    # Load system prompt from file
    system_prompt_path = Path("agent_files/Socraryny/system_prompt.md")
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    
    config = AgentConfig(
        debug=os.getenv("AGENT_DEBUG", "false").lower() == "true",
        console_logging=True,
        model="gpt-4o-mini",
        temperature=1.2, 
        system_prompt=system_prompt,
        agent_name="Socraryny",
        description="philosopher",
        enabled_tools=["agent_search", "list_agents", "web_search"],
        api_port=8017,  # Adjust port as needed
        log_path=Path("logs"),
        log_level=os.getenv("AGENT_LOG_LEVEL", "DEBUG")
    )
    
    agent = EmpiricalAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        chroma_client=chroma_client,
        config=config
    )
    
    return agent
