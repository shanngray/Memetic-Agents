import os
import sys
import shutil
from pathlib import Path
from typing import Dict
import inquirer
from chromadb import PersistentClient
import json

sys.path.append(str(Path(__file__).parents[1]))

from base_agent.config import AgentConfig
from memetic_agent.memetic_agent import MemeticAgent


def create_agent_folder_structure(agent_name: str) -> Path:
    """Create the required folder structure for a new agent."""
    base_path = Path("agent_files") / agent_name
    prompt_modules_path = base_path / "prompt_modules"
    scores_path = prompt_modules_path / "scores"
    
    # Create directories
    prompt_modules_path.mkdir(parents=True, exist_ok=True)
    scores_path.mkdir(parents=True, exist_ok=True)
    
    # Copy base prompt templates
    base_prompts_path = Path("src/system_prompt_templates")
    for prompt_file in base_prompts_path.glob("*.md"):
        shutil.copy2(prompt_file, prompt_modules_path / prompt_file.name)

    base_schemas_path = Path("src/system_prompt_templates/schemas")
    for schema_file in base_schemas_path.glob("*.json"):
        shutil.copy2(schema_file, prompt_modules_path / schema_file.name)
    
    # Write confidence scores to new location
    confidence_scores_path = scores_path / "prompt_confidence_scores.json"
    confidence_scores_path.write_text(json.dumps(default_scores, indent=4), encoding="utf-8")
    
    return base_path

def get_agent_config_from_user() -> Dict:
    """Interactive questionnaire to gather agent configuration."""
    questions = [
        inquirer.Text('agent_name', message="What is the name of your agent?"),
        inquirer.Text('description', message="Provide a brief description of your agent"),
        inquirer.Text('model', message="Which model should the agent use?", default="gpt-4o-mini"),
        inquirer.List('console_logging', 
                     message="Enable console logging?",
                     choices=['True', 'False'],
                     default='True'),
        inquirer.Text('api_port', 
                     message="Which API port should the agent use?",
                     validate=lambda _, x: x.isdigit()),
        inquirer.Text('temperature',
                     message="Set the temperature (0.0-2.0)",
                     validate=lambda _, x: 0 <= float(x) <= 2,
                     default="1.0"),
        inquirer.Checkbox('enabled_tools',
                         message="Select enabled tools",
                         choices=['agent_search', 'list_agents', 'web_search'],
                         default=['agent_search', 'list_agents'])
    ]
    
    answers = inquirer.prompt(questions)
    return answers

def create_new_agent() -> None:
    """Create a new agent based on user input and generate its creation script."""
    # Get configuration from user
    config_data = get_agent_config_from_user()
    
    # Create folder structure
    agent_path = create_agent_folder_structure(config_data['agent_name'])
    
    # Generate the agent creation script
    script_content = f'''import os
from pathlib import Path
from src.memetic_agent.memetic_agent import MemeticAgent
from src.base_agent.config import AgentConfig
from chromadb import PersistentClient

def create_{config_data["agent_name"].lower()}_agent(chroma_client: PersistentClient) -> MemeticAgent:
    """Create an MemeticAgent instance."""
    
    # Load system prompt from file
    system_prompt_path = Path("agent_files/{config_data["agent_name"]}/prompt_modules/sys_prompt.md")
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    
    config = AgentConfig(
        debug=os.getenv("AGENT_DEBUG", "false").lower() == "true",
        console_logging={config_data["console_logging"]},
        model="{config_data["model"]}",
        temperature={config_data["temperature"]},
        system_prompt=system_prompt,
        agent_name="{config_data["agent_name"]}",
        description="{config_data["description"]}",
        enabled_tools={config_data["enabled_tools"]},
        api_port={config_data["api_port"]},
        log_path=Path("logs"),
        log_level=os.getenv("AGENT_LOG_LEVEL", "DEBUG")
    )
    
    agent = MemeticAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        chroma_client=chroma_client,
        config=config
    )
    
    return agent
'''
    
    # Write the creation script to a new file
    script_path = Path(f"src/agents/create_{config_data['agent_name'].lower()}_agent.py")
    script_path.write_text(script_content)
    
    print(f"\nAgent {config_data['agent_name']} created successfully!")
    print(f"You can find the agent's prompt files in: {agent_path}")
    print(f"Agent creation script generated at: {script_path}")

if __name__ == "__main__":
    create_new_agent()