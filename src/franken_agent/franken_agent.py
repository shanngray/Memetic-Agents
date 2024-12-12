import sys
import json
import re
import inspect
from pathlib import Path
from filelock import FileLock
from datetime import datetime
from functools import lru_cache
import importlib
from typing import Callable
from filelock import FileLock
from chromadb import PersistentClient

# Add project root to path
sys.path.append(str(Path(__file__).parents[2]))

from base_agent.base_agent import BaseAgent
from base_agent.config import AgentConfig
from base_agent.models import Message

class MemorisedTechniqueError(Exception):
    """Raised when the memorised technique file is not found."""
    pass

class FrankenAgent(BaseAgent):
    def __init__(self, api_key: str, chroma_client: PersistentClient, config: AgentConfig = None):
        """Initialize FrankenAgent with learning capabilities."""
        super().__init__(api_key=api_key, chroma_client=chroma_client, config=config)
        
        self.system_path = self.files_path / "system_prompt.md"

        # Initialize technique-related paths
        self.techniques_path = Path("agent_files") / self.config.agent_name / "techniques"
        self.techniques_path.mkdir(parents=True, exist_ok=True)
        self.memorised_path = self.techniques_path / "memorised.txt"
        
        # Load techniques and memorised technique
        self.techniques = self._load_techniques()
        try:
            self.memorised = self._load_memorised()
        except MemorisedTechniqueError:
            # Set default technique if none memorised
            self.memorised = "default"
            self._save_memorised("default")
            
        # Add technique-related tools to internal_tools dictionary
        self.internal_tools.update({
            "add_technique": self.add_technique,
            "apply_technique": self.apply_technique,
            "update_system_prompt": self.update_system_prompt
        })

        # Register the new internal tools
        for tool_name, tool_func in {
            "add_technique": self.add_technique,
            "apply_technique": self.apply_technique,
            "update_system_prompt": self.update_system_prompt
        }.items():
            self.register_tool(tool_func)

        # Update system prompt with technique information
        self._initialize_system_prompt()

    @lru_cache(maxsize=3)
    def _get_technique_content(self, technique_name: str) -> str:
        """Load and cache technique content."""
        technique_path = self.techniques_path / f"{technique_name}.md"
        try:
            return technique_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return "No technique content found"

    def _load_memorised(self) -> str:
        """Load the memorised technique name."""
        try:
            with FileLock(f"{self.memorised_path}.lock"):
                memorised = self.memorised_path.read_text(encoding="utf-8").strip()
                return memorised.replace(".md", "")
        except FileNotFoundError:
            self.logger.debug("No memorised.txt found")
            raise MemorisedTechniqueError(
                "memorised.txt file not found. Please create the file with a default technique."
            )

    def _save_memorised(self, technique_name: str) -> None:
        """Save the current memorised technique name."""
        self.logger.debug(f"Saving new memorised technique: {technique_name}")
        technique_name = technique_name.replace('.md', '')
        with FileLock(f"{self.memorised_path}.lock"):
            self.memorised_path.write_text(technique_name, encoding="utf-8")

    def _load_techniques(self) -> dict:
        """Load all available techniques."""
        techniques = {}
        for file in self.techniques_path.glob("*.md"):
            name = file.stem
            try:
                content = file.read_text(encoding="utf-8")
                techniques[name] = content
            except Exception as e:
                self.logger.debug(f"Error loading technique {name}: {e}")
                
        self.techniques = techniques
        return techniques

    async def add_technique(self, name: str, instructions: str) -> str:
        """When you come across a new prompt engineering technique, use this tool to add it to your repertoire of reasoning techniques."""
        try:
            file_path = self.techniques_path / f"{name}.md"
            with FileLock(f"{file_path}.lock"):
                file_path.write_text(instructions, encoding="utf-8")
                
            # Add to techniques dictionary.
            self.techniques[name] = instructions
            
            # Apply technique
            await self.apply_technique(name)

            return f"Added new technique: {name}"
            
        except Exception as e:
            raise ValueError(f"Failed to add technique: {str(e)}")

    async def apply_technique(self, technique_name: str) -> str:
        """If you are having trouble solving a problem, use this tool to switch to using a different reasoning technique."""
        self.logger.debug(f"Attempting to apply technique: {technique_name}")
        
        if technique_name in self.techniques:
            # Update memorised technique
            self._save_memorised(technique_name)
            self.memorised = technique_name
            
            # Get technique content directly from techniques dict
            technique_content = self.techniques[technique_name]
            self.logger.debug(f"Loaded technique content: {technique_content[:100]}...")  # First 100 chars
            
            # Update system message in current conversation
            if self.current_conversation_id in self.conversations:
                system_message = next(
                    (msg for msg in self.conversations[self.current_conversation_id] 
                     if msg.role == "system"),
                    None
                )
                
                new_content = f"{self.config.system_prompt}\n\nCurrent active technique:\n{technique_content}"
                
                if system_message:
                    self.logger.debug("Updating existing system message")
                    system_message.content = new_content
                else:
                    self.logger.debug("Creating new system message")
                    self.conversations[self.current_conversation_id].insert(0, Message(
                        role="system",
                        content=new_content
                    ))
                
                self.logger.debug(f"System message updated successfully for technique: {technique_name}")
                return f"Applied {technique_name} technique"
                
                self.logger.warning(f"Current conversation not found: {self.current_conversation_id}")
            return f"Failed to apply technique: current conversation not found"
            
        self.logger.warning(f"Unknown technique requested: {technique_name}")
        return f"Unknown technique: {technique_name}"

    def _get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all available tools."""
        tool_descriptions = []
        
        for tool_name in self.config.enabled_tools:
            try:
                # Import the tool module
                if str(self.config.tools_path) not in sys.path:
                    sys.path.append(str(self.config.tools_path))
                    
                module = importlib.import_module(tool_name)
                
                # Get the main function's docstring
                main_func = getattr(module, tool_name)
                description = main_func.__doc__ or "No description available"
                
                tool_descriptions.append(f"- {tool_name}: {description.strip()}\n")
                
            except Exception as e:
                self.logger.error(f"Error loading tool description for {tool_name}: {str(e)}")
                continue

        for tool_name, tool_func in self.internal_tools.items():
            doc = (inspect.getdoc(tool_func) or "").split("\n")[0]
            tool_descriptions.append(f"- {tool_name}: {doc.strip()}\n")

        return "\n".join(tool_descriptions)

    def _initialize_system_prompt(self) -> None:
        """Initialize system prompt with tools and technique information."""
        # Get tool descriptions
        tools_desc = self._get_tool_descriptions()
        
        # Get technique descriptions
        techniques_desc = "\n".join([
            f"- {name}: {content.split('\n')[0]}\n" 
            for name, content in self.techniques.items()
        ])

        # Load current technique content
        technique_content = self._get_technique_content(self.memorised)
        
        self.logger.debug(f"Initializing system prompt with technique: {self.memorised}")
        
        # Build complete system prompt
        system_prompt = (
            f"{self.config.system_prompt}\n\n"
            f"Available tools:\n{tools_desc}\n\n"
            f"Available reasoning techniques (can be applied with the apply_technique tool):\n{techniques_desc}\n\n"
            f"Current active technique:\n{technique_content}"
        )
        self.config.system_prompt = system_prompt
        
        self.logger.info(f"System prompt:\n\n {system_prompt}\n\n")

        # Update system message in default conversation
        if self.current_conversation_id in self.conversations:
            system_message = next(
                (msg for msg in self.conversations[self.current_conversation_id] 
                 if msg.role == "system"),
                None
            )
            if system_message:
                system_message.content = system_prompt
            else:
                self.conversations[self.current_conversation_id].insert(0, Message(
                    role="system",
                    content=system_prompt
                ))
        

    async def update_system_prompt(self, new_prompt: str) -> str:
        """Update your system prompt when you want to modify your core behavior.
        Use this tool to permanently change how you operate."""
        with FileLock(f"{self.system_path}.lock"):
            self.system_path.write_text(new_prompt, encoding="utf-8")
        return "System prompt updated successfully"