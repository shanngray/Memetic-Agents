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
from typing import List, Dict
import uuid
# Add project root to path
sys.path.append(str(Path(__file__).parents[2]))

from base_agent.base_agent import BaseAgent
from base_agent.config import AgentConfig
from base_agent.models import Message

#TODO: Empirical agent will have a series of modules that make up its system prompt. Some modules will be updated sub consciously via memory and others it will
# have conscious control over.
#It will also be able to create new modules.

class EmpiricalAgent(BaseAgent):
    def __init__(self, api_key: str, chroma_client: PersistentClient, config: AgentConfig = None):
        """Initialize EmpiricalAgent with reasoning capabilities."""
        super().__init__(api_key=api_key, chroma_client=chroma_client, config=config)
        
        self.system_path = self.files_path / "system_prompt.md"
        
        # Initialize reasoning module path
        self.reasoning_path = Path("agent_files") / self.config.agent_name / "reasoning"
        self.reasoning_path.mkdir(parents=True, exist_ok=True)
        self.reasoning_module_file = self.reasoning_path / "reasoning_module.md"
        
        # Load reasoning module
        self.reasoning_module = self._load_reasoning_module()
        
        # Add reasoning-related tool to internal_tools dictionary
        self.internal_tools.update({
            "update_reasoning_module": self.update_reasoning_module,
            "update_system_prompt": self.update_system_prompt
        })

        # Register the new internal tools
        for tool_name, tool_func in {
            "update_reasoning_module": self.update_reasoning_module,
            "update_system_prompt": self.update_system_prompt
        }.items():
            self.register_tool(tool_func)

        # Update system prompt with reasoning module
        self._initialize_system_prompt()

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

    def _load_reasoning_module(self) -> str:
        """Load the reasoning module content."""
        try:
            with FileLock(f"{self.reasoning_module_file}.lock"):
                return self.reasoning_module_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            default_reasoning = "Default reasoning process:\n1. Analyze the input\n2. Break down the problem\n3. Formulate response"
            self.reasoning_module_file.write_text(default_reasoning, encoding="utf-8")
            return default_reasoning

    async def update_reasoning_module(self, new_reasoning: str) -> str:
        """Update the reasoning module that guides your thinking process."""
        try:
            with FileLock(f"{self.reasoning_module_file}.lock"):
                self.reasoning_module_file.write_text(new_reasoning, encoding="utf-8")
            self.reasoning_module = new_reasoning
            
            # Update system prompt with new reasoning module
            await self._update_system_with_reasoning()
            return "Reasoning module updated successfully"
            
        except Exception as e:
            raise ValueError(f"Failed to update reasoning module: {str(e)}")

    async def _update_system_with_reasoning(self) -> None:
        """Update system message with current reasoning module."""
        if self.current_conversation_id in self.conversations:
            system_message = next(
                (msg for msg in self.conversations[self.current_conversation_id] 
                 if msg.role == "system"),
                None
            )
            
            new_content = f"{self.config.system_prompt}\n\nReasoning Module:\n{self.reasoning_module}"
            
            if system_message:
                system_message.content = new_content
            else:
                self.conversations[self.current_conversation_id].insert(0, Message(
                    role="system",
                    content=new_content
                ))

    def _initialize_system_prompt(self) -> None:
        """Initialize system prompt with tools and reasoning module."""
        # Get tool descriptions
        tools_desc = self._get_tool_descriptions()
        
        # Build complete system prompt
        system_prompt = (
            f"{self.config.system_prompt}\n\n"
            f"Available tools:\n{tools_desc}\n\n"
            f"Reasoning Module:\n{self.reasoning_module}"
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

    async def _transfer_to_long_term(self, days_threshold: int = 0) -> None:
        """Transfer short-term memories into long-term storage as atomic memories with SPO metadata.
        
        Args:
            days_threshold: Number of days worth of memories to keep in short-term. 
                           Memories older than this will be processed into long-term storage.
                           Default is 0 (process all memories).
        """
        try:
            log_event(self.logger, "agent.memorising", "Beginning atomic memory extraction process")
            
            # Retrieve recent memories from short-term storage
            short_term_memories = await self.memory.retrieve(
                query="",  # Empty query to get all memories
                collection_names=["short_term"],
                n_results=100
            )
            
            # Filter memories based on threshold
            threshold_date = datetime.now() - timedelta(days=days_threshold)
            short_term_memories = [
                memory for memory in short_term_memories
                if datetime.fromisoformat(memory["metadata"].get("timestamp", "")) < threshold_date
            ]
            
            if not short_term_memories:
                log_event(self.logger, "memory.error", "No recent memories found for atomization")
                await self.set_status(self._previous_status)
                return

            # Process each memory individually
            for memory in short_term_memories:
                try:
                    # Extract atomic memories using LLM
                    atomic_response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": """Extract atomic memories from this interaction. Each atomic memory should be a simple, 
                            factual statement that can be represented in Subject-Predicate-Object format. Extract as many atomic memories as necessary.

                            Format the response as a JSON array of objects, where each object has:
                            {
                                "statement": "The complete factual statement",
                                "subject": "The entity performing the action or being described",
                                "predicate": "The action or relationship",
                                "object": "The target of the action or description",
                                "confidence": A number between 0 and 1 indicating certainty,
                                "category": One of ["fact", "preference", "capability", "relationship", "belief", "goal"],
                                "metatags": A list of semantic #tags that describe the memory and can link it to other memories,
                                "thoughts": Your thoughts on the memory and its usefulness.
                            }

                            Example:
                            {
                                "statement": "Alice likes to code in Python",
                                "subject": "Alice",
                                "predicate": "likes coding in",
                                "object": "Python",
                                "confidence": 0.9,
                                "category": "preference",
                                "metatags": ["#coding", "#python", "#alice"],
                                "thoughts": "Alice might be able to help me with my coding problems."
                            }"""},
                            {"role": "user", "content": f"Memory content:\n{memory['content']}"}
                        ],
                        response_format={ "type": "json_object" }
                    )
                    
                    atomic_memories = json.loads(atomic_response.choices[0].message.content)
                    
                    # Store each atomic memory
                    for atomic in atomic_memories:
                        metadata = {
                            "memory_id": str(uuid.uuid4()),
                            "original_timestamp": memory["metadata"].get("timestamp"),
                            "source_type": "atomic_memory",
                            "subject": atomic["subject"],
                            "predicate": atomic["predicate"],
                            "object": atomic["object"],
                            "confidence": atomic["confidence"],
                            "category": atomic["category"],
                            "timestamp": datetime.now().isoformat()
                        }

                        await self.memory.store(
                            content=atomic["statement"],
                            collection_name="long_term",
                            metadata=metadata
                        )
                        
                        log_event(self.logger, "memory.atomic.stored",
                                 f"Stored atomic memory: {atomic['statement']} ({atomic['category']})")

                    # Save to disk for debugging/backup
                    await self._save_atomic_memory_to_disk(atomic_memories, memory["metadata"])

                except Exception as e:
                    log_error(self.logger, f"Failed to process memory into atomic form: {str(e)}", exc_info=e)
                    continue
            
            await self._cleanup_short_term_memories()
            
            log_event(self.logger, "memory.atomization.complete",
                     f"Completed memory atomization for {len(short_term_memories)} memories")
             
        except Exception as e:
            log_error(self.logger, "Failed to atomize memories", exc_info=e)
        finally:
            if self.status != AgentStatus.SHUTTING_DOWN:
                await self.set_status(self._previous_status)

    async def _save_atomic_memory_to_disk(self, atomic_memories: List[Dict], original_metadata: Dict) -> None:
        """Save atomic memories to disk for debugging/backup."""
        try:
            memory_dump_path = self.files_path / f"{self.config.agent_name}_atomic_memory_dump.md"
            with FileLock(f"{memory_dump_path}.lock"):
                with open(memory_dump_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n## Atomic Memories Generated {datetime.now().isoformat()}\n")
                    f.write("### Original Memory Metadata\n")
                    for key, value in original_metadata.items():
                        f.write(f"{key}: {value}\n")
                    
                    f.write("\n### Extracted Atomic Memories\n")
                    for atomic in atomic_memories:
                        f.write(f"\n#### {atomic['category'].title()} (confidence: {atomic['confidence']})\n")
                        f.write(f"Statement: {atomic['statement']}\n")
                        f.write(f"Subject: {atomic['subject']}\n")
                        f.write(f"Predicate: {atomic['predicate']}\n")
                        f.write(f"Object: {atomic['object']}\n")
                    f.write("\n---\n")
                
            log_event(self.logger, "memory.dumped", 
                     f"Saved {len(atomic_memories)} atomic memories to disk")
        except Exception as e:
            log_error(self.logger, 
                     f"Failed to save atomic memories to disk: {str(e)}")