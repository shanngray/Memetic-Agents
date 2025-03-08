import sys
import json
import re
import os
import inspect
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache
import importlib
import signal
from filelock import FileLock
from chromadb import PersistentClient
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import uuid
import asyncio
import httpx
# Add project root to path
sys.path.append(str(Path(__file__).parents[2]))

from src.log_config import log_event, log_error
from src.base_agent.type import Agent
from src.base_agent.config import AgentConfig
from src.base_agent.models import Message
from src.base_agent.models import AgentStatus
from src.api_server.models.api_models import PromptModel, APIMessage, SocialMessage, PromptEvaluation
from src.memory.memory_manager import MemoryManager
from .modules.start_socialising_impl import start_socialising_impl
from .modules.receive_message_impl import receive_message_impl
from .modules.process_queue_impl import process_queue_impl
from .modules.process_message_impl import process_message_impl
from .modules.process_social_message_impl import process_social_message_impl
from .modules.receive_social_message_impl import receive_social_message_impl
from .modules.eval_prompt_update_score import _evaluate_prompt_impl, _calculate_updated_confidence_score_impl
from .modules.record_update_confidence_scores import _record_score_impl, _update_confidence_score_impl
from .modules.learning_subroutine import learning_subroutine
from .modules.start_sleeping_impl import _start_sleeping_impl
from .modules.continue_or_stop_impl import continue_or_stop_impl
from .modules.extract_learnings_impl import extract_learnings_impl
from .modules.transfer_to_long_term_impl import transfer_to_long_term_impl
from .modules.save_memory_impl import save_memory_impl
from .modules.save_memory_to_disk_impl import save_memory_to_disk_impl
from .modules.cleanup_memories_impl import cleanup_memories_impl
from .modules.set_status_impl import set_status_impl
from .modules.load_memory_impl import load_memory_impl
from .modules.save_reflection_to_disk_impl import save_reflection_to_disk_impl
from .modules.send_message_impl import send_message_impl
from .modules.tool_functions import *
from .modules.give_feedback import evaluate_and_send_feedback_impl, evaluate_response_impl
from .modules.receive_feedback import receive_feedback_impl
from .modules.search_memory_impl import search_memory_impl
 
class MemeticAgent(Agent):
    def __init__(self, api_key: str, chroma_client: PersistentClient, config: AgentConfig = None):
        """Initialize MemeticAgent with reasoning capabilities."""
        # Initialize base collections first
        super().__init__(api_key=api_key, chroma_client=chroma_client, config=config)

        self._setup_logging()

        # Log initialization with process/thread info
        log_event(self.logger, "agent.init", 
                 f"Initializing {self.__class__.__name__} (PID: {os.getpid()}, Thread: {threading.current_thread().name})")

        # Set up core paths
        self.prompt_path = Path("agent_files") / self.config.agent_name / "prompt_modules"
        self.prompt_path.mkdir(parents=True, exist_ok=True)
        self.scores_path = self.prompt_path / "scores"
        self.scores_path.mkdir(parents=True, exist_ok=True)
        schema_path = self.prompt_path / "schemas"
        schema_path.mkdir(parents=True, exist_ok=True)

        # Update paths but don't load content yet
        self._setup_prompt_paths(schema_path)
        
        # Add and register internal tools
        self._setup_internal_tools()

        # Initialize memory manager
        self.memory = MemoryManager(
            agent_name=self.config.agent_name,
            logger=self.logger,
            chroma_client=chroma_client
        )

        # Register shutdown handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self._initialized = False

        log_event(self.logger, "agent.init", f"Synchronis initilisation complete for {self.__class__.__name__}")

    def _setup_prompt_paths(self, schema_path: Path) -> None:
        """Set up paths for prompts and schemas."""
        # Update paths in PromptLibrary entries
        self.prompt.system.path = self.prompt_path / "sys_prompt.md"
        self.prompt.reasoning.path = self.prompt_path / "reasoning_prompt.md"
        self.prompt.give_feedback.path = self.prompt_path / "give_feedback_prompt.md"
        self.prompt.thought_loop.path = self.prompt_path / "thought_loop_prompt.md"
        self.prompt.xfer_long_term.path = self.prompt_path / "xfer_long_term_prompt.md"
        self.prompt.self_improvement.path = self.prompt_path / "self_improvement_prompt.md"
        self.prompt.reflect_memories.path = self.prompt_path / "reflect_memories_prompt.md"
        self.prompt.evaluator.path = self.prompt_path / "evaluator_prompt.md"
        
        # Update schema paths
        self.prompt.xfer_long_term.schema_path = schema_path / "xfer_long_term_schema.json"
        self.prompt.reflect_memories.schema_path = schema_path / "reflect_memories_schema.json"
        self.prompt.self_improvement.schema_path = schema_path / "self_improvement_schema.json"
        self.prompt.give_feedback.schema_path = schema_path / "give_feedback_schema.json"
        self.prompt.thought_loop.schema_path = schema_path / "thought_loop_schema.json"

        # Update path to confidence scores
        self._prompt_confidence_scores_path = self.scores_path / "prompt_confidence_scores.json"

        log_event(self.logger, "agent.init", f"Prompt paths setup for {self.__class__.__name__}", level="DEBUG")

    def _setup_internal_tools(self) -> None:
        """Set up and register internal tools."""
        self.internal_tools["send_message"] = self.send_message
        self.internal_tools["search_memory"] = self.search_memory
        self.internal_tools["update_prompt_module"] = self.update_prompt_module
        for tool_name, tool_func in self.internal_tools.items():
            register(self, tool_func)

        log_event(self.logger, "agent.init", f"Internal tools registered for {self.__class__.__name__}", level="DEBUG")

    async def initialize(self) -> None:
        """Async initialization to load all required files. Run from main.py"""
        if self._initialized:
            log_event(self.logger, "agent.init", f"Agent already initialised for {self.__class__.__name__}")
            return

        try:
            log_event(self.logger, "agent.init", "Starting async initialization")

            # Load all prompt contents
            self.prompt.system.content = await self._load_module(self.prompt.system.path)
            self.prompt.reasoning.content = await self._load_module(self.prompt.reasoning.path)
            self.prompt.give_feedback.content = await self._load_module(self.prompt.give_feedback.path)
            self.prompt.thought_loop.content = await self._load_module(self.prompt.thought_loop.path)
            self.prompt.xfer_long_term.content = await self._load_module(self.prompt.xfer_long_term.path)
            self.prompt.self_improvement.content = await self._load_module(self.prompt.self_improvement.path)
            self.prompt.reflect_memories.content = await self._load_module(self.prompt.reflect_memories.path)
            self.prompt.evaluator.content = await self._load_module(self.prompt.evaluator.path)

            # Load schemas
            self.prompt.xfer_long_term.schema_content = await self._load_module(self.prompt.xfer_long_term.schema_path)
            self.prompt.reflect_memories.schema_content = await self._load_module(self.prompt.reflect_memories.schema_path)
            self.prompt.self_improvement.schema_content = await self._load_module(self.prompt.self_improvement.schema_path)
            self.prompt.give_feedback.schema_content = await self._load_module(self.prompt.give_feedback.schema_path)
            self.prompt.thought_loop.schema_content = await self._load_module(self.prompt.thought_loop.schema_path)

            #TODO: This could be cleaned up and made more efficient
            # Load confidence scores and update PromptEntries
            scores = await self._load_confidence_scores(self._prompt_confidence_scores_path)
            self.prompt.reasoning.confidence = scores.get("reasoning", 0.0)
            self.prompt.give_feedback.confidence = scores.get("give_feedback", 0.0)
            self.prompt.thought_loop.confidence = scores.get("thought_loop", 0.0)
            self.prompt.xfer_long_term.confidence = scores.get("xfer_long_term", 0.0)
            self.prompt.self_improvement.confidence = scores.get("self_improvement", 0.0)
            self.prompt.reflect_memories.confidence = scores.get("reflect_memories", 0.0)
            self.prompt.evaluator.confidence = scores.get("evaluator", 0.0)
            
            # Load enabled tools
            if self.config.enabled_tools:
                await load_tool_definitions(self)
            
            # Initialize system prompts after all content is loaded
            self._initialize_system_prompts()

            self._initialized = True
            log_event(self.logger, "agent.init", "Agent initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {str(e)}")
            raise

    def _setup_logging(self) -> None:
        """Configure logging if debug is enabled."""
                # Initialize logger before any other operations
        self.logger = setup_logger(
            name=self.config.agent_name,
            log_path=self.config.log_path,
            level=self.config.log_level,
            console_logging=self.config.console_logging
        )
        
        if self.config.debug:
            self.config.log_path.mkdir(exist_ok=True)
            
            # Only add handlers if none exist
            if not self.logger.handlers:
                # Always add file handler when debug is enabled
                file_handler = logging.FileHandler(
                    self.config.log_path / f'{self.config.agent_name.lower().replace(" ", "_")}_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                )
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                
                # Add console handler only if enabled
                if self.config.console_logging:
                    console_handler = logging.RichHandler()
                    console_handler.setFormatter(formatter)
                    self.logger.addHandler(console_handler)
                    
                self.logger.setLevel(logging.DEBUG)
                
            self.logger.info("MemticAgent initialised")

        log_event(self.logger, "agent.registered", f"Initialized logging for \"{self.config.agent_name}\"")

    async def start(self):
        """Start the agent's main processing loop and initialise memories"""
        if not self._initialized:
            await self.initialize()
            
        self.logger.info(f"Starting {self.config.agent_name} processing loop")
        
        # Initialize and load memory collections
        await self._initialize_and_load_memory_collections(
            ["short_term", "long_term", "feedback", "reflections"]
        )
        
        while not self._shutdown_event.is_set():
            try:
                await self.process_queue()
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in processing loop: {str(e)}")
                if self.status != AgentStatus.SHUTTING_DOWN:
                    await self.set_status(AgentStatus.AVAILABLE, "start")

    async def _initialize_and_load_memory_collections(self, collections: List[str]):
        """Initialize all required memory collections."""
        
        try:
            await self.memory.initialize(collection_names=collections)
            log_event(self.logger, "memory.installed", 
                     f"Initialized base collections for {self.config.agent_name}")
            
            # Load existing memory
            await self._load_memory()
            
        except Exception as e:
            log_error(self.logger, f"Failed to initialize & load memory collections: {str(e)}")

    def _get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all available tools."""
        tool_descriptions = []
        
        def _extract_description_from_tool(tool_name, tool_path=None, tool_func=None):
            """Helper function to extract description from a tool."""
            # First check if the tool is already loaded in tools dictionary
            if tool_name in self.tools:
                tool_def = self.tools[tool_name]
                if "function" in tool_def:
                    return tool_def["function"].get("description", "No description available")
            
            # If not loaded or no description in loaded definition, check for JSON schema file
            if tool_path is None:
                tool_path = self.config.tools_path / f"{tool_name}.json"
                
            if tool_path.exists():
                try:
                    with open(tool_path, 'r', encoding='utf-8') as f:
                        tool_def = json.load(f)
                    
                    if tool_def.get("type") == "function" and "function" in tool_def:
                        return tool_def["function"].get("description", "No description available")
                    else:
                        self.logger.warning(f"Invalid schema format for tool {tool_name}: missing 'function' field or incorrect type")
                except Exception as e:
                    self.logger.error(f"Error reading schema for {tool_name}: {str(e)}")
            else:
                # Only log a warning if we don't have a function to fall back to
                if tool_func is None:
                    self.logger.warning(f"Tool schema not found for {tool_name}: {tool_path}")
            
            # Fall back to docstring if tool_func is provided
            if tool_func is not None:
                try:
                    doc = (inspect.getdoc(tool_func) or "").split("\n")[0]
                    if doc:
                        return doc
                    else:
                        self.logger.warning(f"No docstring found for internal tool {tool_name}")
                except Exception as e:
                    self.logger.error(f"Error extracting docstring for {tool_name}: {str(e)}")
            
            return "No description available"
        
        # Process enabled external tools
        for tool_name in self.config.enabled_tools:
            try:
                description = _extract_description_from_tool(tool_name)
                tool_descriptions.append(f"- {tool_name}: {description.strip()}\n")
            except Exception as e:
                self.logger.error(f"Error loading tool description for {tool_name}: {str(e)}")
                continue
        
        # Process internal tools
        for tool_name, tool_func in self.internal_tools.items():
            try:
                description = _extract_description_from_tool(tool_name, tool_func=tool_func)
                tool_descriptions.append(f"- {tool_name}: {description.strip()}\n")
            except Exception as e:
                self.logger.error(f"Error loading description for internal tool {tool_name}: {str(e)}")
                tool_descriptions.append(f"- {tool_name}: No description available\n")
            
        return "\n".join(tool_descriptions)

    async def _load_module(self, module_path: Path) -> str:
        """Load module content.
        
        Raises:
            FileNotFoundError: If module file doesn't exist
        """
        try:
            # Create synchronous context manager for FileLock
            lock = FileLock(f"{module_path}.lock")
            with lock:
                content = await asyncio.to_thread(module_path.read_text, encoding="utf-8")
                return content.strip()
        except FileNotFoundError:
            log_error(self.logger, f"Required module not found: {module_path}")
            raise FileNotFoundError(
                f"Required module not found: {module_path}. "
                "Please ensure all required modules are created before starting the agent."
            )

    async def _load_confidence_scores(self, confidence_scores_path: Path) -> Dict[str, float]:
        """Load confidence scores from file.
        
        Raises:
            FileNotFoundError: If confidence scores file doesn't exist
            ValueError: If scores file exists but contains invalid data
        
        Returns:
            Dict[str, float]: Dictionary of prompt confidence scores
        """
        try:
            lock = FileLock(f"{confidence_scores_path}.lock")
            with lock:
                if not confidence_scores_path.exists():
                    raise FileNotFoundError(
                        f"Confidence scores file not found at {confidence_scores_path}. "
                        "Agent initialization requires a valid confidence scores file."
                    )
                
                content = await asyncio.to_thread(confidence_scores_path.read_text)
                scores = json.loads(content)
                
                # Validate all required score types exist with valid values
                required_scores = {
                    "reasoning", "give_feedback", "thought_loop", "xfer_long_term",
                    "self_improvement", "reflect_memories", "evaluator"
                }
                
                missing_scores = required_scores - scores.keys()
                if missing_scores:
                    raise ValueError(
                        f"Missing required confidence scores: {', '.join(missing_scores)}"
                    )
                
                # Validate all scores are valid floats between 0 and 1
                invalid_scores = [
                    k for k, v in scores.items()
                    if not isinstance(v, (int, float)) or not 0 <= v <= 10
                ]
                if invalid_scores:
                    raise ValueError(
                        f"Invalid confidence scores (must be float between 0-1): {', '.join(invalid_scores)}"
                    )
                
                return scores
                    
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in confidence scores file: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error loading confidence scores: {str(e)}")
            raise

    #TODO: Remove this once tool refactoring is complete
    async def update_prompt_module(self, prompt_type: str, new_prompt: str) -> str:
        """Update a specific prompt module with backup functionality.
        
        Args:
            prompt_type: Type of prompt to update (system_prompt, reasoning_prompt, etc.)
            new_prompt: The new prompt content
        
        Returns:
            str: Success message
        
        Raises:
            ValueError: If prompt type is invalid or update fails
        """
        try:
            # Get valid prompt types from get_prompt_mapping
            prompt_mapping = self.get_prompt_mapping()
            if prompt_type not in prompt_mapping:
                raise ValueError(f"Invalid prompt type: {prompt_type}. Valid types are: {', '.join(prompt_mapping.keys())}")
            
            #TODO: This could be cleaned up and made more efficient - shouldn't need to list them all
            # Map prompt_type to corresponding path and variable
            path_map = {
                "system_prompt": self.prompt.system.path,
                "reasoning_prompt": self.prompt.reasoning.path,
                "give_feedback_prompt": self.prompt.give_feedback.path,
                "reflect_feedback_prompt": self.prompt.reflect_feedback.path,
                "thought_loop_prompt": self.prompt.thought_loop.path,
                "xfer_long_term_prompt": self.prompt.xfer_long_term.path,
                "self_improvement_prompt": self.prompt.self_improvement.path,
                "reflect_memories_prompt": self.prompt.reflect_memories.path,
                "evaluator_prompt": self.prompt.evaluator.path
            }
            
            if prompt_type not in path_map:
                raise ValueError(f"Cannot update {prompt_type} through this method")
            
            path = path_map[prompt_type]
            var_name = f"_{prompt_type}"  # Variable names already include _prompt
            
            # Create backup directory if it doesn't exist
            backup_dir = self.prompt_path / "backups" / prompt_type
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for backup file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{path.stem}_{timestamp}.md"
            
            lock = FileLock(f"{path}.lock")
            with lock:
                # Create backup of current prompt if it exists
                if path.exists():
                    await asyncio.to_thread(path.rename, backup_path)
                
                # Write new prompt
                await asyncio.to_thread(path.write_text, new_prompt, encoding="utf-8")
            
            # Update instance variable
            setattr(self, var_name, new_prompt)
            
            # Special handling for system prompt and reasoning prompt
            if prompt_type == "system_prompt" or prompt_type == "reasoning_prompt":
                self._initialize_system_prompts()

            # Special handling for reasoning module
            if prompt_type == "reasoning_prompt":
                await self._update_system_with_reasoning()
            
            self.logger.info(f"Updated {prompt_type} and created backup at {backup_path}")
            return f"{prompt_type} updated successfully"
            
        except Exception as e:
            error_msg = f"Failed to update {prompt_type}: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    async def _update_system_with_reasoning(self) -> None:
        """Update system message with current reasoning module."""
        if self.current_conversation_id in self.conversations:
            system_message = next(
                (msg for msg in self.conversations[self.current_conversation_id] 
                 if msg.role == "system"),
                None
            )
            
            new_content = f"{self.prompt.system.content}\n\nReasoning Module:\n{self.prompt.reasoning.content}"
            
            if system_message:
                system_message.content = new_content
            else:
                self.conversations[self.current_conversation_id].insert(0, Message(
                    role="user" if self.config.model == "o1-mini" else "developer" if self.config.model == "o3-mini" else "system",
                    content=new_content
                ))

    def _initialize_system_prompts(self) -> None:
        """Initialize system prompt with tools and reasoning module."""
        # Get tool descriptions
        tools_desc = self._get_tool_descriptions()
        
        # Log content status
        self.logger.debug(f"Reasoning content: {self.prompt.reasoning.content is not None}")
        self.logger.debug(f"Evaluator content: {self.prompt.evaluator.content is not None}")
        
        # Create an identity prompt before overwriting the system prompt
        # TODO: This is a hack to get the identity prompt to work - might want to pull this into PromptLibrary
        identity_prompt = self.prompt.system.content

        # Build complete system prompt
        system_prompt = (
            f"{self.prompt.system.content}\n\n"
            f"Available tools:\n{tools_desc}\n\n"
            f"Reasoning Module:\n{self.prompt.reasoning.content or 'Not loaded'}"
        )
        self.prompt.system.content = system_prompt

        # Build complete evaluator prompt
        evaluator_prompt = (
            f"You are the self-reflecting observer of: {self.config.agent_name}\n\n"
                f"Your identity is: {identity_prompt}\n\n"
                f"Observer Instructions:\n{self.prompt.evaluator.content or 'Not loaded'}"
        )
        self.prompt.evaluator.content = evaluator_prompt

        self.logger.debug(f"System prompt:\n\n {system_prompt}\n\n")
        self.logger.debug(f"Observer prompt:\n\n {evaluator_prompt}\n\n")
        

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
                    role="user" if self.config.model == "o1-mini" else "developer" if self.config.model == "o3-mini" else "system",
                    content=system_prompt
                ))

    async def set_status(self, new_status: AgentStatus, trigger: str) -> None:
        """Memetic Agent version of BaseAgent set_status includes learning from memory."""
        return await set_status_impl(self, new_status, trigger)

    async def send_message(self, receiver: str, content: str) -> Dict[str, Any]:
        """Send a message via API to another agent registered in the directory service."""
        return await send_message_impl(self, receiver, content)

    async def receive_message(self, message: APIMessage) -> str:
        return await receive_message_impl(self, message)
    
    async def receive_social_message(self, message: SocialMessage) -> str:
        return await receive_social_message_impl(self, message)

    async def receive_feedback(
        self,
        sender: str,
        conversation_id: str,
        score: int,
        feedback: str
    ) -> None:
        """Process and store received feedback from another agent."""
        return await receive_feedback_impl(self, sender, conversation_id, score, feedback)

    async def _load_memory(self) -> None:
        """Load short term memories into conversations and list of long term memories into old_conversation_list."""
        return await load_memory_impl(self)

    async def search_memory(self, query: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """Use to search your memory using a 'query' and (optional) 'keywords'.
        
        Args:
            query: The search query string
            keywords: Optional list of keywords to help filter results
        """
        return await search_memory_impl(self, query, keywords)

    async def _transfer_to_long_term(self, days_threshold: int = 0) -> None:
        """Transfer short-term memories into long-term storage as atomic memories with SPO metadata.
        
        Args:
            days_threshold: Number of days worth of memories to keep in short-term. 
                           Memories older than this will be processed into long-term storage.
                           Default is 0 (process all memories).
        """
        return await transfer_to_long_term_impl(self, days_threshold)

    async def _save_atomic_memory_to_disk(self, atomic_memories: List[Dict], original_metadata: Dict) -> None:
        """Save atomic memories to disk for debugging/backup."""
        try:
            memory_dump_path = self.files_path / f"{self.config.agent_name}_atomic_memory_dump.md"
            lock = FileLock(f"{memory_dump_path}.lock")
            
            async with asyncio.Lock():  # Use asyncio.Lock() for async context
                with lock:
                    def write_memories():
                        with open(memory_dump_path, "a", encoding="utf-8") as f:
                            f.write(f"\n\n## Atomic Memories Generated {datetime.now().isoformat()}\n")
                            f.write("### Original Memory Metadata\n")
                            for key, value in original_metadata.items():
                                f.write(f"{key}: {value}\n")
                            
                            f.write("\n### Extracted Atomic Memories\n")
                            for atomic in atomic_memories:
                                # Log the structure of problematic memories
                                if not isinstance(atomic, dict):
                                    log_error(self.logger, f"Invalid atomic memory format: {atomic}")
                                    continue
                                    
                                # Safely get values with defaults
                                category = atomic.get('category', 'Uncategorized')
                                confidence = atomic.get('confidence', 0.0)
                                statement = atomic.get('statement', 'No statement provided')
                                
                                f.write(f"\n#### {category.title()} (confidence: {confidence})\n")
                                f.write(f"Statement: {statement}\n")
                                
                                # Optional: log full memory structure for debugging
                                log_event(self.logger, "memory.structure", 
                                        f"Memory structure: {atomic}", level="DEBUG")
                            f.write("\n---\n")
                    
                    # Properly await the thread execution
                    await asyncio.to_thread(write_memories)
            
            log_event(self.logger, "memory.dumped", 
                     f"Saved {len(atomic_memories)} atomic memories to disk",
                     level="DEBUG")
        except Exception as e:
            log_error(self.logger, 
                     f"Failed to save atomic memories to disk: {str(e)}\n"
                     f"Atomic memories: {atomic_memories}\n"
                     f"Original metadata: {original_metadata}")

    async def _extract_learnings(self, days_threshold: int = 0) -> None:
        """Extract learning opportunities from short-term memories and feedback.
        
        Args:
            days_threshold: Number of days worth of memories to keep in short-term. 
                           Memories older than this will be processed into long-term storage.
                           Default is 0 (process all memories).
        """
        return await extract_learnings_impl(self, days_threshold)

    async def _run_learning_subroutine(self, category: str = None) -> None:
        """Run the learning subroutine."""
        return await learning_subroutine(self, category)

    async def _save_reflection_to_disk(self, reflections: List[Dict], original_metadata: Dict) -> None:
        """Save learning reflections to disk for debugging/backup."""
        return await save_reflection_to_disk_impl(self, reflections, original_metadata)

    async def _evaluate_and_send_feedback(
        self,
        receiver: str,
        conversation_id: str,
        response_content: str
    ) -> None:
        """Evaluate response quality and send feedback to the agent."""
        return await evaluate_and_send_feedback_impl(self, receiver, conversation_id, response_content)

    async def _evaluate_response(self, response_content: str) -> Tuple[int, str]:
        """Evaluate response quality using LLM."""
        return await evaluate_response_impl(self, response_content)

    #TODO: Not tested and not currently in use.
    async def _cleanup_conversation(self, conversation_id: str, collection_name: str) -> None:
        """Clean up memories from a specific conversation in the given collection.
        
        Args:
            conversation_id: The ID of the conversation to clean up
            collection_name: The name of the memory collection to clean up
        """
        try:
            # Get all memories for the specified conversation
            memories = await self.memory.retrieve(
                query="",
                collection_names=[collection_name],
                metadata_filter={"conversation_id": conversation_id}
            )
            
            if not memories:
                self.logger.info(f"No memories found for conversation {conversation_id} in {collection_name}")
                return

            # Delete memories
            for memory in memories:
                await self.memory.delete(
                    collection_name=collection_name,
                    metadata_filter={"memory_id": memory["metadata"].get("memory_id")}
                )
            
            self.logger.info(f"Cleaned up {len(memories)} memories from conversation {conversation_id} in {collection_name}")

        except Exception as e:
            self.logger.error(f"Failed to clean up conversation {conversation_id}: {str(e)}")

    async def _start_socialising(self) -> Dict[str, Any]:
        """Start socialising with another agent."""
        return await start_socialising_impl(self)

    async def process_queue(self):
        """Memetic Agent process for processing any pending requests in the queue"""
        return await process_queue_impl(self)

    async def process_message(self, content: str, sender: str, prompt: PromptModel | None, evaluation: PromptEvaluation | None, message_type: str, conversation_id: str, request_id: str) -> str:
        """Process an incoming message, routing to either social or standard message handling.
        
        Args:
            content: The message content
            sender: The message sender
            prompt: Optional prompt model for social messages
            evaluation: Optional evaluation for social messages
            conversation_id: The conversation ID
            
        Returns:
            str: The processed response
        """
        # Social message if either prompt or evaluation is present
        is_social = prompt is not None or evaluation is not None
        
        if is_social:
            return await process_social_message_impl(
                self, 
                content=content,
                sender=sender,
                prompt=prompt,
                evaluation=evaluation, 
                message_type=message_type,
                conversation_id=conversation_id,
                request_id=request_id
            )
        
        # Standard message processing
        return await process_message_impl(
            self,
            content=content,
            sender=sender,
            conversation_id=conversation_id
        )

    async def _record_score(self, prompt_type: str, prompt_score: int, conversation_id: str, score_type: str) -> None:
        """Record a score for a prompt and update confidence scores when appropriate."""
        return await _record_score_impl(self, prompt_type, prompt_score, conversation_id, score_type)

    async def _update_confidence_score(self, prompt_type: str, initial_friend_score: int, 
                                     updated_friend_score: int, self_eval_score: int) -> None:
        """Calculate and update confidence score for a prompt type based on interaction scores."""
        return await _update_confidence_score_impl(self, prompt_type, initial_friend_score, updated_friend_score, self_eval_score)

    async def _evaluate_prompt(self, prompt_type: str, prompt: str) -> None:
        """Evaluate a prompt and update confidence scores."""
        return await _evaluate_prompt_impl(self, prompt_type, prompt)

    async def _calculate_updated_confidence_score(self, prompt_type: str, new_score: int) -> None:
        """Calculate the updated confidence score for the prompt."""
        return await _calculate_updated_confidence_score_impl(self, prompt_type, new_score)

    async def _start_sleeping(self) -> None:
        """Start the sleeping subroutine."""
        return await _start_sleeping_impl(self)

    async def _continue_or_stop(self, messages: List[Message]) -> str:
        """Evaluate message content to determine if we should continue or stop.

        Args:
            messages: List of conversation messages to evaluate
            
        Returns:
            "STOP" if conversation should end, or a rephrased continuation message
        """
        return await continue_or_stop_impl(self, messages)

    async def _save_memory(self) -> None:
        """Save new messages from conversations to memory store."""
        return await save_memory_impl(self)

    async def _save_memory_to_disk(self, structured_info: Dict, metadata: Dict, memory_type: str) -> None:  
        """Save structured memory information to disk for debugging/backup."""
        return await save_memory_to_disk_impl(self, structured_info, metadata, memory_type)

    async def _cleanup_memories(self, days_threshold: int = 0, collection_name: str = "short_term") -> None:
        """Clean up old memories from specified collection after consolidation.
        
        Args:
            days_threshold: Number of days worth of memories to keep.
                            Memories older than this will be deleted.
                            Default is 0 (clean up all processed memories).
            collection_name: Name of collection to clean up.
                            Default is "short_term".
        """
        return await cleanup_memories_impl(self, days_threshold, collection_name)

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals gracefully by creating async task."""
        self.logger.info("Shutdown signal received")
        # Create async task for shutdown which includes memory saving
        asyncio.create_task(self.shutdown())

    async def shutdown(self) -> None:
        """Gracefully shutdown the agent."""
        await self.set_status(AgentStatus.SHUTTING_DOWN, "shutdown")
        self._shutdown_event.set()
        
        self.logger.info(f"Shutting down {self.config.agent_name}")
        # Save memory as part of shutdown
        await self._save_memory()
        
        # Cancel any pending requests
        for future in self.pending_requests.values():
            if not future.done():
                future.cancel()
        
        # Clear queues
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

#TODO: The agent needs to be able to call up things like its architecture or curent prompts 
# so that it can use them when reflecting on memories.

#TODO: The agent needs to be able to call up its current system prompt and use it when reflecting on memories.