import sys
import json
import re
import inspect
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache
import importlib
from typing import Callable
from filelock import FileLock
from chromadb import PersistentClient
from typing import List, Dict, Any
import uuid
import asyncio
import httpx
# Add project root to path
sys.path.append(str(Path(__file__).parents[2]))

from src.log_config import log_event, log_error
from base_agent.base_agent import BaseAgent
from base_agent.config import AgentConfig
from base_agent.models import Message
from base_agent.models import AgentStatus
from src.api_server.models.api_models import PromptModel, APIMessage, SocialMessage, PromptEvaluation
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

class MemeticAgent(BaseAgent):
    def __init__(self, api_key: str, chroma_client: PersistentClient, config: AgentConfig = None):
        """Initialize MemeticAgent with reasoning capabilities."""
        # Initialize base collections first
        super().__init__(api_key=api_key, chroma_client=chroma_client, config=config)
        
        self.prompt_path = Path("agent_files") / self.config.agent_name / "prompt_modules"
        self.prompt_path.mkdir(parents=True, exist_ok=True)

        # Create core directories
        self.scores_path = self.prompt_path / "scores"
        self.scores_path.mkdir(parents=True, exist_ok=True)

        # Set up paths for schemas
        schema_path = self.prompt_path / "schemas"
        schema_path.mkdir(parents=True, exist_ok=True)
        
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

        #TODO System prompt is currently loaded via config ... this should be made consistent with the sub modules


        # Add reasoning-related tool to internal_tools dictionary
        self.internal_tools.update({
            "update_prompt_module": self.update_prompt_module,
            "update_system_prompt": self.update_system_prompt
        })

        # Register the new internal tools
        for tool_name, tool_func in {
            "update_prompt_module": self.update_prompt_module,
            "update_system_prompt": self.update_system_prompt
        }.items():
            self.tool_mod.register(tool_func)

    async def initialize(self) -> None:
        """Async initialization to load all required files. Run from main.py"""
        try:
            # Load all prompt contents
            self.prompt.reasoning.content = await self._load_module(self.prompt.reasoning.path)
            self.prompt.give_feedback.content = await self._load_module(self.prompt.give_feedback.path)
            self.prompt.thought_loop.content = await self._load_module(self.prompt.thought_loop.path)
            self.prompt.xfer_long_term.content = await self._load_module(self.prompt.xfer_long_term.path)
            self.prompt.self_improvement.content = await self._load_module(self.prompt.self_improvement.path)
            self.prompt.reflect_memories.content = await self._load_module(self.prompt.reflect_memories.path)
            self.prompt.evaluator.content = await self._load_module(self.prompt.evaluator.path)

            # Load schemas
            self.prompt.xfer_long_term.schema = await self._load_module(self.prompt.xfer_long_term.schema_path)
            self.prompt.reflect_memories.schema = await self._load_module(self.prompt.reflect_memories.schema_path)
            self.prompt.self_improvement.schema = await self._load_module(self.prompt.self_improvement.schema_path)
            self.prompt.give_feedback.schema = await self._load_module(self.prompt.give_feedback.schema_path)
            self.prompt.thought_loop.schema = await self._load_module(self.prompt.thought_loop.schema_path)

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
            
            # Initialize system prompts after all content is loaded
            self._initialize_system_prompts()
            
            self.logger.info("Agent initialization completed successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {str(e)}")
            raise

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

    async def update_system_prompt(self, new_prompt: str) -> str:
        """Update your system prompt when you want to modify your core behavior.
        Use this tool to permanently change how you operate."""
        lock = FileLock(f"{self.prompt.system.path}.lock")
        with lock:
            await asyncio.to_thread(self.prompt.system.path.write_text, new_prompt, encoding="utf-8")
        return "System prompt updated successfully"

    async def set_status(self, new_status: AgentStatus, trigger: str) -> None:
        """Memetic Agent version of BaseAgent set_status includes learning from memory."""
        try:
            if new_status == self.status:
                log_event(self.logger, "status.change", 
                         f"Status unchanged - Current: /{self.status.name}, "
                         f"New: {new_status.name}, Trigger: {trigger}")
                return

            valid_transitions = AgentStatus.get_valid_transitions(self.status)
            
            if new_status not in valid_transitions:
                log_event(self.logger, "status.change", 
                         f"Invalid status transition - Current: /{self.status.name}, "
                         f"New: {new_status.name}, Trigger: {trigger}", level="ERROR")
                raise ValueError(
                    f"Invalid status transition from /{self.status.name} to /{new_status.name} caused by {trigger}"
                )

            # Store previous status before updating
            previous_status = self.status
            
            # Update status
            self.status = new_status
            self._previous_status = previous_status
            self._status_history.append({
                "timestamp": datetime.now().isoformat(),
                "from": previous_status,
                "to": self.status,
                "trigger": trigger
            })
            log_event(
                self.logger,
                f"agent.{self.status.name}",
                f"Status changed: /{previous_status.name} -> /{self.status.name} ({trigger})"
            )
            
            # Handle MEMORISING state tasks
            if new_status == AgentStatus.MEMORISING and previous_status != AgentStatus.MEMORISING:
                try:
                    # Create tasks but store their references
                    transfer_task = asyncio.create_task(self._transfer_to_long_term())
                    learning_task = asyncio.create_task(self._extract_learnings())
                    
                    # Wait for both tasks to complete before cleanup
                    await asyncio.gather(transfer_task, learning_task)
                    
                    # Now run cleanup
                    clean_short_task = asyncio.create_task(self._cleanup_memories(days_threshold=0, collection_name="short_term"))
                    clean_feedback_task = asyncio.create_task(self._cleanup_memories(days_threshold=0, collection_name="feedback"))
                    await asyncio.gather(clean_short_task, clean_feedback_task)
                finally:
                    # Direct status update without recursive call
                    if self.status != AgentStatus.SHUTTING_DOWN:
                        self.status = previous_status
                        log_event(
                            self.logger,
                            f"agent.{self.status.name}",
                            f"Status restored: {AgentStatus.MEMORISING.name} -> {self.status.name} (Memory processing complete)"
                        )

            if new_status == AgentStatus.LEARNING:
                try:
                    await self._run_learning_subroutine()
                    if self.status != AgentStatus.SHUTTING_DOWN:
                        await self.set_status(AgentStatus.AVAILABLE, "Learning complete")
                except Exception as e:
                    self.logger.error(f"Learning failed: {str(e)}")
                    if self.status != AgentStatus.SHUTTING_DOWN:
                        await self.set_status(AgentStatus.AVAILABLE, "Learning failed")

            if new_status == AgentStatus.SOCIALISING:
                try:
                    # Create task for socializing but don't await it
                    asyncio.sleep(0.1)
                    asyncio.create_task(self._start_socialising())
                except Exception as e:
                    self.logger.error(f"Failed to start socialising task: {str(e)}")
                    if self.status != AgentStatus.SHUTTING_DOWN:
                        await self.set_status(AgentStatus.AVAILABLE, "Failed to start socialising")
                
            if new_status == AgentStatus.SLEEPING:
                try:
                    # Create task for sleeping but don't await it
                    asyncio.sleep(0.1)
                    await self._start_sleeping()
                    if self.status != AgentStatus.SHUTTING_DOWN:
                        await self.set_status(AgentStatus.AVAILABLE, "Sleeping complete")
                except Exception as e:
                    self.logger.error(f"Failed to start sleeping task: {str(e)}")
                    if self.status != AgentStatus.SHUTTING_DOWN:
                        await self.set_status(AgentStatus.AVAILABLE, "Failed to start sleeping")
        except Exception as e:
            self.logger.error(f"Status update failed: {str(e)}")

    async def receive_message(self, message: APIMessage) -> str:
        return await receive_message_impl(self, message)
    
    async def receive_social_message(self, message: SocialMessage) -> str:
        return await receive_social_message_impl(self, message)

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
        try:
            reflection_dump_path = self.files_path / f"{self.config.agent_name}_reflection_dump.md"
            lock = FileLock(f"{reflection_dump_path}.lock")
            
            async with asyncio.Lock():  # Use asyncio.Lock() for async context
                with lock:
                    def write_reflections():
                        with open(reflection_dump_path, "a", encoding="utf-8") as f:
                            f.write(f"\n\n## Learning Reflections Generated {datetime.now().isoformat()}\n")
                            f.write("### Original Memory Metadata\n")
                            for key, value in original_metadata.items():
                                f.write(f"{key}: {value}\n")
                            
                            f.write("\n### Extracted Learnings\n")
                            for reflection in reflections:
                                category = reflection.get('category', 'Uncategorized')
                                importance = reflection.get('importance', 'N/A')
                                lesson = reflection.get('lesson', 'No lesson recorded')
                                thoughts = reflection.get('thoughts', 'No additional thoughts')
                                
                                f.write(f"\n#### {category.title()} (importance: {importance})\n")
                                f.write(f"Lesson: {lesson}\n")
                                f.write(f"Thoughts: {thoughts}\n")
                            f.write("\n---\n")
                    
                    # Properly await the thread execution
                    await asyncio.to_thread(write_reflections)
            
            log_event(self.logger, "reflection.dumped", 
                     f"Saved {len(reflections)} learning reflections to disk",
                     level="DEBUG")
        except Exception as e:
            log_error(self.logger, 
                     f"Failed to save learning reflections to disk: {str(e)}")

    async def _process_feedback(self, days_threshold: int = 0) -> None:
        """Process and transfer feedback to long-term memory."""
        return await process_feedback_impl(self, days_threshold)

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

    async def start(self):
        """Start the agent's main processing loop and initialise memories"""
        self.logger.info(f"Starting {self.config.agent_name} processing loop")
        
        # Initialize and load memory collections before starting the processing loop
        # Include memetic-specific collections
        await self._initialize_and_load_memory_collections(
            ["short_term", "long_term", "feedback", "reflections"]
        )
        
        while not self._shutdown_event.is_set():
            try:
                await self.process_queue()
                await asyncio.sleep(0.1)  # Prevent CPU spinning
            except Exception as e:
                self.logger.error(f"Error in processing loop: {str(e)}")
                if self.status != AgentStatus.SHUTTING_DOWN:
                    await self.set_status(AgentStatus.AVAILABLE, "start")

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
        
#TODO: The agent needs to be able to call up things like its architecture or curent prompts 
# so that it can use them when reflecting on memories.

#TODO: The agent needs to be able to call up its current system prompt and use it when reflecting on memories.