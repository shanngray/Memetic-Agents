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

#TODO: Memetic agent will have a series of modules that make up its system prompt. Some modules will be updated sub consciously via memory and others it will
# have conscious control over.
#It will also be able to create new modules.

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

        # Set up paths for all prompt modules
        self._reasoning_module_path = self.prompt_path / "reasoning_prompt.md"
        self._give_feedback_module_path = self.prompt_path / "give_feedback_prompt.md"
        self._thought_loop_module_path = self.prompt_path / "thought_loop_prompt.md"
        self._xfer_long_term_module_path = self.prompt_path / "xfer_long_term_prompt.md"
        self._self_improvement_module_path = self.prompt_path / "self_improvement_prompt.md"
        self._reflect_memories_module_path = self.prompt_path / "reflect_memories_prompt.md"
        self._evaluator_module_path = self.prompt_path / "evaluator_prompt.md"

        # Set up paths for schemas
        self._xfer_long_term_schema_path = self.prompt_path / "schemas/xfer_long_term_schema.json"
        self._reflect_memories_schema_path = self.prompt_path / "schemas/reflect_memories_schema.json"
        self._self_improvement_schema_path = self.prompt_path / "schemas/self_improvement_schema.json"
        self._give_feedback_schema_path = self.prompt_path / "schemas/give_feedback_schema.json"

        # Update path to confidence scores
        self._prompt_confidence_scores_path = self.scores_path / "prompt_confidence_scores.json"

        #TODO System prompt is currently loaded via config ... this should be made consistent with the sub modules
        # Path to system prompt
        self.system_path = self.prompt_path / "sys_prompt.md"

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

        # Update system prompt with reasoning module
        self._initialize_system_prompts()


    async def initialize(self) -> None:
        """Async initialization to load all required files. Run from server.py"""
        try:
            # Load all modules
            self._reasoning_prompt = await self._load_module(self._reasoning_module_path)
            self._give_feedback_prompt = await self._load_module(self._give_feedback_module_path)
            self._thought_loop_prompt = await self._load_module(self._thought_loop_module_path)
            self._xfer_long_term_prompt = await self._load_module(self._xfer_long_term_module_path)
            self._self_improvement_prompt = await self._load_module(self._self_improvement_module_path)
            self._reflect_memories_prompt = await self._load_module(self._reflect_memories_module_path)
            self._evaluator_prompt = await self._load_module(self._evaluator_module_path)

            # Load schemas
            self._xfer_long_term_schema = await self._load_module(self._xfer_long_term_schema_path)
            self._reflect_memories_schema = await self._load_module(self._reflect_memories_schema_path)
            self._self_improvement_schema = await self._load_module(self._self_improvement_schema_path)
            self._give_feedback_schema = await self._load_module(self._give_feedback_schema_path)
            
            # Load confidence scores
            self._prompt_confidence_scores = await self._load_confidence_scores(self._prompt_confidence_scores_path)
            
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
        """Load module content."""
        try:
            # Create synchronous context manager for FileLock
            lock = FileLock(f"{module_path}.lock")
            with lock:
                content = await asyncio.to_thread(module_path.read_text, encoding="utf-8")
                return content.strip()
        except FileNotFoundError:
            default_contents = "Module not found. Tell your maker to update: " + str(module_path)
            with FileLock(f"{module_path}.lock"):
                await asyncio.to_thread(module_path.write_text, default_contents, encoding="utf-8")
            return default_contents

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
            
            # Map prompt_type to corresponding path and variable
            path_map = {
                "system_prompt": self.system_path,
                "reasoning_prompt": self._reasoning_module_path,
                "give_feedback_prompt": self._give_feedback_module_path,
                "thought_loop_prompt": self._thought_loop_module_path,
                "xfer_long_term_prompt": self._xfer_long_term_module_path,
                "self_improvement_prompt": self._self_improvement_module_path,
                "reflect_memories_prompt": self._reflect_memories_module_path,
                "evaluator_prompt": self._evaluator_module_path
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
            
            new_content = f"{self._system_prompt}\n\nReasoning Module:\n{self._reasoning_prompt}"
            
            if system_message:
                system_message.content = new_content
            else:
                self.conversations[self.current_conversation_id].insert(0, Message(
                    role="system",
                    content=new_content
                ))

    def _initialize_system_prompts(self) -> None:
        """Initialize system prompt with tools and reasoning module."""
        # Get tool descriptions
        tools_desc = self._get_tool_descriptions()
        
        # Build complete system prompt
        system_prompt = (
            f"{self._system_prompt}\n\n"
            f"Available tools:\n{tools_desc}\n\n"
            f"Reasoning Module:\n{self._reasoning_prompt}"
        )
        self._system_prompt = system_prompt

        # Build complete evaluator prompt
        evaluator_prompt = (
            f"You are the self-reflecting observer of: {self.config.agent_name}\n\n"
            f"Your identity is: {self._system_prompt}\n\n"
            # f"Available tools:\n{tools_desc}\n\n"
            f"Observer Instructions:\n{self._evaluator_prompt}"
        )
        self._evaluator_prompt = evaluator_prompt

        # self.logger.debug(f"System prompt:\n\n {system_prompt}\n\n")
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
                    role="system",
                    content=system_prompt
                ))

    async def update_system_prompt(self, new_prompt: str) -> str:
        """Update your system prompt when you want to modify your core behavior.
        Use this tool to permanently change how you operate."""
        lock = FileLock(f"{self.system_path}.lock")
        with lock:
            await asyncio.to_thread(self.system_path.write_text, new_prompt, encoding="utf-8")
        return "System prompt updated successfully"

    async def set_status(self, new_status: AgentStatus, trigger: str) -> None:
        """Memetic Agent version of BaseAgent set_status includes learning from memory."""
        try:
            if new_status == self.status:
                log_event(self.logger, "status.change", 
                         f"Status unchanged - Current: {self.status.name}, "
                         f"New: {new_status.name}, Trigger: {trigger}")
                return

            valid_transitions = AgentStatus.get_valid_transitions(self.status)
            
            if new_status not in valid_transitions:
                log_event(self.logger, "status.change", 
                         f"Invalid status transition - Current: {self.status.name}, "
                         f"New: {new_status.name}, Trigger: {trigger}", level="ERROR")
                raise ValueError(
                    f"Invalid status transition from {self.status.name} to {new_status.name} caused by {trigger}"
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
                f"Status changed: {previous_status.name} -> {self.status.name} ({trigger})"
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
                    #learning_task = asyncio.create_task(self._run_learning_subroutine())
                    #await learning_task
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
        try:
            if self.status != AgentStatus.MEMORISING:
                # await self.set_status(AgentStatus.MEMORISING, "Long-Term Memory transfer triggered")
                log_error(self.logger, "Agent must be in MEMORISING state to transfer to long-term")
                return

            log_event(self.logger, "agent.memorising", "Beginning atomic memory extraction process", level="DEBUG")
            
            # Retrieve recent memories from short-term storage
            short_term_memories = await self.memory.retrieve(
                query="",  # Empty query to get all memories
                collection_names=["short_term"],
                n_results=100
            )
            
            # Add debug logging
            for memory in short_term_memories:
                log_event(self.logger, "memory.debug", 
                         f"Memory structure: {memory}", level="DEBUG")

            # Filter memories based on threshold
            threshold_date = datetime.now() - timedelta(days=days_threshold)
            short_term_memories = [
                memory for memory in short_term_memories
                if datetime.fromisoformat(memory["metadata"].get("timestamp", "")) < threshold_date
            ]
            
            if not short_term_memories:
                log_error(self.logger, "memory.error", "No recent memories found for atomization")
                # await self.set_status(self._previous_status, "No recent memories found for atomization")
                return

            # Group memories by conversation ID
            conversation_memories = {}
            for memory in short_term_memories:
                conv_id = memory["metadata"].get("conversation_id")
                if conv_id not in conversation_memories:
                    conversation_memories[conv_id] = []
                conversation_memories[conv_id].append(memory)

            # Sort memories within each conversation by timestamp
            for conv_id in conversation_memories:
                conversation_memories[conv_id].sort(
                    key=lambda x: datetime.fromisoformat(x["metadata"].get("timestamp", ""))
                )

            # Process each conversation group
            for conv_id, memories in conversation_memories.items():
                try:
                    # Combine memory content in chronological order
                    combined_content = "\n".join(memory["content"] for memory in memories)
                    
                    # Store original metadata from first memory in conversation
                    original_metadata = memories[0]["metadata"] if memories else {}
                    
                    feedback_items = await self.memory.retrieve(
                        query="",
                        collection_names=["feedback"],
                        n_results=100,
                        metadata_filter={"conversation_id": conv_id}
                    )

                    # Add feedback content if any exists
                    if feedback_items:
                        feedback_content = "\n".join(item["content"] for item in feedback_items)
                        combined_content += f"\n\n{feedback_content}"

                    log_event(self.logger, "memory.transfer.content",
                             f"Combined content:\n{combined_content[:100]}...",
                             level="DEBUG")
                    # Extract atomic memories using LLM
                    
                    full_prompt = self._xfer_long_term_prompt + "\n\nFormat your response as an array of objects with the following schema:\n" + self._xfer_long_term_schema

                    atomic_response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": full_prompt},
                            {"role": "user", "content": f"Memory content:\n{combined_content}"}
                        ],
                        response_format={ "type": "json_object" }
                    )
                    
                    # Parse and validate the LLM response
                    raw_response = atomic_response.choices[0].message.content
                    log_event(self.logger, "memory.atomic.response", f"Raw LLM response: {raw_response}", level="DEBUG")

                    try:
                        atomic_memories = json.loads(raw_response)
                        # Check if response has memories/atomic_memories key and convert to list
                        if isinstance(atomic_memories, dict):
                            for key in ["memories", "atomic_memories", "atomized_memories"]:
                                if key in atomic_memories:
                                    atomic_memories = atomic_memories[key]
                                    break
                        
                        # Ensure atomic_memories is a list
                        if not isinstance(atomic_memories, list):
                            atomic_memories = [atomic_memories]
                        
                        # Validate each memory has required fields
                        validated_memories = []
                        required_fields = ["statement", "metatags", "thoughts", "confidence", "category"]
                        
                        for memory in atomic_memories:
                            if isinstance(memory, dict) and all(k in memory for k in required_fields):
                                validated_memories.append(memory)
                            else:
                                log_event(self.logger, "memory.atomic.invalid", 
                                         f"Invalid memory format, missing required fields: {memory}",
                                         level="DEBUG")
                        
                        if not validated_memories:
                            # If no valid memories, retry with LLM with more explicit instructions
                            retry_prompt = (
                                f"{self._xfer_long_term_prompt}\n\n"
                                "IMPORTANT: Your response must be a JSON array of objects with this exact structure:\n"
                                f"{self._xfer_long_term_schema}"
                            )
                            
                            retry_response = await self.client.chat.completions.create(
                                model=self.config.model,
                                messages=[
                                    {"role": "system", "content": retry_prompt},
                                    {"role": "user", "content": f"Previous response was invalid. Please reformat this content into the exact structure specified:\n{raw_response}"}
                                ],
                                response_format={ "type": "json_object" }
                            )
                            
                            retry_content = retry_response.choices[0].message.content
                            log_event(self.logger, "memory.atomic.retry", f"Retry response: {retry_content}", level="DEBUG")
                            
                            try:
                                retry_json = json.loads(retry_content)
                                if isinstance(retry_json, dict) and "atomic_memories" in retry_json:
                                    validated_memories = retry_json["atomic_memories"]
                                else:
                                    log_error(self.logger, "Retry failed to produce valid format")
                                    continue
                            except json.JSONDecodeError:
                                log_error(self.logger, f"Failed to parse retry response as JSON: {retry_content}")
                                continue

                        # Continue processing with validated_memories
                        for atomic in validated_memories:
                            try:
                                # Get original memory metadata safely
                                original_metadata = memory.get("metadata", {})
                                
                                metadata = {
                                    "memory_id": str(uuid.uuid4()),
                                    "original_timestamp": original_metadata.get("timestamp", datetime.now().isoformat()),
                                    "source_type": "atomic_memory",
                                    "confidence": atomic.get("confidence", 0.5),  # Default confidence if missing
                                    "category": atomic.get("category", "uncategorized"),  # Default category if missing
                                    "timestamp": datetime.now().isoformat(),
                                    "original_memory_id": original_metadata.get("chroma_id", "unknown"),  # Track original memory
                                    "conversation_id": original_metadata.get("conversation_id", "unknown")
                                }

                                formatted_memory = (
                                    f"{atomic['statement']}\n\n"
                                    f"MetaTags: {', '.join(atomic.get('metatags', []))}\n\n"
                                    f"Thoughts: {atomic.get('thoughts', 'No additional thoughts')}"
                                )
                                
                                await self.memory.store(
                                    content=atomic["statement"],
                                    collection_name="long_term",
                                    metadata=metadata
                                )
                                
                                log_event(self.logger, "memory.atomic.stored",
                                         f"Stored atomic memory: {atomic['statement']} ({atomic.get('category', 'uncategorized')})",
                                         level="DEBUG")
                            except Exception as e:
                                log_error(self.logger, f"Failed to store atomic memory: {str(e)}")
                                continue

                        # Save to disk with the stored original metadata
                        await self._save_atomic_memory_to_disk(atomic_memories, original_metadata)

                    except json.JSONDecodeError:
                        log_error(self.logger, f"Failed to parse LLM response as JSON: {raw_response}")
                        raise

                except Exception as e:
                    log_error(self.logger, f"Failed to process conversation {conv_id} into atomic form: {str(e)}", exc_info=e)
                    continue
            
            log_event(self.logger, "memory.atomization.complete",
                     f"Completed memory atomization for {len(short_term_memories)} memories",
                     level="DEBUG")
             
        except Exception as e:
            log_error(self.logger, "Failed to atomize memories", exc_info=e)
        finally:
            if self.status == AgentStatus.MEMORISING:
                await self.set_status(self._previous_status, "Memory atomization complete")

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
        try:
            if self.status != AgentStatus.MEMORISING:
                log_error(self.logger, "Agent must be in MEMORISING state to extract learnings")
                return
                # await self.set_status(AgentStatus.MEMORISING, "Extracting Learnings triggered")
            
            log_event(self.logger, "agent.memorising", "Beginning learning memory extraction process", level="DEBUG")
            
            # Retrieve and filter memories
            short_term_memories = await self.memory.retrieve(
                query="",
                collection_names=["short_term"],
                n_results=1000  # Hard limit (1000 entries) prevents infinite processing
            )
            
            # Filter memories based on threshold
            threshold_date = datetime.now() - timedelta(days=days_threshold)
            short_term_memories = [
                memory for memory in short_term_memories
                if datetime.fromisoformat(memory["metadata"].get("timestamp", "")) < threshold_date
            ]
            
            total_memories = len(short_term_memories)
            log_event(self.logger, "memory.extraction.progress", 
                     f"Found {total_memories} memories to process")
            
            if not short_term_memories:
                log_error(self.logger, "memory.error", "No recent memories found for reflection")
                # await self.set_status(self._previous_status, "No recent memories found for reflection")
                return

            # Group and sort memories
            conversation_memories = {}
            for memory in short_term_memories:
                conv_id = memory["metadata"].get("conversation_id")
                if conv_id not in conversation_memories:
                    conversation_memories[conv_id] = []
                conversation_memories[conv_id].append(memory)

            total_conversations = len(conversation_memories)
            log_event(self.logger, "memory.extraction.progress", 
                     f"Grouped into {total_conversations} conversations")

            # Process each conversation group
            for conv_idx, (conv_id, memories) in enumerate(conversation_memories.items(), 1):
                try:
                    log_event(self.logger, "memory.extraction.progress", 
                             f"Processing conversation {conv_idx} of {total_conversations}")
                    
                    # Sort memories within conversation by timestamp
                    memories.sort(key=lambda x: datetime.fromisoformat(x["metadata"].get("timestamp", "")))
                    
                    # Combine memory content
                    combined_content = "\n".join(memory["content"] for memory in memories)
                    
                    # Get feedback for this conversation
                    feedback_items = await self.memory.retrieve(
                        query="",
                        collection_names=["feedback"],
                        n_results=100,  # Hard limit prevents infinite feedback processing
                        metadata_filter={"conversation_id": conv_id}
                    )

                    if feedback_items:
                        feedback_content = "\n".join(item["content"] for item in feedback_items)
                        combined_content += f"\n\nFeedback:\n{feedback_content}"

                    full_prompt = self._reflect_memories_prompt + "\n\nFormat your response as an array of objects with the following schema:\n" + self._reflect_memories_schema

                    # Extract learnings using LLM
                    reflection_response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": full_prompt},
                            {"role": "user", "content": f"Memory content:\n{combined_content}"}
                        ],
                        response_format={ "type": "json_object" }
                    )
                    
                    raw_response = reflection_response.choices[0].message.content
                    
                    # Process reflections with one retry attempt
                    try:
                        reflections = self._process_reflection_response(raw_response)
                        if not reflections:  # If initial processing fails, try retry prompt
                            reflections = await self._retry_reflection_processing(
                                self._reflect_memories_prompt, raw_response
                            )
                            if not reflections:  # If retry fails, skip this conversation
                                log_event(self.logger, "memory.reflection.skip", 
                                         f"Skipping conversation {conv_id} due to invalid response format")
                                continue
                        
                        # Store validated reflections
                        total_reflections = len(reflections)
                        for refl_idx, reflection in enumerate(reflections, 1):
                            log_event(self.logger, "memory.reflection.progress", 
                                     f"Storing reflection {refl_idx} of {total_reflections} for conversation {conv_idx}")
                            
                            await self._store_reflection(reflection, conv_id, memories[0]["metadata"])
                        
                        # Save to disk for debugging
                        await self._save_reflection_to_disk(reflections, memories[0]["metadata"])
                        
                    except Exception as e:
                        log_error(self.logger, f"Failed to process reflections for conversation {conv_id}: {str(e)}")
                        continue

                except Exception as e:
                    log_error(self.logger, f"Failed to process conversation {conv_id}: {str(e)}")
                    continue
            
            log_event(self.logger, "memory.reflection.complete",
                     f"Completed memory reflection for {total_conversations} conversations")
             
        except Exception as e:
            log_error(self.logger, "Failed to process reflections", exc_info=e)
        finally:
            if self.status == AgentStatus.MEMORISING:
                await self.set_status(self._previous_status, "Memory reflection complete")

    async def _run_learning_subroutine(self, category: str = None) -> None:
        """Run the learning subroutine."""
        return await learning_subroutine(self, category)

    def _process_reflection_response(self, raw_response: str) -> List[Dict]:
        """Process and validate reflection response from LLM."""
        try:
            reflections = json.loads(raw_response)
            # Check for various possible key names
            if isinstance(reflections, dict):
                for key in ["learning_opportunities", "lessons", "reflections", "learnings"]:
                    if key in reflections:
                        reflections = reflections[key]
                        break
            
            # Ensure reflections is a list
            if not isinstance(reflections, list):
                reflections = [reflections]
            
            # Validate required fields
            validated_reflections = []
            required_fields = ["lesson", "importance", "category", "thoughts"]
            
            for reflection in reflections:
                if isinstance(reflection, dict) and all(k in reflection for k in required_fields):
                    validated_reflections.append(reflection)
            
            return validated_reflections
        except json.JSONDecodeError:
            return []

    async def _retry_reflection_processing(self, base_prompt: str, failed_response: str) -> List[Dict]:
        """Retry processing reflections with more explicit instructions."""
        retry_prompt = (
            f"{base_prompt}\n\n"
            "IMPORTANT: Your response must be a JSON object with this exact structure:\n"
            '{\n'
            '    "learning_opportunities": [\n'
            '        {\n'
            '            "lesson": "The main learning point",\n'
            '            "importance": 0.8,\n'
            '            "category": "category_name",\n'
            '            "thoughts": "Additional context and reasoning"\n'
            '        }\n'
            '    ]\n'
            '}'
        )
        
        retry_response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": retry_prompt},
                {"role": "user", "content": f"Previous response was invalid. Please reformat this content into the exact structure specified:\n{failed_response}"}
            ],
            response_format={ "type": "json_object" }
        )
        
        return self._process_reflection_response(retry_response.choices[0].message.content)

    async def _store_reflection(self, reflection: Dict, conv_id: str, original_metadata: Dict) -> None:
        """Store a single reflection in memory."""
        metadata = {
            "memory_id": str(uuid.uuid4()),
            "conversation_id": conv_id,
            "original_timestamp": original_metadata.get("timestamp"),
            "source_type": "learning_reflection",
            "importance": reflection["importance"],
            "category": reflection["category"],
            "timestamp": datetime.now().isoformat()
        }

        content = (
            f"Lesson: {reflection['lesson']}\n"
            f"Thoughts: {reflection['thoughts']}\n"
        )

        await self.memory.store(
            content=reflection["lesson"],
            collection_name="reflections",
            metadata=metadata
        )
        
        log_event(self.logger, "memory.reflection.stored",
                 f"Stored learning reflection: {reflection['lesson']} ({reflection['category']})", level="DEBUG")

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
        
#TODO: The agent needs to eb able to call up things like its architecture or curent prompts 
# so that it can use them when reflecting on memories.

#TODO: The agent needs to be able to call up its current system prompt and use it when reflecting on memories.