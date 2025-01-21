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
from typing import List, Dict, Any
import uuid
import asyncio
# Add project root to path
sys.path.append(str(Path(__file__).parents[2]))

from src.log_config import log_event
from base_agent.base_agent import BaseAgent
from base_agent.config import AgentConfig
from base_agent.models import Message
from base_agent.models import AgentStatus
from src.api_server.models.api_models import PromptModel, APIMessage
from .modules.start_socialising_impl import start_socialising_impl
from .modules.receive_message_impl import receive_message_impl
from .modules.process_queue_impl import process_queue_impl
from .modules.process_message_impl import process_message_impl
from .modules.process_social_message_impl import process_social_message_impl

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

        self._reasoning_module_path = self.prompt_path / "reasoning_prompt.md"
        self._give_feedback_module_path = self.prompt_path / "give_feedback_prompt.md"
        self._thought_loop_module_path = self.prompt_path / "thought_loop_prompt.md"
        self._xfer_long_term_module_path = self.prompt_path / "xfer_long_term_prompt.md"
        self._self_improvement_module_path = self.prompt_path / "self_improvement_prompt.md"
        self._reflect_memories_module_path = self.prompt_path / "reflect_memories_prompt.md"
        self._evaluator_module_path = self.prompt_path / "evaluator_prompt.md"

        self._xfer_long_term_schema_path = self.prompt_path / "schemas/xfer_long_term_schema.json"
        self._reflect_memories_schema_path = self.prompt_path / "schemas/reflect_memories_schema.json"

        #TODO System prompt is currently loaded via config ... this should be made consistent with the sub modules
        # Path to system prompt
        self.system_path = self.prompt_path / "sys_prompt.md"
        
        # Load sub modules
        self._reasoning_prompt = self._load_module(self._reasoning_module_path)
        self._give_feedback_prompt = self._load_module(self._give_feedback_module_path)
        self._thought_loop_prompt = self._load_module(self._thought_loop_module_path)
        self._xfer_long_term_prompt = self._load_module(self._xfer_long_term_module_path)
        self._self_improvement_prompt = self._load_module(self._self_improvement_module_path)
        self._reflect_memories_prompt = self._load_module(self._reflect_memories_module_path)
        self._evaluator_prompt = self._load_module(self._evaluator_module_path)

        self._xfer_long_term_schema = self._load_module(self._xfer_long_term_schema_path)
        self._reflect_memories_schema = self._load_module(self._reflect_memories_schema_path)
        
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

    def _load_module(self, module_path: Path) -> str:
        """Load module content."""
        try:
            with FileLock(f"{module_path}.lock"):
                return module_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            default_contents = "Module not found. Tell your maker to update: " + str(module_path)
            module_path.write_text(default_contents, encoding="utf-8")
            return default_contents

    async def update_prompt_module(self, prompt_type: str, new_prompt: str) -> str:
        """Update a specific prompt module.
        
        Args:
            prompt_type: Type of prompt to update (reasoning, give_feedback, thought_loop, etc.)
            new_prompt: The new prompt content
        
        Returns:
            str: Success message
        
        Raises:
            ValueError: If prompt type is invalid or update fails
        """
        try:
            # Map prompt_type to corresponding path and variable
            prompt_map = {
                "reasoning": (self._reasoning_module_path, "_reasoning_prompt"),
                "give_feedback": (self._give_feedback_module_path, "_give_feedback_prompt"),
                "thought_loop": (self._thought_loop_module_path, "_thought_loop_prompt"),
                "xfer_long_term": (self._xfer_long_term_module_path, "_xfer_long_term_prompt"),
                "self_improvement": (self._self_improvement_module_path, "_self_improvement_prompt"),
                "reflect_memories": (self._reflect_memories_module_path, "_reflect_memories_prompt"),
                "evaluator": (self._evaluator_module_path, "_evaluator_prompt")
            }
            
            if prompt_type not in prompt_map:
                raise ValueError(f"Invalid prompt type: {prompt_type}. Valid types are: {', '.join(prompt_map.keys())}")
            
            path, var_name = prompt_map[prompt_type]
            
            # Update file
            with FileLock(f"{path}.lock"):
                path.write_text(new_prompt, encoding="utf-8")
            
            # Update instance variable
            setattr(self, var_name, new_prompt)
            
            # If updating reasoning module, update system prompt
            if prompt_type == "reasoning":
                await self._update_system_with_reasoning()
                
            return f"{prompt_type} module updated successfully"
            
        except Exception as e:
            raise ValueError(f"Failed to update {prompt_type} module: {str(e)}")

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

    def _initialize_system_prompt(self) -> None:
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

    async def set_status(self, new_status: AgentStatus, trigger: str) -> None:
        """Memetic Agent version of BaseAgent set_status includes learning from memory."""
        async with self._status_lock:
            if new_status == self.status:
                return

            valid_transitions = AgentStatus.get_valid_transitions(self.status)
            
            if new_status not in valid_transitions:
                raise ValueError(
                    f"Invalid status transition from {self.status} to {new_status} caused by {trigger}"
                )

            self._previous_status = self.status
            self.status = new_status
            self._status_history.append({
                "timestamp": datetime.now().isoformat(),
                "from": self._previous_status,
                "to": self.status,
                "trigger": trigger
            })
            log_event(
                self.logger,
                f"agent.{self.status}",
                f"Status changed: {self._previous_status} -> {self.status} ({trigger})"
            )
            
            # Trigger memory consolidation when entering MEMORISING state
            if new_status == AgentStatus.MEMORISING:
                # Create tasks but store their references
                transfer_task = asyncio.create_task(self._transfer_to_long_term())
                learning_task = asyncio.create_task(self._extract_learnings())
                
                # Wait for both tasks to complete before cleanup
                await asyncio.gather(transfer_task, learning_task)
                
                # Now run cleanup
                await self._cleanup_memories("short_term")
                await self._cleanup_memories("feedback")

            if new_status == AgentStatus.LEARNING:
                asyncio.create_task(self._run_learning_subroutine())

    async def receive_message(self, message: APIMessage) -> str:
        return await receive_message_impl(self, message)

    async def _transfer_to_long_term(self, days_threshold: int = 0) -> None:
        """Transfer short-term memories into long-term storage as atomic memories with SPO metadata.
        
        Args:
            days_threshold: Number of days worth of memories to keep in short-term. 
                           Memories older than this will be processed into long-term storage.
                           Default is 0 (process all memories).
        """
        try:
            if self.status != AgentStatus.MEMORISING:
                self.set_status(AgentStatus.MEMORISING, "Long-Term Memory transfer triggered")

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
                    
                    feedback_items = await self.memory.retrieve(
                        query="",
                        collection_names=["feedback"],
                        n_results=100,
                        where={"metadata.conversation_id": conv_id}
                    )

                    # Add feedback content if any exists
                    if feedback_items:
                        feedback_content = "\n".join(item["content"] for item in feedback_items)
                        combined_content += f"\n\n{feedback_content}"

                    print("\n//----------------COMBINED CONTENT----------------//\n")
                    print(combined_content)
                    print("\n//------------------------------------------------//\n")
                    # Extract atomic memories using LLM
                    atomic_response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": self._xfer_long_term_prompt},
                            {"role": "user", "content": f"Memory content:\n{combined_content}"}
                        ],
                        response_format={ "type": "json_object" }
                    )
                    
                    atomic_memories = json.loads(atomic_response.choices[0].message.content)
                    
                    # Store each atomic memory
                    for atomic in atomic_memories:
                        formatted_memory = (
                            f"{atomic['statement']}\n\n"
                            f"MetaTags: {', '.join(atomic['metatags'])}\n\n"
                            f"Thoughts: {atomic['thoughts']}"
                        )
                        
                        metadata = {
                            "memory_id": str(uuid.uuid4()),
                            "original_timestamp": memory["metadata"].get("timestamp"),
                            "source_type": "atomic_memory", #Is this correct format for chroma?
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
                    log_error(self.logger, f"Failed to process conversation {conv_id} into atomic form: {str(e)}", exc_info=e)
                    continue
            
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
                    f.write("\n---\n")
                
            log_event(self.logger, "memory.dumped", 
                     f"Saved {len(atomic_memories)} atomic memories to disk")
        except Exception as e:
            log_error(self.logger, 
                     f"Failed to save atomic memories to disk: {str(e)}")

    # TODO: By making extract learnings a two-step process we run the risk of losing important details. We also create a situation where the agent is making
    # modifications using the output from a previous LLM call.
    # Another solution might be to flag the memory as a learning opportunity and then run the reflection process on the original memory.
    async def _extract_learnings(self, days_threshold: int = 0) -> None:
        """Extract learning opportunities from short-term memories and feedback.
        
        Args:
            days_threshold: Number of days worth of memories to keep in short-term. 
                           Memories older than this will be processed into long-term storage.
                           Default is 0 (process all memories).
        """
        try:
            if self.status != AgentStatus.MEMORISING:
                self.set_status(AgentStatus.MEMORISING, "Extracting Learnings triggered")
            
            log_event(self.logger, "agent.memorising", "Beginning learning memory extraction process")
            
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
                log_event(self.logger, "memory.error", "No recent memories found for reflection")
                await self.set_status(self._previous_status)
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
                    
                    feedback_items = await self.memory.retrieve(
                        query="",
                        collection_names=["feedback"],
                        n_results=100,
                        where={"metadata.conversation_id": conv_id}
                    )

                    # Add feedback content if any exists
                    if feedback_items:
                        feedback_content = "\n".join(item["content"] for item in feedback_items)
                        combined_content += f"\n\nFeedback:\n{feedback_content}"

                    # Extract learnings using LLM
                    reflection_response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": self._reflect_memories_prompt},
                            {"role": "user", "content": f"Memory content:\n{combined_content}"}
                        ],
                        response_format={ "type": "json_object" }
                    )
                    
                    reflections = json.loads(reflection_response.choices[0].message.content)
                    
                    # Store each reflection
                    for reflection in reflections:
                        formatted_memory = (
                            f"{reflection['lesson']}\n\n"
                            f"Importance: {reflection['importance']}\n\n"
                            f"Category: {reflection['category']}\n\n"
                            f"Thoughts: {reflection['thoughts']}"
                        )
                        
                        metadata = {
                            "memory_id": str(uuid.uuid4()),
                            "conversation_id": conv_id,
                            "original_timestamp": memories[0]["metadata"].get("timestamp"),
                            "source_type": "learning_reflection",
                            "importance": reflection["importance"],
                            "category": reflection["category"],
                            "timestamp": datetime.now().isoformat()
                        }

                        await self.memory.store(
                            content=reflection["lesson"],
                            collection_name="reflections",
                            metadata=metadata
                        )
                        
                        log_event(self.logger, "memory.reflection.stored",
                                 f"Stored learning reflection: {reflection['lesson']} ({reflection['category']})")

                    # Save to disk for debugging/backup
                    await self._save_reflection_to_disk(reflections, memories[0]["metadata"])

                except Exception as e:
                    log_error(self.logger, f"Failed in reflection of conversation {conv_id}: {str(e)}", exc_info=e)
                    continue
            
            log_event(self.logger, "memory.reflection.complete",
                     f"Completed memory reflection for {len(conversation_memories)} conversations")
             
        except Exception as e:
            log_error(self.logger, "Failed to process reflections", exc_info=e)
        finally:
            if self.status != AgentStatus.SHUTTING_DOWN:
                await self.set_status(self._previous_status)

    async def _run_learning_subroutine(self, category: str) -> None:
        """Run the learning subroutine."""
        #TODO: Run through all of the feedback and sort into categories related to different system prompts.
        #TODO: Do the same thing with memory lessons.
        #TODO: Assemble all the feedback/lessons in a category and turn it into a conversation.
        #TODO: Feed the conversation along with the old versions of the corresponding system prompt and its output 
        #TODO: schema into the LLM.
        #TODO: the LLM will return a new version of the system prompt using the same output schema.
        #TODO: update the system prompt.

        #TODO: Need to make sure these all link up ... some of them like tools will need to behave differently..
        

        output_schema = "No schema provided"
        match category:
            case "tools":
                pass
            case "agentic structure":
                pass            
            case "giving feedback":
                existing_prompt = self._give_feedback_prompt
            case "memory reflection":
                existing_prompt = self._reflect_memories_prompt
                output_schema = self._reflect_memories_schema
            case "long term memory transfer":
                existing_prompt = self._xfer_long_term_prompt
                output_schema = self._xfer_long_term_schema
            case "thought loop":
                existing_prompt = self._thought_loop_prompt
            case "reasoning":
                existing_prompt = self._reasoning_prompt
            case "self improvement":
                existing_prompt = self._self_improvement_prompt
            case "insight":
                pass

        #TODO: Might be useful to allow the agent to update the descriptions in the output schema. 
        # Especially if we find that the agents are having trouble adhering to the schema.
        combined_prompt = f"{self._self_improvement_prompt}\n\nExisting Prompt:\n{existing_prompt}\n\nOutput Schema:\n{output_schema}\n"
        
        learning_opps = await self.memory.retrieve(
                query="",
                collection_names=["reflections"],
                where={"metadata.category": category}
            )

        consolidated_content = f"## {category.title()} Learning Opportunities"
        num = 1
        for opp in learning_opps:
            consolidated_content += f"\n\n#{num}. {opp['content']}\nImportance: {opp['importance']}\nThoughts: {opp['thoughts']}\n\n"
            num += 1

        improved_prompt_response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": combined_prompt},
                    {"role": "user", "content": f"Memory content:\n{consolidated_content}"}
                ],
                response_format={ "type": "json_object" }
            )
        
        improved_json = json.loads(improved_prompt_response.choices[0].message.content)
        improved_prompt = improved_json["prompt"]
        # thoughts = improved_json["thoughts"]
        
        self._update_prompt(category, improved_prompt)

    def _update_prompt(self, category: str, improved_prompt: str) -> None:
        """Update the system prompt and create a backup of the previous version.
        
        Args:
            category: The category of prompt being updated
            improved_prompt: The new improved prompt content
        """
        # Map category to file paths
        path_map = {
            "tools": None,  # TODO: Add path when implemented
            "agentic structure": None,  # TODO: Add path when implemented
            "giving feedback": self._give_feedback_module_path,
            "memory reflection": self._reflect_memories_module_path,
            "long term memory transfer": self._xfer_long_term_module_path,
            "thought loop": self._thought_loop_module_path,
            "reasoning": self._reasoning_module_path,
            "self improvement": self._self_improvement_module_path,
            "insight": None  # TODO: Add path when implemented
        }

        prompt_path = path_map.get(category)
        if not prompt_path:
            raise ValueError(f"No file path configured for category: {category}")

        try:
            # Create backup directory if it doesn't exist
            backup_dir = self.prompt_path / "backups" / category
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp for backup file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{prompt_path.stem}_{timestamp}.md"

            with FileLock(f"{prompt_path}.lock"):
                # Create backup of current prompt
                if prompt_path.exists():
                    prompt_path.rename(backup_path)
                
                # Write new prompt
                prompt_path.write_text(improved_prompt, encoding="utf-8")

            # Update the corresponding module variable
            match category:
                case "giving feedback":
                    self._give_feedback_prompt = improved_prompt
                case "memory reflection":
                    self._reflect_memories_prompt = improved_prompt
                case "long term memory transfer":
                    self._xfer_long_term_prompt = improved_prompt
                case "thought loop":
                    self._thought_loop_prompt = improved_prompt
                case "reasoning":
                    self._reasoning_prompt = improved_prompt
                case "self improvement":
                    self._self_improvement_prompt = improved_prompt

            self.logger.info(f"Updated {category} prompt and created backup at {backup_path}")

        except Exception as e:
            self.logger.error(f"Failed to update {category} prompt: {str(e)}")
            raise

    async def _save_reflection_to_disk(self, reflections: List[Dict], original_metadata: Dict) -> None:
        """Save learning reflections to disk for debugging/backup."""
        try:
            reflection_dump_path = self.files_path / f"{self.config.agent_name}_reflection_dump.md"
            with FileLock(f"{reflection_dump_path}.lock"):
                with open(reflection_dump_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n## Learning Reflections Generated {datetime.now().isoformat()}\n")
                    f.write("### Original Memory Metadata\n")
                    for key, value in original_metadata.items():
                        f.write(f"{key}: {value}\n")
                    
                    f.write("\n### Extracted Learnings\n")
                    for reflection in reflections:
                        f.write(f"\n#### {reflection['category'].title()} (importance: {reflection['importance']})\n")
                        f.write(f"Lesson: {reflection['lesson']}\n")
                        f.write(f"Thoughts: {reflection['thoughts']}\n")
                    f.write("\n---\n")
                
            log_event(self.logger, "reflection.dumped", 
                     f"Saved {len(reflections)} learning reflections to disk")
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
                where={"metadata.conversation_id": conversation_id}
            )
            
            if not memories:
                self.logger.info(f"No memories found for conversation {conversation_id} in {collection_name}")
                return

            # Delete memories
            for memory in memories:
                await self.memory.delete(
                    collection_name=collection_name,
                    where={"metadata.memory_id": memory["metadata"].get("memory_id")}
                )
            
            self.logger.info(f"Cleaned up {len(memories)} memories from conversation {conversation_id} in {collection_name}")

        except Exception as e:
            self.logger.error(f"Failed to clean up conversation {conversation_id}: {str(e)}")

    async def start_socialising(self) -> Dict[str, Any]:
        """Start socialising with another agent."""
        return await start_socialising_impl(self)

    async def process_queue(self):
        """Memetic Agent process for processing any pending requests in the queue"""
        return await process_queue_impl(self)

    async def process_message(self, content: str, sender: str, prompt: PromptModel, conversation_id: str) -> str:
        """Memetic Agent process for processing a message"""
        if prompt is not None:
            return await process_social_message_impl(self, content, sender, prompt, conversation_id)
        else:
            return await process_message_impl(self, content, sender, conversation_id)

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
                    await self.set_status(AgentStatus.IDLE, "start")

#TODO: This agent needs to eb able to call up things liek its architecture or curent prompts 
# so that it can use them when reflecting on memories.

#TODO: This agent needs to be able to call up its current system prompt and use it when reflecting on memories.