import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from openai import AsyncOpenAI
import os
import sys
import importlib.util
from pathlib import Path
import httpx
import atexit
import signal
import traceback
import uuid
from fastapi import HTTPException
import inspect
from chromadb import PersistentClient
from filelock import FileLock
from icecream import ic
import contextvars
import threading

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parents[2]))

from .type import Agent
from .models import Message, ToolCall, AgentStatus, Request
from .config import AgentConfig
from api_server.models.api_models import APIMessage
from src.log_config import setup_logger, log_tool_call, log_agent_message, log_error, log_event
from src.memory.memory_manager import MemoryManager
from src.base_agent.tool_manager import ToolManager, ToolError
from src.base_agent.communication_module.receive_message_impl import receive_message_impl
from src.base_agent.communication_module.process_message_impl import process_message_impl
from src.base_agent.communication_module.send_message_impl import send_message_impl
from src.base_agent.feedback_module.give_feedback import evaluate_response_impl, evaluate_and_send_feedback_impl
from src.base_agent.feedback_module.accept_feedback import receive_feedback_impl, process_feedback_impl
from src.base_agent.communication_module.process_queue_impl import process_queue_impl

class BaseAgent(Agent):  # Change to inherit from Agent
    def __init__(
        self,
        api_key: str,
        chroma_client: PersistentClient,
        config: Optional[AgentConfig] = None
    ):
        """Initialize the base agent with OpenAI client and configuration."""
       
        # Call parent class init first
        super().__init__(api_key, chroma_client, config)
        
        # Initialize logger before any other operations
        self.logger = setup_logger(
            name=self.config.agent_name,
            log_path=self.config.log_path,
            level=self.config.log_level,
            console_logging=self.config.console_logging
        )
        
        self._setup_logging()

        # Log initialization with process/thread info
        log_event(self.logger, "agent.init", 
                 f"Initializing {self.__class__.__name__} (PID: {os.getpid()}, Thread: {threading.current_thread().name})")

        # Load System Prompts
        self._system_prompt = self.config.system_prompt
        self._give_feedback_prompt = """Evaluate the quality of the following response. 
                Provide:
                1. A score from 0-10 (where 10 is excellent)
                2. Brief, constructive feedback (Include WHAT is good/bad, WHY it matters, and HOW to improve)
                
                Format your response as a JSON object with 'score' and 'feedback' fields."""
        self._thought_loop_prompt = (
                    "Based on the conversation so far, determine if a complete response has been provided "
                    "and if the initial query has been fully addressed. If complete, respond with exactly 'STOP'. "
                    "Otherwise, respond with clear instructions on what steps to take next, starting with 'We still need to...' "
                    "and ending with '...Please continue.'"
                )
        self._xfer_long_term_prompt = """Analyze this memory and extract information in SPO (Subject-Predicate-Object) format.
                            Format your response as a JSON object with this structure:
                            {
                                "memories": [
                                    {
                                        "content": "SPO statement in 1-3 sentences",
                                        "type": "fact|decision|preference|pattern",
                                        "tags": ["#tag1", "#tag2", ...],
                                        "importance": 1-10
                                    }
                                ]
                            }
                            
                            Include type as a tag and add other relevant topic/context tags.
                            Keep statements clear and concise.
                            
                            Example Input: 'During the community meeting last Tuesday, Sarah Johnson presented a detailed proposal for a new youth center 
                            downtown. The proposal included a $500,000 budget plan and received strong support from local business owners. Several attendees 
                            raised concerns about parking availability, but Sarah addressed these by suggesting a partnership with the nearby church for 
                            additional parking spaces during peak hours.'

                            Example Output:
                            {
                                "memories": [
                                    {
                                        "content": "Sarah Johnson (S) presented (P) a youth center proposal at the community meeting (O). The proposal 
                                        outlined a $500,000 budget and garnered support from local businesses.",
                                        "type": "fact",
                                        "tags": ["#community_development", "#youth_services", "#proposal", "#budget"],
                                        "importance": 8
                                    }
                                ]
                            }
                            """
        self._xfer_feedback_prompt = """Analyze this feedback and extract key insights.
                        Format your response as a JSON object with this structure:
                        {
                            "insights": [
                                {
                                    "content": "Clear statement of insight/learning",
                                    "category": "strength|weakness|improvement|pattern",
                                    "tags": ["#relevant_tag1", "#relevant_tag2"],
                                    "importance": 1-10,
                                    "action_items": ["specific action to take", ...]
                                }
                            ]
                        }"""
        self._reflect_feedback_prompt = """You are an analytical assistant helping to reflect on feedback received.
                Consider patterns, trends, and areas for improvement. Focus on actionable insights
                and concrete steps for improvement."""

        # Initialize tool manager
        self.tool_mod = ToolManager(self)
        
        # Add and register internal tools
        self.internal_tools["send_message"] = self.send_message
        self.internal_tools["search_memory"] = self.search_memory
        for tool_name, tool_func in self.internal_tools.items():
            self.tool_mod.register(tool_func)
            
        # Load enabled external tools
        if self.config.enabled_tools:
            self._load_tool_definitions()

        # Initialize memory manager
        self.memory = MemoryManager(
            agent_name=self.config.agent_name,
            logger=self.logger,
            chroma_client=chroma_client
        )
        
        # Register shutdown handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    async def initialize(self) -> None:
        """Implemented in MemeticAgent only. Run from server.py"""
        pass

    def _setup_logging(self) -> None:
        """Configure logging if debug is enabled."""
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
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(formatter)
                    self.logger.addHandler(console_handler)
                    
                self.logger.setLevel(logging.DEBUG)
                
            self.logger.info("BaseAgent initialized")

        log_event(self.logger, "agent.registered", f"Initialized {self.config.agent_name}")

    def _load_tool_definitions(self) -> None:
        """Load tool definitions from JSON files."""
        for tool_name in self.config.enabled_tools:
            try:
                tool_path = self.config.tools_path / f"{tool_name}.json"
                if not tool_path.exists():
                    raise ToolError(f"Tool definition not found: {tool_path}")

                log_event(self.logger, "tool.loading", f"Loading tool definition: {tool_path}", level="DEBUG")
                with open(tool_path) as f:
                    tool_def = json.load(f)
                    self.tool_mod.register(tool_def)
                    
            except Exception as e:
                log_event(self.logger, "tool.error", f"Failed to load tool {tool_name}: {str(e)}", level="ERROR")

    def register_tools(self, tools: List[Dict[str, Any]]) -> None:
        """Register multiple tools/functions that the agent can use.
        
        Args:
            tools: List of tool definitions to register
            
        Raises:
            ValueError: If any tool definition is invalid
        """
        for tool in tools:
            self.tool_mod.register(tool)
            
    async def process_message(self, content: str, sender: str, conversation_id: str) -> str:
        return await process_message_impl(self, content, sender, conversation_id)

    async def _continue_or_stop(self, messages: List[Message]) -> str:
        """Evaluate message content to determine if we should continue or stop.
        
        Args:
            messages: List of conversation messages to evaluate
            
        Returns:
            "STOP" if conversation should end, or a rephrased continuation message
        """
        try:
            # Create prompt to evaluate conversation state
            system_prompt = Message(
                role="user" if self.config.model == "o1-mini" else "developer" if self.config.model == "o3-mini" else "system",
                content=self._thought_loop_prompt
            )
            
            # OLD APPROACH - Swap system prompt with thought loop prompt
            #messages = messages[1:]  # Remove first message
            #messages.insert(0, system_prompt)  # Add new system prompt at start
            
            # NEW APPROACH - Add thought loop prompt to the end of the message list
            messages.append(system_prompt)
            
            # Make LLM call to evaluate
            raw_response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[msg.dict() for msg in messages],
                **({"temperature": 0.7} if self.config.model not in ["o1-mini", "o3-mini"] else {}),
                **({"reasoning_effort": self.config.reasoning_effort} if self.config.model == "o3-mini" else {})
            )
            
            response = raw_response.choices[0].message.content.strip()

            log_event(self.logger, "conversation.evaluation", "Starting continue/stop evaluation", level="DEBUG")
            log_event(self.logger, "conversation.agent", f"Agent: {self.config.agent_name}", level="DEBUG")
            
            # Log messages in a condensed format
            for msg in messages:
                msg_details = {k: v for k, v in {
                    'role': msg.role,
                    'content': msg.content,
                    'sender': msg.sender,
                    'receiver': msg.receiver,
                    'name': msg.name,
                    'tool_calls': msg.tool_calls
                }.items() if v}  # Only include non-empty fields
                
                log_event(self.logger, "conversation.message", f"Message details: {msg_details}", level="DEBUG")
            
            log_event(self.logger, "conversation.response", f"LLM Response: {response}", level="DEBUG")       
            
            # Add human oversight
            print("\n|------------HUMAN OVERSIGHT------------|\n")
            print("\nProposed response:", response)
            human_input = input("\nAccept this response? (y/n, or type new response): ").strip()
            
            if human_input.lower() == 'y':
                pass  # Keep existing response
            elif human_input.lower() == 'n':
                response = "STOP"  # Force conversation to end
            else:
                # Use human's custom response
                response = human_input
            # Log the continuation decision
            if "STOP" in response:
                log_event(self.logger, "conversation.complete", "Conversation marked as complete")
            else:
                log_event(self.logger, "conversation.continue", f"Continuing conversation: {response}")
                
            return response
            
        except Exception as e:
            log_error(self.logger, "Error evaluating conversation continuation", exc_info=e)
            return "STOP"  # Default to stopping on error

        return response

    async def receive_message(self, message: APIMessage) -> str:
        return await receive_message_impl(self, message)

    async def send_message(self, receiver: str, content: str) -> Dict[str, Any]:
        """Send a message via API to another agent registered in the directory service."""
        return await send_message_impl(self, receiver, content)

    async def _evaluate_and_send_feedback(
        self,
        receiver: str,
        conversation_id: str,
        response_content: str
    ) -> None:
        """Evaluate response quality and send feedback to the agent."""
        return await evaluate_feedback_impl(self, receiver, conversation_id, response_content)

    async def _evaluate_response(self, response_content: str) -> Tuple[int, str]:
        """Evaluate response quality using LLM."""
        return await evaluate_response_impl(self, response_content)

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
        try:
            log_event(self.logger, "memory.load.start", "Beginning memory load process")
            
            # Initialize empty containers
            self.conversations = {}
            self.old_conversation_list = {}
            
            collection_names = self.memory.collections.keys()

            # Ensure collections are initialized before proceeding
            for collection_name in collection_names:
                if collection_name not in self.memory.collections:
                    raise ValueError(f"Collection '{collection_name}' not found. Available collections: {list(self.memory.collections.keys())}")

            # Part 1: Load conversations from short-term memory
            short_term_collection = self.memory._get_collection("short_term")
            results = short_term_collection.get()
            
            if not results["documents"]:
                log_event(self.logger, "memory.load.complete", 
                         "Short-term memory collection is empty - starting fresh")
                return
            
            # Group results by conversation_id
            conversation_groups = {}
            for doc, metadata in zip(results["documents"], results["metadatas"]):
                conv_id = metadata.get("conversation_id")
                if conv_id:
                    if conv_id not in conversation_groups:
                        conversation_groups[conv_id] = {
                            "content": [],
                            "participants": set(),
                            "timestamps": []
                        }
                    conversation_groups[conv_id]["content"].append(doc)
                    if "participants" in metadata:
                        conversation_groups[conv_id]["participants"].update(
                            metadata["participants"].split(",")
                        )
                    if "timestamp" in metadata:
                        conversation_groups[conv_id]["timestamps"].append(metadata["timestamp"])

            # Convert grouped content into conversations
            loaded_conversations = 0
            for conv_id, group in conversation_groups.items():
                try:
                    # Start with system prompt
                    messages = [Message(role="user" if self.config.model == "o1-mini" else "developer" if self.config.model == "o3-mini" else "system", content=self._system_prompt)]
                    
                    # Combine all content for this conversation
                    combined_content = "\n".join(group["content"])
                    
                    # Parse the combined content into messages
                    message_chunks = combined_content.split("\n")
                    for chunk in message_chunks:
                        if chunk.strip():
                            # Try to parse role and content from chunk
                            if ": " in chunk:
                                role, content = chunk.split(": ", 1)
                                # Convert role to standard format
                                role = role.lower()
                                if role not in ["system", "user", "assistant", "tool"]:
                                    role = "user"
                            else:
                                # Default to user role if format is unclear
                                role = "user"
                                content = chunk
                                
                            messages.append(Message(
                                role=role,
                                content=content,
                                timestamp=min(group["timestamps"]) if group["timestamps"] else datetime.now().isoformat()
                            ))
                
                    self.conversations[conv_id] = messages
                    loaded_conversations += 1
                    log_event(self.logger, "memory.loaded", 
                             f"Loaded conversation {conv_id} with {len(messages)} messages")
                    
                except Exception as e:
                    log_error(self.logger, 
                             f"Error parsing conversation {conv_id}: {str(e)}")
                    continue

            # Part 2: Load old conversations list
            old_conversations_file = self.files_path / "old_conversations.json"
            if old_conversations_file.exists():
                try:
                    with open(old_conversations_file, "r") as f:
                        self.old_conversation_list = json.load(f)
                    log_event(self.logger, "memory.loaded", 
                             f"Loaded {len(self.old_conversation_list)} old conversations")
                except json.JSONDecodeError as e:
                    log_error(self.logger, 
                             f"Error loading old conversations: {str(e)}")
                    # Initialize empty if file is corrupted
                    self.old_conversation_list = {}
            else:
                # Initialize empty if file doesn't exist
                self.old_conversation_list = {}

            log_event(self.logger, "memory.load.complete", 
                     f"Successfully loaded {loaded_conversations} conversations")

        except Exception as e:
            log_error(self.logger, "Failed to load memory", exc_info=e)
            # Initialize empty containers on error
            self.conversations = {}
            self.old_conversation_list = {}

    async def _save_memory(self) -> None:
        """Save new messages from conversations to memory store."""
        try:
            # Get the latest timestamp for each conversation from short-term memory
            short_term = self.memory._get_collection("short_term")
            latest_timestamps = {}
            
            for conversation_id in self.conversations:
                # Changed from get() with order_by to get() with where clause
                results = short_term.get(
                    where={"conversation_id": conversation_id}
                )
                
                # Manually find the latest timestamp from results
                if results["metadatas"]:
                    timestamps = [
                        metadata.get("timestamp") 
                        for metadata in results["metadatas"] 
                        if metadata.get("timestamp")
                    ]
                    if timestamps:
                        latest_timestamps[conversation_id] = max(timestamps)
                    else:
                        latest_timestamps[conversation_id] = None
                else:
                    latest_timestamps[conversation_id] = None

            # Save only new messages for each conversation
            for conversation_id, conversation in self.conversations.items():
                latest_timestamp = latest_timestamps.get(conversation_id)
                
                # Filter messages that are newer than the latest saved timestamp
                new_messages = []
                if latest_timestamp:
                    new_messages = [
                        msg for msg in conversation 
                        if msg.role != "system" and msg.timestamp > latest_timestamp
                    ]
                else:
                    # If no previous messages, save all except system message
                    new_messages = [msg for msg in conversation if msg.role != "system"]

                if not new_messages:
                    continue

                # Format new messages for storage
                content = "\n".join([
                    f"{msg.role}: {msg.content}" 
                    for msg in new_messages
                ])
                
                # Get all participants from new messages
                participants = set(
                    getattr(msg, 'sender', None) or msg.role 
                    for msg in new_messages
                )
                participants_str = ",".join(sorted(participants))
                
                metadata = {
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "participants": participants_str
                }
                
                try:
                    await self.memory.store(
                        content=content,
                        collection_name="short_term",
                        metadata=metadata
                    )
                    log_event(self.logger, "memory.saved", 
                             f"Saved {len(new_messages)} new messages for conversation {conversation_id}")
                    
                    # Save memory dump to disk
                    memory_dump_path = self.files_path / f"{self.config.agent_name}_short_term_memory_dump.md"
                    try:
                        with FileLock(f"{memory_dump_path}.lock"):
                            with open(memory_dump_path, "a", encoding="utf-8") as f:
                                # Add timestamp header
                                f.write(f"\n\n## Memory Entry {datetime.now().isoformat()}\n")
                                for key, value in metadata.items():
                                    f.write(f"{key}: {value}\n")
                                f.write("\n")
                                f.write(content)
                                f.write("\n---\n")
                        log_event(self.logger, "memory.dumped", 
                                f"Saved new messages for conversation {conversation_id} to disk")
                    except Exception as e:
                        log_error(self.logger,
                                f"Failed to save conversation {conversation_id} to disk: {str(e)}")
                    
                except Exception as e:
                    log_error(self.logger, 
                             f"Failed to save conversation {conversation_id}: {str(e)}")
                    
        except Exception as e:
            log_error(self.logger, "Failed to save memory", exc_info=e)

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

    async def start(self):
        """Start the agent's main processing loop and initialise memories"""
        self.logger.info(f"Starting {self.config.agent_name} processing loop")
        
        # Initialize and load memory collections before starting the processing loop
        await self._initialize_and_load_memory_collections(["short_term", "long_term", "feedback"])
        
        while not self._shutdown_event.is_set():
            try:
                await self.process_queue()
                await asyncio.sleep(0.1)  # Prevent CPU spinning
            except Exception as e:
                self.logger.error(f"Error in processing loop: {str(e)}")
                if self.status != AgentStatus.SHUTTING_DOWN:
                    await self.set_status(AgentStatus.AVAILABLE, "start")

    async def process_queue(self):
        """Base Agent process for processing any pending requests in the queue"""
        return await process_queue_impl(self)

    async def set_status(self, new_status: AgentStatus, trigger: str) -> None:
        """Update agent status with validation and logging."""
        async with self._status_lock:
            if new_status == self.status:
                return

            valid_transitions = AgentStatus.get_valid_transitions(self.status)
            
            if new_status not in valid_transitions:
                raise ValueError(
                    f"Invalid status transition from {self.status.name} to {new_status.name} caused by {trigger}"
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
                f"Status changed: {self._previous_status.name} -> {self.status.name} ({trigger})"
            )
            
            # Trigger memory consolidation when entering MEMORISING state
            if new_status == AgentStatus.MEMORISING:
                asyncio.create_task(self._transfer_to_long_term())

    async def _transfer_to_long_term(self, days_threshold: int = 0) -> None:
        """Transfer short-term memories into long-term storage using simplified SPO format.
        
        Args:
            days_threshold: Number of days worth of memories to keep in short-term. 
                           Memories older than this will be processed into long-term storage.
                           Default is 0 (process all memories).
        """
        try:
            log_event(self.logger, "agent.memorising", 
                     f"Beginning memory consolidation process (threshold: {days_threshold} days)")
            
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
                log_event(self.logger, "memory.error", 
                         f"No memories older than {days_threshold} days found for consolidation")
                await self.set_status(self._previous_status, "transfer to long term - no memories found")
                return
            
            # Process each memory individually
            for memory in short_term_memories:
                try:
                    # Extract structured information using LLM
                    response = await self.client.chat.completions.create(
                        model=self.config.submodel,
                        messages=[
                            {"role": "system", "content": self._xfer_long_term_prompt},
                            {"role": "user", "content": f"Memory to analyze:\n{memory['content']}"}
                        ],
                        response_format={ "type": "json_object" }
                    )
                    
                    structured_info = json.loads(response.choices[0].message.content)
                    
                    # Store each extracted memory
                    for mem in structured_info["memories"]:
                        # Format content with tags section
                        formatted_content = f"{mem['content']}\n\nMetaTags: {', '.join(mem['tags'])}"
                        
                        # Simplified metadata structure with only scalar values
                        metadata = {
                            "memory_id": str(uuid.uuid4()),
                            "original_timestamp": memory["metadata"].get("timestamp"),
                            "memory_type": mem["type"],
                            "importance": mem["importance"],
                            "primary_tag": mem["tags"][0] if mem["tags"] else "#untagged",
                            "timestamp": datetime.now().isoformat()
                        }

                        await self.memory.store(
                            content=formatted_content,
                            collection_name="long_term",
                            metadata=metadata
                        )
                        
                        log_event(self.logger, "agent.memorising",
                                 f"Processed memory {metadata['memory_id']} into long-term storage")

                        # Save to disk for debugging/backup
                        await self._save_memory_to_disk(formatted_content, metadata, "memory")

                except Exception as e:
                    log_error(self.logger, f"Failed to process memory: {str(e)}", exc_info=e)
                    continue
                
            await self._cleanup_memories(days_threshold, "short_term")
            
            log_event(self.logger, "memory.memorising.complete",
                     f"Completed memory consolidation for {len(short_term_memories)} memories")
                 
        except Exception as e:
            log_error(self.logger, "Failed to consolidate memories", exc_info=e)
        finally:
            if self.status != AgentStatus.SHUTTING_DOWN:
                await self.set_status(self._previous_status, "transfer to long term - complete")

    async def _save_memory_to_disk(self, structured_info: Dict, metadata: Dict, memory_type: str) -> None:
        """Save structured memory information to disk for debugging/backup.
        
        Args:
            structured_info: Dictionary containing the structured information to save
            metadata: Dictionary containing metadata about the memory
            memory_type: Type of memory ('memory' or 'feedback')
        """
        try:
            # Use different files for different memory types
            file_suffix = "feedback" if memory_type == "feedback" else "long_term"
            memory_dump_path = self.files_path / f"{self.config.agent_name}_{file_suffix}_memory_dump.md"
            
            with FileLock(f"{memory_dump_path}.lock"):
                with open(memory_dump_path, "a", encoding="utf-8") as f:
                    # Write header with timestamp
                    f.write(f"\n\n## {memory_type.title()} Entry {metadata['timestamp']}\n")
                    
                    # Write metadata section
                    f.write("### Metadata\n")
                    for key, value in metadata.items():
                        if isinstance(value, list):
                            value = ", ".join(value)
                        f.write(f"{key}: {value}\n")
                    
                    # Write content section based on memory type
                    if memory_type == "feedback":
                        f.write("\n### Feedback Insights\n")
                        if isinstance(structured_info, dict) and "insights" in structured_info:
                            for insight in structured_info["insights"]:
                                f.write(f"\n#### Insight\n")
                                f.write(f"Content: {insight['content']}\n")
                                f.write(f"Category: {insight['category']}\n")
                                f.write(f"Importance: {insight['importance']}\n")
                                f.write("Action Items:\n")
                                for item in insight['action_items']:
                                    f.write(f"- {item}\n")
                                f.write(f"Tags: {', '.join(insight['tags'])}\n")
                        else:
                            # Handle case where structured_info is a string
                            f.write(f"\n{structured_info}\n")
                    else:
                        # Handle regular memory entries
                        f.write("\n### Extracted Information\n")
                        if isinstance(structured_info, dict):
                            for category in ["facts", "decisions", "preferences", "patterns"]:
                                if structured_info.get(category):
                                    f.write(f"\n#### {category.title()}\n")
                                    for item in structured_info[category]:
                                        f.write(f"- {item}\n")
                        else:
                            # Handle case where structured_info is a string
                            f.write(f"\n{structured_info}\n")
                        
                        f.write("\n---\n")  # Add separator between entries

                    log_event(self.logger, f"{memory_type}.dumped", 
                             f"Saved {memory_type} {metadata.get('memory_id') or metadata.get('insight_id')} to disk")
                     
        except Exception as e:
            log_error(self.logger, 
                     f"Failed to save {memory_type} {metadata.get('memory_id') or metadata.get('insight_id')} to disk: {str(e)}")

    async def _cleanup_memories(self, days_threshold: int = 0, collection_name: str = "short_term") -> None:
        """Clean up old memories from specified collection after consolidation.
        
        Args:
            days_threshold: Number of days worth of memories to keep.
                           Memories older than this will be deleted.
                           Default is 0 (clean up all processed memories).
            collection_name: Name of collection to clean up.
                           Default is "short_term".
        """
        try:
            # Ensure days_threshold is an integer
            try:
                days_threshold = int(days_threshold)
            except (TypeError, ValueError):
                log_error(self.logger, f"Invalid days_threshold value: {days_threshold}. Using default of 0.")
                days_threshold = 0

            log_event(self.logger, "memory.cleanup.start", 
                     f"Starting cleanup of {collection_name} memories older than {days_threshold} days")
            
            # Retrieve all memories from specified collection
            old_memories = await self.memory.retrieve(
                query="",
                collection_names=[collection_name],
                n_results=1000
            )
            
            # Filter based on threshold
            cutoff_time = datetime.now() - timedelta(days=days_threshold)
            old_memories = [
                memory for memory in old_memories
                if datetime.fromisoformat(memory["metadata"].get("timestamp", "")) < cutoff_time
            ]
            
            # Delete old memories using the ChromaDB ID from metadata
            deleted_count = 0
            for memory in old_memories:
                chroma_id = memory["metadata"].get("chroma_id")
                
                # Only update old_conversation_list for short_term memories
                if collection_name == "short_term":
                    conversation_id = memory["metadata"].get("conversation_id")
                    if conversation_id:
                        self.old_conversation_list[conversation_id] = "placeholder_name"
                
                if chroma_id:
                    await self.memory.delete(chroma_id, collection_name)
                    deleted_count += 1
                
            log_event(self.logger, "memory.cleanup", 
                     f"Cleaned up {deleted_count} memories from {collection_name} collection")
             
        except Exception as e:
            log_error(self.logger, f"Failed to cleanup {collection_name} memories", exc_info=e)

    #TODO: Need to fix internal tools so that the descriptions and tool definitions work properly.
    async def search_memory(self, query: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """Use to search your memory using a 'query' and (optional) 'keywords'.
        
        Args:
            query: The search query string
            keywords: Optional list of keywords to help filter results
            
        Returns:
            Dict containing search results and metadata
        """
        try:
            # Create metadata filter if keywords provided
            metadata_filter = None
            if keywords:
                # Use $in operator instead of $contains for keyword matching
                # This will match if any of the keywords exactly match the content
                metadata_filter = {
                    "content": {"$in": keywords}
                }

            # Search both short and long term memory
            memories = await self.memory.retrieve(
                query=query,
                collection_names=["short_term", "long_term"],
                n_results=5,
                metadata_filter=metadata_filter
            )

            log_event(self.logger, "memory.searched", 
                     f"Memory search for '{query}' returned {len(memories)} results")

            return {
                "query": query,
                "results": memories,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            error_msg = f"Failed to search memory: {str(e)}"
            log_error(self.logger, error_msg, exc_info=e)
            return {
                "query": query,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }

    async def _process_feedback(self, days_threshold: int = 0) -> None:
        """Process and transfer feedback to long-term memory.
        
        Process is not active in BaseAgent, current implementation uses a separate
        LLM call per feedback item which is inefficient and expensive.
        
        """
        return await process_feedback_impl(self, days_threshold)

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

