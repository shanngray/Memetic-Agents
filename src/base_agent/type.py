from enum import Enum
from typing import Dict, Optional, Any, List
from datetime import datetime
import asyncio
import contextvars
from pathlib import Path

from openai import AsyncOpenAI
from chromadb import PersistentClient

from .models import Message, AgentStatus
from .config import AgentConfig
from src.log_config import setup_logger
from src.api_server.models.api_models import APIMessage
from .prompt_library import PromptLibrary, PromptEntry


class Agent:
    """Base class for all agent implementations.
    
    This class defines the core interface and common attributes that all agents must implement.
    It provides the foundational structure for agent behavior and state management.
    """
    
    def __init__(
        self,
        api_key: str,
        chroma_client: PersistentClient,
        config: Optional[AgentConfig] = None
    ):
        """Initialize the base agent.

        Args:
            api_key: OpenAI API key for LLM access
            chroma_client: ChromaDB client for memory management
            config: Optional agent configuration, uses defaults if not provided
        """
        self.config = config or AgentConfig()
        self.client = AsyncOpenAI(api_key=api_key)
        self.logger = setup_logger(self.config.agent_name)

        # Core state attributes
        self.status = AgentStatus.AVAILABLE
        self._previous_status = AgentStatus.AVAILABLE
        self._status_history: List[Dict[str, Any]] = []
        self._status_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

        # Replace individual prompt attributes with PromptLibrary instance
        self.prompt = PromptLibrary()

        # Request handling
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.waiting_for: Optional[str] = None # TODO: Is this even used????
        
        # Tool management
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.internal_tools: Dict[str, Any] = {}
        
        # Conversation management
        self.conversations: Dict[str, List[Message]] = {}
        self.old_conversation_list: Dict[str, str] = {}
        self.current_conversation_id = contextvars.ContextVar('current_conversation_id', default="") #TODO: Changed to empty string - watch for bugs
        
        # File management
        self.files_path = Path("agent_files") / self.config.agent_name
        self.files_path.mkdir(parents=True, exist_ok=True)
        self.agent_directory = None

    async def initialize(self) -> None:
        """Async initialization to load all required files. Run from main.py"""
        raise NotImplementedError("Subclasses must implement initialize")
    #TODO: the prompt_mapping needs to be kept in sync with both individual agents prompts and the master list of prompts
    def get_prompt_mapping(self) -> dict[str, str]:
        """Returns a mapping of prompt names to their current values."""
        return {
            name: getattr(self.prompt, prompt_type).content or ""
            for name, prompt_type in {
                "system_prompt": "system",
                "reasoning_prompt": "reasoning",
                "give_feedback_prompt": "give_feedback",
                "reflect_feedback_prompt": "reflect_feedback",
                "reflect_memories_prompt": "reflect_memories",
                "self_improvement_prompt": "self_improvement",
                "thought_loop_prompt": "thought_loop",
                "xfer_long_term_prompt": "xfer_long_term",
                "evaluator_prompt": "evaluator"
            }.items()
        }

    def get_prompt_schema(self, prompt_name: str) -> str | None:
        """Returns the schema for a given prompt name."""
        prompt_map = {
            "xfer_long_term_prompt": self.prompt.xfer_long_term.schema,
            "reflect_memories_prompt": self.prompt.reflect_memories.schema,
            "self_improvement_prompt": self.prompt.self_improvement.schema,
            "give_feedback_prompt": self.prompt.give_feedback.schema
        }
        return prompt_map.get(prompt_name)

    async def process_message(self, content: str, sender: str, conversation_id: str) -> str:
        """Process an incoming message.

        Args:
            content: Message content to process
            sender: ID of the sending agent
            conversation_id: ID of the conversation thread

        Returns:
            str: Response message content
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement process_message")

    async def receive_message(self, message: APIMessage) -> str:
        """Handle an incoming message from another agent.

        Args:
            message: APIMessage containing sender, content, and conversation_id

        Returns:
            str: Response message content
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement receive_message")

    async def send_message(self, receiver: str, content: str) -> Dict[str, Any]:
        """Send a message to another agent.

        Args:
            receiver: ID of the receiving agent
            content: Message content to send

        Returns:
            Dict[str, Any]: Response data from the receiving agent
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement send_message")

    async def _evaluate_and_send_feedback(
        self,
        receiver: str,
        conversation_id: str,
        response_content: str
    ) -> None:
        """Evaluate response quality and send feedback to the agent."""
        raise NotImplementedError("Subclasses must implement _evaluate_and_send_feedback")

    async def shutdown(self) -> None:
        """Gracefully shutdown the agent.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement shutdown")

    async def start(self) -> None:
        """Start the agent's main processing loop and initialize memories.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement start")

    async def process_queue(self) -> None:
        """Process any pending requests in the queue.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement process_queue")

    async def _continue_or_stop(self, messages: List[Message]) -> str:
        """Evaluate message content to determine if we should continue or stop.
        
        Args:
            messages: List of conversation messages to evaluate
            
        Returns:
            "STOP" if conversation should end, or a rephrased continuation message
        """
        raise NotImplementedError("Subclasses must implement _continue_or_stop")

    async def set_status(self, new_status: AgentStatus, trigger: str) -> None:
        """Update agent status with validation and logging.
        
        Args:
            new_status: New status to set
            trigger: Description of what triggered the status change
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement set_status")

    async def search_memory(self, query: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """Search agent's memory using query and optional keywords.
        
        Args:
            query: The search query string
            keywords: Optional list of keywords to help filter results
            
        Returns:
            Dict[str, Any]: Search results and metadata
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement search_memory")

    async def _transfer_to_long_term(self, days_threshold: int = 0) -> None:
        """Transfer short-term memories into long-term storage.
        
        Args:
            days_threshold: Number of days worth of memories to keep in short-term
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _transfer_to_long_term")

    async def _cleanup_memories(self, days_threshold: int = 0, collection_name: str = "short_term") -> None:
        """Clean up old memories from specified collection.
        
        Args:
            days_threshold: Number of days worth of memories to keep
            collection_name: Name of collection to clean up
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _cleanup_memories")

    async def _save_memory_to_disk(self, structured_info: Dict, metadata: Dict, memory_type: str) -> None:
        """Save structured memory information to disk.
        
        Args:
            structured_info: Dictionary containing the structured information
            metadata: Dictionary containing metadata about the memory
            memory_type: Type of memory ('memory' or 'feedback')
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _save_memory_to_disk")

    async def _record_score(self, prompt_type: str, prompt_score: int, conversation_id: str, score_type: str) -> None:
        """Record a score for a prompt.
        
        Args:
            prompt_type: Type of prompt being evaluated
            prompt_score: Score for the prompt (0-10)
            conversation_id: ID of the conversation thread
            score_type: Type of score being recorded
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _record_score")

    async def _update_confidence_score(self, prompt_type: str, initial_friend_score: int, 
                                     updated_friend_score: int, self_eval_score: int) -> None:
        """Calculate and update confidence score for a prompt type based on interaction scores.
        
        Args:
            prompt_type: Type of prompt being scored
            initial_friend_score: Friend's initial evaluation score (0-10)
            updated_friend_score: Friend's evaluation of updated prompt (0-10)
            self_eval_score: Agent's self-evaluation score (0-10)
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _update_confidence_score")

    async def _evaluate_prompt(self, prompt_type: str, prompt: str) -> int:
        """Evaluate a prompt and update confidence scores.
        
        Args:
            prompt_type: Type of prompt being evaluated
            prompt: The prompt to evaluate
        
        Returns:
            int: The score for the prompt (0-10)

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _evaluate_prompt")

    def _calculate_updated_confidence_score(self, prompt_type: str, new_score: int) -> None:
        """Calculate the updated confidence score for the prompt.
        
        Args:
            prompt_type: Type of prompt being scored
            new_score: The new score for the prompt (0-10)
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement _calculate_updated_confidence_score")

    async def update_prompt_module(self, prompt_type: str, new_prompt: str) -> str:
        """Update a specific prompt module with backup functionality.
        
        Args:
            prompt_type: Type of prompt to update (reasoning, give_feedback, thought_loop, etc.)
            new_prompt: The new prompt content
        
        Returns:
            str: Success message
        
        Raises:
            ValueError: If prompt type is invalid or update fails
        """
        raise NotImplementedError("Subclasses must implement update_prompt_module")