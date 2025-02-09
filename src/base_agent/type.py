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

        # System Prompts
        self._system_prompt = "You are a helpful AI assistant"
        self._give_feedback_prompt = "Placeholder for giving feedback"
        self._thought_loop_prompt = "Placeholder for deciding whether to continue or stop"
        self._xfer_long_term_prompt = "Placeholder for transferring long term memories"
        self._xfer_feedback_prompt = "Placeholder for transferring feedback"
        self._reflect_feedback_prompt = "Placeholder for reflecting on feedback"
        self._evaluator_prompt = "Placeholder for evaluating prompts"
        self._reasoning_prompt = "Placeholder for reasoning module"
        self._reflect_memories_prompt = "Placeholder for reflecting on memories"
        self._self_improvement_prompt = "Placeholder for self improvement"

        # System Prompt Schemas - schemas are in json format but stored as strings
        self._xfer_long_term_schema: str = ""
        self._reflect_memories_schema: str = ""
        self._self_improvement_schema: str = ""
        self._give_feedback_schema: str = ""

        # System Prompt Confidence Scores (Floating Point between 0 and 10)
        self._prompt_confidence_scores = {
            "system": 0.0,
            "give_feedback": 0.0,
            "thought_loop": 0.0,
            "xfer_long_term": 0.0,
            "xfer_feedback": 0.0,
            "reflect_feedback": 0.0,
            "evaluator": 0.0,
            "reasoning": 0.0,
            "reflect_memories": 0.0,
            "self_improvement": 0.0
        }

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
        """Async initialization to load all required files. Run from server.py"""
        raise NotImplementedError("Subclasses must implement initialize")
    #TODO: the prompt_mapping needs to be kept in sync with both individual agents prompts and the master list of prompts
    def get_prompt_mapping(self) -> dict[str, str]:
        """Returns a mapping of prompt names to their current values."""
        return {
            "system_prompt": self._system_prompt,
            "reasoning_prompt": self._reasoning_prompt,
            "give_feedback_prompt": self._give_feedback_prompt,
            "reflect_feedback_prompt": self._reflect_feedback_prompt,
            "reflect_memories_prompt": self._reflect_memories_prompt,
            "self_improvement_prompt": self._self_improvement_prompt,
            "thought_loop_prompt": self._thought_loop_prompt,
            "xfer_long_term_prompt": self._xfer_long_term_prompt,
            "evaluator_prompt": self._evaluator_prompt
        }

    def get_prompt_schema(self, prompt_name: str) -> str | None:
        """Returns the schema for a given prompt name."""
        if prompt_name == "xfer_long_term_prompt":
            return self._xfer_long_term_schema
        elif prompt_name == "reflect_memories_prompt":
            return self._reflect_memories_schema
        elif prompt_name == "self_improvement_prompt":
            return self._self_improvement_schema
        elif prompt_name == "give_feedback_prompt":
            return self._give_feedback_schema
        else:
            return None


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