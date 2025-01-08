from typing import List, Dict, Any, Optional, Set
from enum import Enum, IntEnum
from dataclasses import dataclass
from pydantic import BaseModel, Field
from datetime import datetime

class ToolCall(BaseModel):
    """Represents a tool call made by the assistant."""
    id: str
    type: str = "function"
    function: Dict[str, Any]

class Message(BaseModel):
    """Represents a chat message in the conversation.
    
    Attributes:
        role: The role of the message sender (system, user, assistant, function)
        content: The message content
        name: Optional name for function calls
        tool_calls: Optional list of tool calls made by the assistant
        tool_call_id: Optional tool call ID for tool response messages
        sender: Optional sender for function calls
        receiver: Optional receiver for function calls
        timestamp: Optional timestamp for function calls
    """
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Name for function calls")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls made by assistant")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID for tool response messages")
    sender: Optional[str] = Field(None, description="Sender for function calls")
    receiver: Optional[str] = Field(None, description="Receiver for function calls")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    class Config:
        frozen = False  # Make messages immutable

    def dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for API."""
        result = {"role": self.role, "content": self.content}
        if self.name is not None:
            result["name"] = self.name
        if self.tool_calls is not None:
            result["tool_calls"] = [tc.dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        if self.sender is not None:
            result["sender"] = self.sender
        if self.receiver is not None:
            result["receiver"] = self.receiver
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        return result

class AgentStatus(IntEnum):
    IDLE = 0
    QUEUE_PROCESSING = 1
    MESSAGE_RECEIVED = 2
    MESSAGE_PROCESSING = 3
    TOOL_EXECUTING = 4
    WAITING_RESPONSE = 5
    LEARNING = 6
    MEMORISING = 7
    SOCIALISING = 8
    SHUTTING_DOWN = 9

    @classmethod
    def get_valid_transitions(cls, current_status: 'AgentStatus') -> Set['AgentStatus']:
        """Define valid status transitions."""
        VALID_TRANSITIONS = {
            cls.IDLE: {cls.MESSAGE_RECEIVED, cls.LEARNING, 
                      cls.MEMORISING, cls.SOCIALISING, cls.SHUTTING_DOWN},
            cls.MESSAGE_RECEIVED: {cls.QUEUE_PROCESSING, cls.WAITING_RESPONSE, cls.SHUTTING_DOWN},
            cls.QUEUE_PROCESSING: {cls.IDLE, cls.WAITING_RESPONSE, cls.MESSAGE_PROCESSING, cls.SHUTTING_DOWN},
            cls.MESSAGE_PROCESSING: {cls.QUEUE_PROCESSING, cls.TOOL_EXECUTING, 
                                   cls.WAITING_RESPONSE, cls.IDLE, cls.SHUTTING_DOWN},
            cls.TOOL_EXECUTING: {cls.MESSAGE_PROCESSING, cls.WAITING_RESPONSE, cls.IDLE,
                                cls.SHUTTING_DOWN},
            cls.WAITING_RESPONSE: {cls.QUEUE_PROCESSING, cls.MESSAGE_PROCESSING, cls.TOOL_EXECUTING, 
                                 cls.IDLE, cls.SHUTTING_DOWN},
            cls.LEARNING: {cls.IDLE, cls.SHUTTING_DOWN},
            cls.MEMORISING: {cls.IDLE, cls.LEARNING, cls.SHUTTING_DOWN},
            cls.SOCIALISING: {cls.IDLE, cls.WAITING_RESPONSE, cls.SHUTTING_DOWN},
            cls.SHUTTING_DOWN: set()
        }
        return VALID_TRANSITIONS.get(current_status, set())

@dataclass
class Request:
    """Request model for agent communication"""
    id: str
    sender: str
    content: str
    timestamp: str
    response: Optional[str] = None
