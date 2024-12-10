from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    """
    Represents a query request to an agent.
    """
    question: str = Field(
        ...,
        description="The question or prompt to be processed by the agent",
        examples=["What is the current market analysis?"]
    )
    max_turns: int | None = Field(
        default=5,
        description="Maximum number of conversation turns allowed",
        ge=1,
        le=10
    )
    debug: bool | None = Field(
        default=False,
        description="Enable debug mode for detailed processing information"
    )

class QueryResponse(BaseModel):
    """
    Represents a response to a query request.
    """
    response: str = Field(
        ...,
        description="The agent's response to the query",
        examples=["Based on current market analysis..."]
    )
    status: str = Field(
        default="success",
        description="Status of the query processing",
        enum=["success", "error", "pending"],
        examples=["success"]
    )
    error: str | None = Field(
        default=None,
        description="Error message if processing failed"
    )

class APIMessage(BaseModel):
    """
    Defines the structure for inter-agent communication messages.
    """
    sender: str = Field(
        ...,
        description="Identifier of the sending agent",
        examples=["research_agent"]
    )
    receiver: str = Field(
        ...,
        description="Identifier of the receiving agent",
        examples=["analysis_agent"]
    )
    content: str = Field(
        ...,
        description="Message content to be transmitted",
        examples=["Please analyze this market data..."]
    )
    timestamp: str = Field(
        ...,
        description="ISO format timestamp of message creation",
        examples=["2024-03-15T14:30:00.000Z"]
    )
    message_type: str = Field(
        default="text",
        description="Type of message being sent",
        enum=["text", "command", "data", "status"]
    )
    conversation_id: str = Field(
        ...,
        description="Unique identifier for the conversation thread",
        examples=["conv_123456789"]
    )

    @classmethod
    def create(cls, sender: str, receiver: str, content: str, conversation_id: str, message_type: str = "text") -> "APIMessage":
        """
        Factory method to create a message with current timestamp.
        
        Args:
            sender: Identifier of the sending agent
            receiver: Identifier of the receiving agent
            content: Message content
            conversation_id: Unique conversation identifier
            message_type: Type of message (default: "text")
            
        Returns:
            APIMessage: Newly created message instance
        """
        return cls(
            sender=sender,
            receiver=receiver,
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=conversation_id,
            message_type=message_type
        )

class AgentResponse(BaseModel):
    """
    Represents a response from an agent operation.
    """
    success: bool = Field(
        ...,
        description="Indicates if the operation was successful"
    )
    message: str = Field(
        ...,
        description="Response message or error description"
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional additional data returned by the operation"
    )

class FeedbackMessage(BaseModel):
    """
    Enables performance feedback exchange between agents.
    """
    sender: str = Field(
        ...,
        description="Identifier of the agent providing feedback",
        examples=["evaluation_agent"]
    )
    receiver: str = Field(
        ...,
        description="Identifier of the agent receiving feedback",
        examples=["research_agent"]
    )
    conversation_id: str = Field(
        ...,
        description="Identifier of the conversation being evaluated",
        examples=["conv_123456789"]
    )
    score: int = Field(
        ...,
        description="Numerical evaluation score",
        ge=0,
        le=10,
        examples=[8, 5, 10]
    )
    feedback: str = Field(
        ...,
        description="Detailed feedback message",
        examples=["Response was comprehensive but could be more concise"]
    )
    timestamp: str = Field(
        ...,
        description="ISO format timestamp of feedback creation",
        examples=["2024-03-15T14:30:00.000Z"]
    )
