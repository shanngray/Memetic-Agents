from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

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
    conversation_id: str = Field(
        ...,
        description="Unique identifier for the conversation thread",
        examples=["conv_123456789"]
    )

    @classmethod
    def create(cls, sender: str, receiver: str, content: str, conversation_id: str) -> "APIMessage":
        """
        Factory method to create a message with current timestamp.
        
        Args:
            sender: Identifier of the sending agent
            receiver: Identifier of the receiving agent
            content: Message content
            conversation_id: Unique conversation identifier
            
        Returns:
            APIMessage: Newly created message instance
        """
        return cls(
            sender=sender,
            receiver=receiver,
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=conversation_id
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

class PromptModel(BaseModel):
    prompt: str = Field(..., description="The prompt to be used")
    prompt_type: str = Field(..., description="The type of prompt")
    uuid: str = Field(..., description="The unique identifier for the prompt")
    timestamp: str = Field(..., description="The timestamp that the PromptModel was created")
    owner_agent_name: str = Field(..., description="The name of the agent that owns the prompt")
    status: str = Field(..., description="The status of the prompt") #may want to make this an enum

class SocialMessage(APIMessage):
    """
    Represents a social message between agents that may contain an optional PromptModel.
    """
    prompt: Optional[PromptModel] = Field(default=None, description="The prompt being shared between agents")

    @classmethod
    def create(cls, sender: str, receiver: str, content: str, conversation_id: str, prompt: Optional[PromptModel] = None) -> "SocialMessage":
        return cls(
            sender=sender,
            receiver=receiver,
            content=content,
            conversation_id=conversation_id,
            prompt=prompt,
            timestamp=datetime.utcnow().isoformat()
        )
