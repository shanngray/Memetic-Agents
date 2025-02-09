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
    """
    Represents a prompt model that can be used by an agent.
    """
    prompt: str = Field(..., description="The prompt to be used")
    prompt_type: str = Field(..., description="The type of prompt")
    uuid: str = Field(..., description="The unique identifier for the prompt")
    timestamp: str = Field(..., description="The timestamp that the PromptModel was created")
    owner_agent_name: str = Field(..., description="The name of the agent that owns the prompt")
    status: str = Field(..., description="The status of the prompt") #may want to make this an enum

class PromptEvaluation(BaseModel):
    """
    Represents a feedback score and evaluation for a prompt.
    """
    score: int = Field(..., description="The score of the prompt")
    evaluation: str = Field(..., description="The evaluation of the prompt")
    prompt_type: str = Field(..., description="The type of prompt being evaluated")
    uuid: str = Field(..., description="The unique identifier for the prompt being evaluated (same as the prompt model uuid)")

class SocialMessage(APIMessage):
    """
    Represents a social message between agents that may contain a PromptModel and/or a PromptEvaluation.
    A SocialMessage can be created with either a prompt or an evaluation, depending on the message type.
    """
    prompt: Optional[PromptModel] = Field(None, description="The prompt being shared between agents")
    evaluation: Optional[PromptEvaluation] = Field(None, description="The evaluation of the prompt")
    message_type: str = Field(..., description="The type of message", examples=["InitialPrompt", "EvalResponse", "PromptUpdate", "UpdateResponse", "FinalEval"])

    @classmethod
    def create_with_prompt(cls, sender: str, receiver: str, content: str, conversation_id: str, prompt: PromptModel, message_type: str) -> "SocialMessage":
        """
        Factory method to create an initial or updated social message with a required prompt and current timestamp.
        
        Args:
            sender: Identifier of the sending agent
            receiver: Identifier of the receiving agent
            content: Message content
            conversation_id: Unique conversation identifier
            prompt: PromptModel instance that will be shared
            message_type: The type of message should always be "InitialPrompt"
        Returns:
            SocialMessage: Newly created message instance with prompt
        """
        return cls(
            sender=sender,
            receiver=receiver,
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=conversation_id,
            prompt=prompt,
            message_type=message_type
        )

    @classmethod
    def create_with_prompt_and_eval(cls, sender: str, receiver: str, content: str, conversation_id: str, prompt: PromptModel, evaluation: PromptEvaluation, message_type: str) -> "SocialMessage":
        """
        Factory method to create an initial or updated social message with a required prompt and evaluation and current timestamp.
        """
        return cls(
            sender=sender,
            receiver=receiver,
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=conversation_id,
            prompt=prompt,
            evaluation=evaluation,
            message_type=message_type
        )

    @classmethod
    def create_with_eval(cls, sender: str, receiver: str, content: str, conversation_id: str, evaluation: PromptEvaluation, message_type: str) -> "SocialMessage":
        """
        Factory method to create a response or final social message with a required evaluation and current timestamp.
        
        Args:
            sender: Identifier of the sending agent
            receiver: Identifier of the receiving agent
            content: Message content
            conversation_id: Unique conversation identifier
            evaluation: PromptEvaluation instance that will be shared
            message_type: The type of message always "FinalEval"
        Returns:
            SocialMessage: Newly created message instance with evaluation
        """
        return cls(
            sender=sender,
            receiver=receiver,
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            conversation_id=conversation_id,
            evaluation=evaluation,
            message_type=message_type
        )
