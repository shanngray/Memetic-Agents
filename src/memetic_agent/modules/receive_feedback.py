import datetime
import json
import uuid
from typing import Optional, Any, Dict
from src.log_config import log_event, log_error
from src.base_agent.type import Agent

async def receive_feedback_impl(
    agent: Agent,
    sender: str,
    conversation_id: str,
    score: int,
    feedback: str
) -> None:
    """Process and store received feedback from another agent.
    
    Args:
        agent: The agent receiving the feedback
        sender: Name of the agent providing feedback
        conversation_id: ID of the conversation being rated
        score: Numerical score (typically 0-10)
        feedback: Detailed feedback text
    """
    try:
        # Format feedback content
        formatted_feedback = (
            f"Feedback from {sender} regarding conversation {conversation_id}:\n"
            f"Score: {score}/10\n"
            f"Comments: {feedback}"
        )
        
        # Store in feedback collection with metadata
        metadata = {
            "sender": sender,
            "conversation_id": conversation_id,
            "score": score,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await agent.memory.store(
            content=formatted_feedback,
            collection_name="feedback",
            metadata=metadata
        )
        
        log_event(agent.logger, "feedback.received",
                    f"Received feedback from {sender} for conversation {conversation_id}")
                    
    except Exception as e:
        log_error(agent.logger, f"Failed to process received feedback: {str(e)}")
