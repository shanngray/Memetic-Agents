import httpx
from typing import Dict, Any
import sys
from pathlib import Path
from datetime import datetime


sys.path.append(str(Path(__file__).parents[2]))
from log_config import setup_logger

logger = setup_logger(
    name="SendMessage",
    level="DEBUG",
    console_logging=True
)

from api_server.models.api_models import APIMessage

async def send_message(sender: str, receiver: str, content: str, conversation_id: str) -> Dict[str, Any]:
    """Send a message via API to another agent registered in the directory service.
    
    Args:
        sender: Name of the sending agent
        receiver: Name of the receiving agent
        content: Message content to send
        conversation_id: Unique identifier for the conversation
        
    Returns:
        Dict containing the response formatted as a message from the receiving agent
    """
    logger.debug(f"Attempting to send message from {sender} to {receiver} in conversation {conversation_id}")
    logger.debug(f"Message content: {content}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        message = APIMessage.create(
            sender=sender,
            receiver=receiver,
            content=content,
            conversation_id=conversation_id
        )
        
        try:
            logger.debug(f"Sending POST request to directory service")
            logger.debug(f"Request payload: {message.dict()}")
            
            response = await client.post(
                "http://localhost:8000/agent/message",
                json=message.dict(),
                timeout=30.0
            )
            
            logger.debug(f"Received response: Status {response.status_code}")
            logger.debug(f"Response content: {response.text}")
            
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error occurred: {str(e)}")
                logger.error(f"Response content: {response.text}")
                raise
            
            result = response.json()
            return {
                "role": "assistant",
                "content": result.get("response", ""),
                "sender": receiver,
                "receiver": sender,
                "timestamp": result.get("timestamp") or datetime.now().isoformat()
            }
            
        except httpx.TimeoutException as e:
            logger.error(f"Timeout error: {str(e)}")
            return {
                "role": "error",
                "content": f"Request timed out while sending message to {receiver}: {str(e)}",
                "sender": "system",
                "receiver": sender,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Unexpected error in send_message: {str(e)}")
            raise
