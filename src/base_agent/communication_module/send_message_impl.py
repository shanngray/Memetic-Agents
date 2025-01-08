from datetime import datetime
import httpx
import asyncio
from typing import Dict, Any
from src.base_agent.models import AgentStatus
from src.base_agent.type import Agent
from src.log_config import log_agent_message, log_event, log_error
from src.api_server.models.api_models import APIMessage

async def send_message_impl(agent: Agent, receiver: str, content: str) -> Dict[str, Any]:
    """Send a message via API to another agent registered in the directory service."""
    await agent.set_status(AgentStatus.WAITING_RESPONSE, "waiting for agent response")
    sender = agent.config.agent_name

    if sender == receiver:
        log_event(agent.logger, "agent.error", 
                    f"Cannot send message to self: {sender} -> {receiver}", 
                    level="ERROR")
        raise ValueError(f"Cannot send message to self: {sender} -> {receiver}")
    
    # Retrieve the parent conversation ID from the context variable
    conversation_id = agent.current_conversation_id.get()  # Added to get parent ID from context

    log_agent_message(agent.logger, "out", sender, receiver, content)
    
    async with httpx.AsyncClient(timeout=3000.0) as client:
        message = APIMessage.create(
            sender=sender,
            receiver=receiver,
            content=content,
            conversation_id=conversation_id
        )
        
        try:
            log_event(agent.logger, "directory.route", 
                        f"Sending message to directory service", level="DEBUG")
            log_event(agent.logger, "directory.route", 
                        f"Request payload: {message.dict()}", level="DEBUG")
            
            response = await client.post(
                "http://localhost:8000/agent/message",
                json=message.dict(),
                timeout=3000.0
            )
            
            log_event(agent.logger, "directory.route", 
                        f"Received response: Status {response.status_code}", level="DEBUG")
            log_event(agent.logger, "directory.route", 
                        f"Response content: {response.text}", level="DEBUG")
            
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                log_error(agent.logger, f"HTTP error occurred: {str(e)}")
                log_error(agent.logger, f"Response content: {response.text}")
                raise
            
            result = response.json()
            message_content = result.get("message", "")
            if isinstance(message_content, dict):
                message_content = message_content.get("message", "")
            
            log_agent_message(agent.logger, "in", receiver, sender, message_content)
            
            # Evaluate response quality in background
            asyncio.create_task(agent._evaluate_and_send_feedback(
                receiver=receiver,
                conversation_id=conversation_id,
                response_content=message_content
            ))
            
            # After receiving response
            await agent.set_status(AgentStatus.TOOL_EXECUTING, "received agent response")
            
            return {
                "role": "assistant",
                "content": message_content,
                "sender": receiver,
                "receiver": sender,
                "timestamp": result.get("timestamp") or datetime.now().isoformat()
            }
            
        except httpx.TimeoutException as e:
            error_msg = f"Request timed out while sending message to {receiver}"
            log_error(agent.logger, error_msg, exc_info=e)
            return {
                "role": "error",
                "content": f"{error_msg}: {str(e)}",
                "sender": "system",
                "receiver": sender,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            log_error(agent.logger, f"Unexpected error in send_message", exc_info=e)
            raise
        finally:
            if agent.status == AgentStatus.MESSAGE_PROCESSING or agent.status == AgentStatus.WAITING_RESPONSE:
                await agent.set_status(AgentStatus.TOOL_EXECUTING, "completed send_message")
