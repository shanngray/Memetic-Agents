from src.base_agent.base_agent import BaseAgent
from src.base_agent.communication_module.receive_message_impl import receive_message_impl
from src.api_server.models.api_models import APIMessage
from .modules.receive_social_message import receive_social_message

async def route_incoming_message(agent: BaseAgent, sender: str, content: str, message: APIMessage) -> str:
    """Handle incoming messages, routing social messages to specialized handlers"""
    
    # Check if this is a social message (has prompt field)
    if hasattr(message, 'prompt') and message.prompt is not None:
        return await receive_social_message(agent, sender, content, message)

    else:
        return await receive_message_impl(agent, sender, content, message)
