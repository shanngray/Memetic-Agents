from datetime import datetime
import asyncio
import uuid
from src.base_agent.type import Agent
from src.log_config import log_event    
from src.base_agent.models import Message
from src.api_server.models.api_models import SocialMessage, PromptModel, PromptEvaluation

async def receive_social_message_impl(agent:Agent, message: SocialMessage) -> str:
    """Process for receiving a social message, determining the age of the conversation and adding the message to the queue.
    This method also update the current conversation. This may be problematic with concurrent conversations depending on how the queue is handled.
    TODO: Might make more sense to record the conversation id with the message in the queue and load the conversation once the message is processed."""
    sender = message.sender
    content = message.content
    conversation_id = message.conversation_id
    
    log_event(agent.logger, "agent._message_received", 
              f"Receiving social message - Type: {message.message_type}, "
              f"Has Prompt: {message.prompt is not None}, "
              f"Has Evaluation: {message.evaluation is not None}")
    
    # Get current queue size before adding new message
    queue_size = agent.request_queue.qsize()
    queue_position = queue_size + 1
    
    loaded_conversation = agent.current_conversation_id.get()
    
    log_event(agent.logger, "social.debug", 
              f"Queue state - Size: {queue_size}, Position: {queue_position}, "
              f"Current conv: {loaded_conversation}, New conv: {conversation_id}")

    if conversation_id == loaded_conversation:
        # We are in the current conversation, so we can proceed with processing the message
        log_event(agent.logger, "social.conversation.current", f"We are in the current social conversation: {conversation_id}")
        pass
    elif conversation_id in agent.conversations:
        # We have a short term conversation, we need to update the current conversation_id
        agent.current_conversation_id.set(conversation_id)
        log_event(agent.logger, "social.conversation.short", f"We have a short term social conversation: {conversation_id}")
        pass
    elif conversation_id in agent.old_conversation_list:
        # We have a long term conversation, so we need to load it into the conversation history and update the current conversation_id
        agent.current_conversation_id.set(conversation_id)
        agent.conversations[conversation_id].append(Message(
            role="user" if agent.config.model == "o1-mini" else "developer" if agent.config.model == "o3-mini" else "system", 
            content="This is an old conversation, you may have memories in longer_term memory that will help with context."
            ))
        log_event(agent.logger, "social.conversation.long", f"We have a long term social conversation: {conversation_id}")
        pass
    else:
        # This is a new conversation, we need to update the conversation list and the current conversation_id
        log_event(agent.logger, "social.conversation.new", f"Creating new social conversation: {conversation_id}")
        agent.current_conversation_id.set(conversation_id)
        agent.conversations[conversation_id] = [
            Message(
                role="user" if agent.config.model == "o1-mini" else "developer" if agent.config.model == "o3-mini" else "system", 
                content=agent._system_prompt
                )
        ]
        pass

    # Create request ID and future
    request_id = str(uuid.uuid4())
    future = asyncio.Future()
    agent.pending_requests[request_id] = future
    
    # Queue the request
    request = {
        "id": request_id,
        "sender": sender,
        "content": content,
        "prompt": message.prompt,
        "evaluation": message.evaluation,
        "message_type": message.message_type,
        "conversation_id": conversation_id,
        "timestamp": datetime.now().isoformat(),
        "queue_position": queue_position
    }
    log_event(agent.logger, "queue.added", 
              f"Queuing social request {request_id} from {sender} (position {queue_position} in queue)")
    
    await agent.request_queue.put(request)

    try:
        # Wait for response with timeout
        log_event(agent.logger, "queue.status", 
                 f"Social request {request_id} waiting at position {queue_position}", level="DEBUG")
        response = await asyncio.wait_for(future, timeout=300.0)
        log_event(agent.logger, "queue.completed", 
                 f"Social request {request_id} completed successfully (was position {queue_position})")
        return response
    except asyncio.TimeoutError:
        log_event(agent.logger, "session.timeout", 
                 f"Social request {request_id} timed out after 300 seconds (was position {queue_position})", 
                 level="WARNING")
        raise asyncio.TimeoutError
    finally:
        if request_id in agent.pending_requests:
            del agent.pending_requests[request_id]
            log_event(agent.logger, "queue.status", 
                     f"Cleaned up social request {request_id} (was position {queue_position})", 
                     level="DEBUG")


