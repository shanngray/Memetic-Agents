from datetime import datetime
import asyncio
import uuid
from src.base_agent.models import AgentStatus
from src.base_agent.type import Agent
from src.log_config import log_event    
from src.base_agent.models import Message
from src.api_server.models.api_models import APIMessage

async def receive_message_impl(agent:Agent, message: APIMessage) -> str:
    """Memetic Agent process for receiving message from another agent."""
    sender = message.sender
    content = message.content
    conversation_id = message.conversation_id
    prompt = getattr(message, 'prompt', None)
    await agent.set_status(AgentStatus.MESSAGE_RECEIVED, "processing received message")
    log_event(agent.logger, "agent.message_received", 
              f"Message from {sender} in conversation {conversation_id}")
    log_event(agent.logger, "message.content", f"{content}", level="DEBUG")
    
    # Get current queue size before adding new message
    queue_size = agent.request_queue.qsize()
    queue_position = queue_size + 1
    
    loaded_conversation = agent.current_conversation_id.get()

    if conversation_id == loaded_conversation:
        # We are in the current conversation, so we can proceed with processing the message
        pass
    elif conversation_id in agent.conversations:
        # We have a short term conversation, we need to update the current conversation_id
        agent.current_conversation_id.set(conversation_id)
        pass
    elif conversation_id in agent.old_conversation_list:
        # We have a long term conversation, so we need to load it into the conversation history and update the current conversation_id
        agent.current_conversation_id.set(conversation_id)
        agent.conversations[conversation_id].append(Message(role="system", content="This is an old conversation, you may have memories in longer_term memory that will help with context."))
        pass
    else:
        # This is a new conversation, we need to update the conversation list and the current conversation_id
        log_event(agent.logger, "session.created", f"Creating new conversation: {conversation_id}")
        agent.current_conversation_id.set(conversation_id)
        agent.conversations[conversation_id] = [
            Message(role="system", content=agent._system_prompt)
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
        "prompt": prompt,
        "conversation_id": conversation_id,
        "timestamp": datetime.now().isoformat(),
        "queue_position": queue_position
    }
    log_event(agent.logger, "queue.added", 
              f"Queuing request {request_id} from {sender} (position {queue_position} in queue)")
    
    await agent.request_queue.put(request)
    await agent.set_status(AgentStatus.QUEUE_PROCESSING, "start processing queue") # Should this go before or after the request_queue.put?

    try:
        # Wait for response with timeout
        if agent.status == AgentStatus.QUEUE_PROCESSING:
            await agent.set_status(AgentStatus.WAITING_RESPONSE, "waiting for queue processing")
        log_event(agent.logger, "queue.status", 
                 f"Request {request_id} waiting at position {queue_position}", level="DEBUG")
        response = await asyncio.wait_for(future, timeout=300.0)
        log_event(agent.logger, "queue.completed", 
                 f"Request {request_id} completed successfully (was position {queue_position})")
        return response
    except asyncio.TimeoutError:
        log_event(agent.logger, "session.timeout", 
                 f"Request {request_id} timed out after 300 seconds (was position {queue_position})", 
                 level="WARNING")
        raise
    finally:
        if request_id in agent.pending_requests:
            del agent.pending_requests[request_id]
            log_event(agent.logger, "queue.status", 
                     f"Cleaned up request {request_id} (was position {queue_position})", 
                     level="DEBUG")
        if agent.status != AgentStatus.SHUTTING_DOWN:
            await agent.set_status(AgentStatus.IDLE, "completed message processing")


