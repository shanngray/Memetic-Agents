from base_agent.type import Agent
from base_agent.models import AgentStatus
from log_config import log_event, log_error
import time
import asyncio

async def process_queue_impl(agent: Agent):
    """Memetic Agent process for processing any pending requests in the queue"""
    try:
        # Queue state logging
        current_time = time.time()
        current_size = agent.request_queue.qsize()
        
        if not hasattr(agent, '_last_queue_log'):
            agent._last_queue_log = 0
            agent._last_logged_size = -1
        
        # Log only if queue size changed or 60 seconds elapsed
        if current_size != agent._last_logged_size or (current_time - agent._last_queue_log) >= 60:
            log_event(agent.logger, "queue.state", 
                     f"Process queue called - Status: {agent.status.name}, Queue size: {current_size}")
            agent._last_queue_log = current_time
            agent._last_logged_size = current_size
        
        if agent.status == AgentStatus.SHUTTING_DOWN:
            log_event(agent.logger, "queue.state", "Agent shutting down, skipping queue processing")
            return

        # Yield control before checking queue
        await asyncio.sleep(0.1)

        if not agent.request_queue.empty():
            request = None
            try:
                request = await agent.request_queue.get()
                request_id = request.get('id')
                queue_position = request.get('queue_position', 'unknown')
                current_queue_size = agent.request_queue.qsize()
                
                log_event(agent.logger, "queue.processing", 
                         f"Got request from queue - ID: {request_id}, Type: {request.get('message_type')}, "
                         f"Status: {agent.status.name}")
                
                log_event(agent.logger, "queue.debug", 
                         f"Processing request type: {request.get('message_type', 'default')}, "
                         f"Prompt: {request.get('prompt') is not None}, "
                         f"Evaluation: {request.get('evaluation') is not None}", level="DEBUG")
                
                log_event(agent.logger, "queue.dequeued", 
                         f"Processing request {request_id} from {request['sender']} "
                         f"(position {queue_position}/{current_queue_size} in queue)", level="DEBUG")
                
                log_event(agent.logger, "queue.processing", 
                         f"Processing message content: {request['content']} "
                         f"(was position {queue_position})", level="DEBUG")

                # Yield control before heavy processing
                await asyncio.sleep(0.1)

                response = await agent.process_message(
                    content=request["content"],
                    sender=request["sender"],
                    prompt=request.get("prompt"),
                    evaluation=request.get("evaluation"),
                    message_type=request.get("message_type"),
                    conversation_id=request["conversation_id"],
                    request_id=request_id
                )
                
                # Set the future result so receive_message_impl gets the response
                if request["id"] in agent.pending_requests:
                    agent.pending_requests[request["id"]].set_result(response)
                    log_event(agent.logger, "queue.completed", 
                             f"Request {request['id']} processed successfully")
                    del agent.pending_requests[request["id"]]

                # Yield control after heavy processing
                await asyncio.sleep(0.1)

                log_event(agent.logger, "queue.debug", 
                         f"Response received for request {request_id}: {response[:100]}...", level="DEBUG")

            except Exception as e:
                log_error(agent.logger, 
                         f"Error processing request {request['id'] if request else 'unknown'}: {str(e)}", 
                         exc_info=e)
                # Set exception on future if there's an error
                if request and request["id"] in agent.pending_requests:
                    agent.pending_requests[request["id"]].set_exception(e)
                    del agent.pending_requests[request["id"]]
            finally:
                if request:
                    agent.request_queue.task_done()
                    if agent.status != AgentStatus.SHUTTING_DOWN:
                        log_event(agent.logger, "agent.status", 
                                 f"Agent Status changed from {agent.status.name} to AVAILABLE", 
                                 level="DEBUG")
                        await agent.set_status(AgentStatus.AVAILABLE, "completed message processing")
                
    except Exception as e:
        log_error(agent.logger, f"Critical error in process_queue: {str(e)}", exc_info=e)