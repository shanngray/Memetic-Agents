from src.base_agent.type import Agent
from src.base_agent.models import AgentStatus
from src.log_config import log_event, log_error

async def process_queue_impl(agent: Agent):
    """Base Agent process for processing any pending requests in the queue"""
    if agent.status == AgentStatus.SHUTTING_DOWN:
        return

    if not agent.request_queue.empty():
        current_queue_size = agent.request_queue.qsize()
        request = await agent.request_queue.get()
        queue_position = request.get('queue_position', 'unknown')
        
        try:
            log_event(agent.logger, "queue.dequeued", 
                        f"Processing request {request['id']} from {request['sender']} "
                        f"(position {queue_position}/{current_queue_size} in queue)")
            
            log_event(agent.logger, "queue.processing", 
                        f"Processing message content: {request['content']} "
                        f"(was position {queue_position})", level="DEBUG")
            response = await agent.process_message(
                content=request["content"],
                sender=request["sender"],
                conversation_id=request["conversation_id"]
            )
            
            # Complete the future with the response
            if request["id"] in agent.pending_requests:
                agent.pending_requests[request["id"]].set_result(response)
                log_event(agent.logger, "queue.completed", 
                            f"Request {request['id']} processed successfully "
                            f"(was position {queue_position}/{current_queue_size})")
                del agent.pending_requests[request["id"]]
                
        except Exception as e:
            log_error(agent.logger, 
                        f"Error processing request {request['id']} "
                        f"(was position {queue_position}/{current_queue_size}): {str(e)}", 
                        exc_info=e)
            if request["id"] in agent.pending_requests:
                agent.pending_requests[request["id"]].set_exception(e)
                del agent.pending_requests[request["id"]]
