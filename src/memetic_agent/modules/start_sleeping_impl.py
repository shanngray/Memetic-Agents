from src.base_agent.type import Agent
from src.log_config import log_event, log_error
import asyncio

async def _start_sleeping_impl(agent: Agent) -> None:
    """Start the sleeping subroutine.
    
    Performs cleanup similar to shutdown but preserves agent state:
    1. Saves current memory state
    2. Cancels pending requests
    3. Clears request queues
    4. Returns agent to AVAILABLE status
    """
    try:
        log_event(agent.logger, "agent.sleeping", "Starting sleep routine")
        
        # Save current memory state
        try:
            await agent._save_memory()
            log_event(agent.logger, "memory.saved", "Saved memory state while sleeping")
        except Exception as e:
            log_error(agent.logger, f"Failed to save memory during sleep: {str(e)}", exc_info=e)

        # Cancel any pending requests
        cancelled_count = 0
        for future in agent.pending_requests.values():
            if not future.done():
                future.cancel()
                cancelled_count += 1
        if cancelled_count > 0:
            log_event(agent.logger, "queue.cleanup", f"Cancelled {cancelled_count} pending requests while asleep")

        # Clear request queues
        cleared_count = 0
        while not agent.request_queue.empty():
            try:
                agent.request_queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        if cleared_count > 0:
            log_event(agent.logger, "queue.cleanup", f"Cleared {cleared_count} queued requests while asleep")

        # Return to AVAILABLE status
        if agent.status != agent.AgentStatus.SHUTTING_DOWN:
            await agent.set_status(agent.AgentStatus.AVAILABLE, "Sleep routine complete")
            log_event(agent.logger, "agent.available", "Agent returned to available state after sleep")

    except Exception as e:
        log_error(agent.logger, f"Error during sleep routine: {str(e)}", exc_info=e)
        # Ensure agent returns to available state even if there's an error
        if agent.status != agent.AgentStatus.SHUTTING_DOWN:
            await agent.set_status(agent.AgentStatus.AVAILABLE, "Sleep routine failed")