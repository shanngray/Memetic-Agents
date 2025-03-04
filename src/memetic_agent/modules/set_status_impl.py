from src.base_agent.type import Agent
from src.log_config import log_error, log_event
from datetime import datetime
import asyncio
from src.base_agent.models import AgentStatus

async def set_status_impl(agent: Agent, new_status: AgentStatus, trigger: str) -> None:
    """Memetic Agent version of BaseAgent set_status includes learning from memory."""
    try:
        if new_status == agent.status:
            log_event(agent.logger, "status.change", 
                        f"Status unchanged - Current: /{agent.status.name}, "
                        f"New: {new_status.name}, Trigger: {trigger}")
            return

        valid_transitions = AgentStatus.get_valid_transitions(agent.status)
        
        if new_status not in valid_transitions:
            log_event(agent.logger, "status.change", 
                        f"Invalid status transition - Current: /{agent.status.name}, "
                        f"New: {new_status.name}, Trigger: {trigger}", level="ERROR")
            raise ValueError(
                f"Invalid status transition from /{agent.status.name} to /{new_status.name} caused by {trigger}"
            )

        # Store previous status before updating
        previous_status = agent.status
        
        # Update status
        agent.status = new_status
        agent._previous_status = previous_status
        agent._status_history.append({
            "timestamp": datetime.now().isoformat(),
            "from": previous_status,
            "to": agent.status,
            "trigger": trigger
        })
        log_event(
            agent.logger,
            f"agent.{agent.status.name}",
            f"Status changed: /{previous_status.name} -> /{agent.status.name} ({trigger})"
        )
        
        # Handle MEMORISING state tasks
        if new_status == AgentStatus.MEMORISING and previous_status != AgentStatus.MEMORISING:
            try:
                # Create tasks but store their references
                transfer_task = asyncio.create_task(agent._transfer_to_long_term())
                learning_task = asyncio.create_task(agent._extract_learnings())
                
                # Wait for both tasks to complete before cleanup
                await asyncio.gather(transfer_task, learning_task)
                
                # Now run cleanup
                clean_short_task = asyncio.create_task(agent._cleanup_memories(days_threshold=0, collection_name="short_term"))
                clean_feedback_task = asyncio.create_task(agent._cleanup_memories(days_threshold=0, collection_name="feedback"))
                await asyncio.gather(clean_short_task, clean_feedback_task)
            finally:
                # Direct status update without recursive call
                if agent.status != AgentStatus.SHUTTING_DOWN:
                    agent.status = previous_status
                    log_event(
                        agent.logger,
                        f"agent.{agent.status.name}",
                        f"Status restored: {AgentStatus.MEMORISING.name} -> {agent.status.name} (Memory processing complete)"
                    )

        if new_status == AgentStatus.LEARNING:
            try:
                await agent._run_learning_subroutine()
                if agent.status != AgentStatus.SHUTTING_DOWN:
                    await agent.set_status(AgentStatus.AVAILABLE, "Learning complete")
            except Exception as e:
                agent.logger.error(f"Learning failed: {str(e)}")
                if agent.status != AgentStatus.SHUTTING_DOWN:
                    await agent.set_status(AgentStatus.AVAILABLE, "Learning failed")

        if new_status == AgentStatus.SOCIALISING:
            try:
                # Create task for socializing but don't await it
                asyncio.sleep(0.1)
                asyncio.create_task(agent._start_socialising())
            except Exception as e:
                agent.logger.error(f"Failed to start socialising task: {str(e)}")
                if agent.status != AgentStatus.SHUTTING_DOWN:
                    await agent.set_status(AgentStatus.AVAILABLE, "Failed to start socialising")
            
        if new_status == AgentStatus.SLEEPING:
            try:
                # Create task for sleeping but don't await it
                asyncio.sleep(0.1)
                await agent._start_sleeping()
                if agent.status != AgentStatus.SHUTTING_DOWN:
                    await agent.set_status(AgentStatus.AVAILABLE, "Sleeping complete")
            except Exception as e:
                agent.logger.error(f"Failed to start sleeping task: {str(e)}")
                if agent.status != AgentStatus.SHUTTING_DOWN:
                    await agent.set_status(AgentStatus.AVAILABLE, "Failed to start sleeping")
    except Exception as e:
        agent.logger.error(f"Status update failed: {str(e)}")