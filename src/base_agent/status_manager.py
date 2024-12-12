import asyncio
from typing import Callable, List

class StatusManager:
    """Manages agent status transitions and validation."""
    def __init__(self, logger: Logger):
        self.logger = logger
        self.status = AgentStatus.IDLE
        self._previous_status = None
        self._status_lock = asyncio.Lock()
        
    async def set_status(self, new_status: AgentStatus) -> None:
        # Move status management logic here
        pass
        
    def validate_transition(self, current: AgentStatus, new: AgentStatus) -> bool:
        # Add transition validation logic here
        pass 